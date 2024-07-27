import gymnasium as gym
import pygame
import numpy as np
import random
import math
import time
import logging
from gymnasium import spaces
from gymnasium.envs.registration import register
from collections import defaultdict

# Register the custom environment with communications
register(
    id="CombinedArmsRLComms-v1",
    entry_point="combined_arms_rl_env_MA_comms:CombinedArmsRLEnvMAComms",
)

# Agent attributes dictionary
AGENT_ATTRIBUTES = {
    "artillery": {
        "health": 150,
        "damage": (100, 25),  # Far, Near
        "observation_range": 6,
        "engagement_range": 16,
        "model": "model_artillery"
    },
    "infantry": {
        "health": 100,
        "damage": 25,
        "observation_range": 6,
        "engagement_range": 4,
        "model": "model_infantry"
    },
    "scout": {
        "health": 75,
        "damage": 15,
        "observation_range": 22,
        "engagement_range": 4,
        "model": "model_scout"
    },
    "light_tank": {
        "health": 200,
        "damage": 50,
        "observation_range": 8,
        "engagement_range": 6,
        "model": "model_light_tank"
    },
    "commander": {
        "health": 200,
        "damage": 25,
        "observation_range": 6,
        "engagement_range": 4,
        "model": "model_commander"
    }
}

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class CombinedArmsRLEnvMAComms(gym.Env):
    """
    A multi-agent reinforcement learning environment for simulating combined arms operations with communication.

    This environment simulates a battlefield with multiple types of units (scouts, commanders, artillery, etc.) from
    two opposing teams. It incorporates a communication system where scouts can relay enemy positions to commanders,
    who then communicate this information to artillery units.

    State Space:
        The state is represented as a 3D tensor with shape (grid_width, grid_height, channels).
        Channels:
        - Channel 0: Positions of friendly units
        - Channel 1: Positions of enemy units
        - Channel 2: Communication information (e.g., communicated enemy positions)

    Attributes:
        friendly_agents (list of dict): List of dictionaries representing friendly agents.
            Format: [{"id": str, "type": str, "team": str, "position": [int, int], ...}, ...]
        enemy_agents (list of dict): List of dictionaries representing enemy agents.
            Format: [{"id": str, "type": str, "team": str, "position": [int, int], ...}, ...]
        friendly_agent_ids (list): List of IDs for friendly agents.
            Format: [str, str, ...]
        enemy_agent_ids (list): List of IDs for enemy agents.
            Format: [str, str, ...]
        communication_data (dict): Contains all communication-related information:
            - network (defaultdict): Maps receiver IDs to lists of sender IDs.
                Format: {receiver_id: [sender_id1, sender_id2, ...]}
            - detected_enemies (defaultdict): Maps agent IDs to lists of detected enemy information.
                Format: {agent_id: [{"id": str, "position": [int, int], "health": int}, ...]}
            - last_communicated (dict): Maps agent IDs to their last communicated state.
                Format: {agent_id: frozenset(((enemy_id, (x, y)), ...))}
            - targets (dict): Maps enemy IDs to their communicated positions.
                Format: {enemy_id: [int, int]}
            - artillery_destructions (dict): Tracks artillery destructions for the current step.
                Format: {artillery_id: int}
            - episode_destructions (dict): Tracks artillery destructions for the entire episode.
                Format: {artillery_id: int}
        episode_comm_stats (dict): Tracks communication statistics for the episode.
            Format: {
                'scout_to_commander': {scout_id: int, ...},
                'commander_to_artillery': {commander_id: int, ...},
                'artillery_destroyed_enemies': {artillery_id: int, ...}
            }
        reward_info (dict): Contains reward-related information for the current step.
            Format: {
                'friendly': {
                    'individual': [float, ...],
                    'team': float,
                    'strength': float,
                    'destruction': float
                },
                'enemy': {
                    'individual': [float, ...],
                    'team': float,
                    'strength': float,
                    'destruction': float
                }
            }
        episode_rewards (dict): Tracks cumulative rewards for the entire episode.
            Format: {
                'friendly': {
                    'individual': [float, ...],
                    'team': float,
                    'total': float
                },
                'enemy': {
                    'individual': [float, ...],
                    'team': float,
                    'total': float
                }
            }
        current_step (int): Current step in the episode.
        total_steps (int): Total steps taken across all episodes.
        fired_agents (list): List of agents who fired in the current step.
            Format: [{"id": str, "type": str, "team": str, ...}, ...]

    Helper Functions:
        _create_agents(config, team): Create agents based on the provided configuration.
        _get_default_position(team): Get a default position for an agent of the specified team.
    step(actions): Perform one step in the environment.
        _detect_enemies(agent): Detect enemies within the observation range of the given agent.
        _communicate(agent): Handle the communication process for the given agent.
        _process_communication(): Process all communications for the current step.
        _move_agent(agent, action): Move an agent based on the given action.
        _fire_weapon(agent, target=None): Handle weapon firing for a given agent.
        _calculate_damage(agent, distance): Calculate the damage dealt by an agent at a given distance.
        _get_observation(): Get the current observation of the environment.
        _calculate_reward(): Calculate rewards for all agents based on their actions and the current state.
        _check_done(): Check if the episode is finished.
    reset(): Reset the environment to its initial state.
    render(): Render the current state of the environment.
    close(): Close the environment and release resources.

    Each of these helper functions plays a crucial role in simulating the combined arms environment with communication:

    - Agent creation and positioning are handled by _create_agents and _get_default_position.
    - The core gameplay loop involves detection (_detect_enemies), communication (_communicate
      and _process_communication), movement (_move_agent), and combat (_fire_weapon and _calculate_damage).
    - Observation and reward calculation are managed by _get_observation and _calculate_reward, with supporting
      functions for team-level calculations.
    - The environment state is controlled through step, reset, render, and close methods.

    These functions work together to create a complex, multi-agent environment that simulates combined arms operations
    with a focus on communication and coordination between different types of units.

    Note:
        Each agent is represented as a dictionary with keys including 'id', 'type', 'team', 'position',
        'health', 'detected_enemies', 'last_communicated_state', etc.
    """

    def __init__(self, friendly_config, enemy_config, render_speed=0.05, max_steps=1500, render_mode='human'):
        """
        Initialize the environment.

        Args:
            friendly_config (dict): Configuration for friendly team.
            enemy_config (dict): Configuration for enemy team.
            render_speed (float): Speed of rendering.
            max_steps (int): Maximum number of steps per episode.
            render_mode (str): Mode for rendering ('human' or 'rgb_array').
        """
        super(CombinedArmsRLEnvMAComms, self).__init__()

        # Environment geometry
        self.screen_width, self.screen_height = 1750, 750
        self.grid_width, self.grid_height = 64, 10
        self.cell_size = 25
        self.padding = 20

        # Rendering attributes
        self.render_speed = render_speed
        self.render_mode = render_mode

        # Episode parameters
        self.max_steps = max_steps
        self.current_step = 0
        self.total_steps = 0

        # Team configurations
        self.friendly_config = friendly_config
        self.enemy_config = enemy_config
        self.initial_friendly_count = sum(agent_config["count"] for agent_config in friendly_config.values())
        self.initial_enemy_count = sum(agent_config["count"] for agent_config in enemy_config.values())

        # Action and observation spaces
        self.action_space = spaces.Discrete(11)  # 0-7: move, 8: fire, 9: communicate, 10: hold
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.grid_width, self.grid_height, 3),
            dtype=np.uint8
        )

        # Initialize agents
        self.friendly_agents = self._create_agents(self.friendly_config, team="friendly")
        self.enemy_agents = self._create_agents(self.enemy_config, team="enemy")
        self.friendly_agent_ids = [agent["id"] for agent in self.friendly_agents]
        self.enemy_agent_ids = [agent["id"] for agent in self.enemy_agents]

        # Initialize reward-related attributes
        self.reward_info = None
        self.episode_rewards = {
            'friendly': {
                'individual': [0] * len(self.friendly_agents),
                'team': 0,
                'total': 0
            },
            'enemy': {
                'individual': [0] * len(self.enemy_agents),
                'team': 0,
                'total': 0
            }
        }

        # Firing state tracking
        self.fired_agents = []

        # Communication-related attributes
        self.communication_data = {
            "network": defaultdict(list),
            "detected_enemies": defaultdict(list),
            "last_communicated": {},
            "targets": {},
            "artillery_destructions": {agent['id']: 0 for agent in self.friendly_agents if
                                       agent['type'] == 'artillery'},
            "episode_destructions": {agent['id']: 0 for agent in self.friendly_agents if agent['type'] == 'artillery'}
        }

        # Communication statistics
        self.episode_comm_stats = {
            'scout_to_commander': {agent['id']: 0 for agent in self.friendly_agents if agent['type'] == 'scout'},
            'commander_to_artillery': {agent['id']: 0 for agent in self.friendly_agents if
                                       agent['type'] == 'commander'},
            'artillery_destroyed_enemies': {agent['id']: 0 for agent in self.friendly_agents if
                                            agent['type'] == 'artillery'}
        }

        # Initialize Pygame for rendering
        if self.render_mode == 'human':
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Combined Arms RL Environment - Multi Agent")
        else:
            self.screen = None

    def _create_agents(self, config, team):
        """
        Create agents based on the provided configuration.

        Args:
            config (dict): Configuration dictionary for the team.
            team (str): Team identifier ('friendly' or 'enemy').

        Returns:
            list: List of agent dictionaries.
        """
        agents = []
        for agent_type, agent_config in config.items():
            for i in range(agent_config["count"]):
                position = agent_config["positions"][i] if i < len(
                    agent_config["positions"]) else self._get_default_position(team)
                position = self.ensure_position_in_bounds(position)
                agent_id = f"{team}_{agent_type}_{i}"
                agent = {
                    "id": agent_id,
                    "type": agent_type,
                    "team": team,
                    "position": position,
                    **AGENT_ATTRIBUTES[agent_type],
                    "model": agent_config.get("model", AGENT_ATTRIBUTES[agent_type]["model"]),
                    "fired": False,
                    "target_pos": None,
                    "hit": False,
                    "just_died": False,
                    "just_damaged": False,
                    "detected": False,
                    "detected_enemies": [],
                    "last_communicated_state": set(),
                    "can_communicate": agent_type in ["scout", "commander", "artillery"],
                    "last_communicated": -1,
                    "communication_cooldown": 5
                }
                agents.append(agent)
        return agents

    def _get_default_position(self, team):
        """
        Get a default position for an agent of the specified team.

        Args:
            team (str): Team identifier ('friendly' or 'enemy').

        Returns:
            list: Default position [row, col].
        """
        if team == "friendly":
            return [np.random.randint(0, self.grid_height), np.random.randint(0, min(5, self.grid_width - 1))]
        else:
            return [np.random.randint(0, self.grid_height),
                    np.random.randint(max(0, self.grid_width - 5), self.grid_width - 1)]

    def ensure_position_in_bounds(self, position):
        return [
            max(0, min(position[0], self.grid_height - 1)),
            max(0, min(position[1], self.grid_width - 1))
        ]

    def step(self, actions):
        """
        Perform one step in the environment.

        Args:
            actions (list): List of actions for each agent.

        Returns:
            observation (dict): Current observation of the environment.
            reward (dict): Rewards for each agent.
            done (bool): Whether the episode has ended.
            info (dict): Additional information about the step.
        """

        self.current_step += 1
        self.total_steps += 1

        self.fired_agents.clear()
        self.communication_data["network"].clear()
        for agent in self.friendly_agents + self.enemy_agents:
            agent["fired"] = False
            agent["hit"] = False

        # Reset artillery communicated destructions for this step
        for artillery_id in self.communication_data["artillery_destructions"]:
            self.communication_data["artillery_destructions"][artillery_id] = 0

        all_agents = self.friendly_agents + self.enemy_agents

        for agent, action in zip(all_agents, actions):
            if agent["health"] > 0:
                self._detect_enemies(agent)
                if action < 8:  # Move actions
                    self._move_agent(agent, action)
                elif action == 8:  # Fire action
                    self._fire_weapon(agent)
                elif action == 9:  # Communicate action
                    if (agent["can_communicate"] and
                            (self.current_step - agent["last_communicated"]) > agent["communication_cooldown"]):
                        self._communicate(agent)
                        agent["last_communicated"] = self.current_step
                elif action == 10:  # Hold position
                    pass

        # Allow artillery to act on communicated information
        for agent in self.friendly_agents:
            if agent["type"] == "artillery" and agent["health"] > 0:
                self._fire_weapon(agent)

        # Calculate rewards using the new unified function
        reward_info = self._calculate_reward()

        # Update the reward_info attribute
        self.reward_info = reward_info

        # Combine individual and team rewards for learning
        learning_rewards = {
            'friendly': [ind + reward_info['friendly']['team'] / len(self.friendly_agents) for ind in
                         reward_info['friendly']['individual']],
            'enemy': [ind + reward_info['enemy']['team'] / len(self.enemy_agents) for ind in
                      reward_info['enemy']['individual']]
        }

        # Check if the episode is done
        done = self._check_done()

        # Get the new observation
        observation = self._get_observation()

        # Update episode_comm_stats with artillery communicated destructions
        for artillery_id, count in self.communication_data["artillery_destructions"].items():
            self.communication_data["episode_destructions"][artillery_id] += count

        # Reset artillery_destructions for the next step
        self.communication_data["artillery_destructions"] = {artillery_id: 0 for artillery_id in
                                                             self.communication_data["artillery_destructions"]}

        return observation, learning_rewards, done, {
            "friendly_reward": self.episode_rewards['friendly']['total'],
            "enemy_reward": self.episode_rewards['enemy']['total'],
            "friendly_strength": reward_info['friendly']['strength'],
            "friendly_destruction": reward_info['friendly']['destruction'],
            "enemy_strength": reward_info['enemy']['strength'],
            "enemy_destruction": reward_info['enemy']['destruction'],
            "episode_rewards": self.episode_rewards,
            "step_rewards": reward_info,
            "comm_destructions": self.communication_data["episode_destructions"]
        }

    def _detect_enemies(self, agent):
        """
        Detects enemies within the observation range of the given agent.

        This function checks for all opposing agents within the observation range of the given agent and updates the
        agent's detection status and list of detected enemies.

        Args:
            agent (dict): A dictionary representing the agent performing the detection.
                          Must contain keys: 'team', 'position', 'observation_range'.

        Returns:
            bool: True if any enemies were detected, False otherwise.

        Side effects:
            - Updates the agent's 'detected' status (bool).
            - Updates the agent's 'detected_enemies' list (list of dicts).
            - Updates the self.detected_enemies dictionary if enemies were detected.

        Note:
            The 'detected_enemies' list and self.detected_enemies dictionary are reset each time this function is
            called, only containing enemies currently in range.
        """
        logging.debug(f"\nDetermining if enemies can be detected for {agent['id']} (team: {agent['team']}):")

        # Initializing which agents are on the opposite team of the agent by referencing each agent's attributes,
        # specifically the 'team' attribute.
        opponents = self.enemy_agents if agent['team'] == 'friendly' else self.friendly_agents

        # Creating two more attributes for the agent to signify if an agent "detected" opposing agent (boolean),
        # and what opposing agents the agent has detected (list).
        agent["detected"] = False
        agent["detected_enemies"] = []

        logging.debug(
            f"  Initial state: If enemies have been detected? = {agent['detected']}, "
            f"What enemies were detected = {agent['detected_enemies']}")

        # Analyzing each enemy defined as an opponent.
        for enemy in opponents:
            # Skipping destroyed enemies.
            if enemy['health'] <= 0:
                continue

            # Determining distance and whether the enemy can be observed according to the agent's observation range.
            distance = np.linalg.norm(np.array(agent['position']) - np.array(enemy['position']))
            logging.debug(f"  Checking enemy {enemy['id']} at distance {distance:.2f}")

            if distance <= agent['observation_range']:
                agent["detected"] = True
                agent["detected_enemies"].append(enemy)
                logging.debug(f"    {agent['id']} detected {enemy['id']} at {enemy['position']} and added to "
                              f"detected_enemies list")

        # Adding what enemies were detected to a separate dictionary "self.detected_enemies"
        if agent["detected"]:
            self.communication_data["detected_enemies"][agent["id"]] = [
                {"id": enemy["id"], "position": enemy["position"], "health": enemy["health"]}
                for enemy in agent["detected_enemies"]
            ]
            logging.debug(
                f"  Updated self.detected_enemies[{agent['id']}] with {len(agent['detected_enemies'])} enemies")
        else:
            logging.debug(f"  No enemies detected for {agent['id']}")

        logging.debug(
            f"  Final state: detected = {agent['detected']}, detected_enemies count = {len(agent['detected_enemies'])}")

        return agent["detected"]

    def _communicate(self, agent):
        """
        Handle the communication process for the given agent.

        Args:
            agent (dict): The agent attempting to communicate.

        Side effects:
            Updates the communication_data dictionary with new information.
            Triggers communication processing for commanders.
        """
        if agent["type"] == "scout" and agent['team'] == 'friendly':
            # Create a frozen set of detected enemy positions
            current_state = frozenset((enemy["id"], tuple(enemy["position"])) for enemy in agent["detected_enemies"] if
                                      enemy['team'] != agent['team'])

            # Check if the current state is different from the last communicated state
            if current_state != self.communication_data["last_communicated"].get(agent["id"], frozenset()):
                # Find living commanders to communicate with
                living_commanders = [commander for commander in self.friendly_agents
                                     if commander["type"] == "commander" and commander["health"] > 0]

                if living_commanders:
                    # Communicate with all living commanders
                    for commander in living_commanders:
                        self.communication_data["network"][commander["id"]].append(agent["id"])
                        self.episode_comm_stats['scout_to_commander'][agent['id']] += 1
                        logging.info(f"Scout {agent['id']} communicated with Commander {commander['id']}")

                    # Update the last communicated state and detected enemies
                    self.communication_data["last_communicated"][agent["id"]] = current_state
                    self.communication_data["detected_enemies"][agent["id"]] = [
                        {"id": enemy["id"], "position": enemy["position"], "health": enemy["health"]}
                        for enemy in agent["detected_enemies"]
                        if enemy['team'] != agent['team']
                    ]
                else:
                    logging.info(f"Scout {agent['id']} attempted to communicate but no living commanders available")

        elif agent["type"] == "commander" and agent['team'] == 'friendly':
            # Create a frozen set of all current enemy positions known to this commander
            current_state = frozenset(
                (enemy["id"], tuple(enemy["position"]))
                for scout_id in self.communication_data["network"].get(agent["id"], [])
                for enemy in self.communication_data["detected_enemies"].get(scout_id, [])
            )

            # Check if the current state is different from the last communicated state
            if current_state != self.communication_data["last_communicated"].get(agent["id"], frozenset()):
                # Process and communicate the new information
                self._process_communication(agent["id"])
                # Update the last communicated state for this commander
                self.communication_data["last_communicated"][agent["id"]] = current_state
                logging.info(f"Commander {agent['id']} processed and communicated new information")
            else:
                logging.info(f"Commander {agent['id']} attempted to communicate but had no new information")

    def _process_communication(self, commander_id):
        """
        Process communications for a specific commander.

        Args:
            commander_id (str): The ID of the commander processing communications.

        Side effects:
            Updates the communication_data["targets"] with communicated enemy positions.
        """
        # Clear previous communicated targets
        self.communication_data["targets"].clear()

        # Process communications from commanders to artillery
        # Get all scouts that communicated with this commander
        for scout_id in self.communication_data["network"].get(commander_id, []):
            # Add all enemies detected by each scout to the targets
            for enemy in self.communication_data["detected_enemies"].get(scout_id, []):
                self.communication_data["targets"][enemy["id"]] = enemy["position"]

        # Communicate targets to artillery units
        living_artillery = [artillery for artillery in self.friendly_agents
                            if artillery["type"] == "artillery" and artillery["health"] > 0]

        for artillery in living_artillery:
            self.communication_data["network"][artillery["id"]].append(commander_id)
            self.episode_comm_stats['commander_to_artillery'][commander_id] += 1
            logging.info(f"Commander {commander_id} communicated with Artillery {artillery['id']}")

    def _move_agent(self, agent, action):
        """
        Move an agent based on the given action.

        Args:
            agent (dict): The agent to move.
            action (int): The action to perform (0-7 for movement directions).

        Side effects:
            Updates the agent's position.
        """
        new_position = agent["position"].copy()
        if action == 0:  # Move up
            new_position[0] -= 1
        elif action == 1:  # Move down
            new_position[0] += 1
        elif action == 2:  # Move left
            new_position[1] -= 1
        elif action == 3:  # Move right
            new_position[1] += 1
        elif action == 4:  # Move up-left
            new_position[0] -= 1
            new_position[1] -= 1
        elif action == 5:  # Move up-right
            new_position[0] -= 1
            new_position[1] += 1
        elif action == 6:  # Move down-left
            new_position[0] += 1
            new_position[1] -= 1
        elif action == 7:  # Move down-right
            new_position[0] += 1
            new_position[1] += 1

        agent["position"] = self.ensure_position_in_bounds(new_position)

    def _fire_weapon(self, agent, target=None):
        """
        Handle weapon firing for a given agent.

        Args:
            agent (dict): The agent firing the weapon.
            target (tuple, optional): Specific target position. If None, chooses based on detected or communicated targets.

        Returns:
            bool: True if the agent hit a target, False otherwise.

        Side effects:
            Updates agent and target health, hit status, and firing status.
        """
        agent["hit"] = False
        agent["fired"] = True

        if agent["type"] == "artillery":
            potential_targets = [target] if target else list(self.communication_data["targets"].items())
        else:
            potential_targets = [(enemy['id'], enemy['position']) for enemy in agent.get("detected_enemies", []) if
                                 enemy['team'] != agent['team']]

        for enemy_id, enemy_pos in potential_targets:
            enemy = next((e for e in self.enemy_agents + self.friendly_agents if e['id'] == enemy_id), None)

            if enemy is None or enemy['team'] == agent['team'] or enemy['health'] <= 0:
                continue

            distance = np.linalg.norm(np.array(agent['position']) - np.array(enemy_pos))
            if distance <= agent['engagement_range']:
                damage = self._calculate_damage(agent, distance)
                previous_health = enemy['health']
                enemy['health'] = max(0, enemy['health'] - damage)
                agent["hit"] = True
                agent["target_pos"] = enemy_pos
                self.fired_agents.append(agent.copy())

                logging.info(f"{agent['type']} {agent['id']} fired at target {enemy_id}. "
                             f"Damage dealt: {previous_health - enemy['health']}, "
                             f"Enemy health: {previous_health} -> {enemy['health']}")

                if enemy['health'] == 0:
                    logging.info(f"Enemy {enemy_id} destroyed by {agent['type']} {agent['id']}")
                    enemy['just_died'] = True
                    enemy['destroyed_by'] = agent['id']
                    if agent['type'] == 'artillery' and agent['team'] == 'friendly':
                        self.episode_comm_stats['artillery_destroyed_enemies'][agent['id']] += 1
                        if enemy['id'] in self.communication_data["targets"]:
                            self.communication_data["artillery_destructions"][agent['id']] += 1
                            logging.info(f"Artillery {agent['id']} destroyed communicated target {enemy['id']}")
                elif enemy['health'] < previous_health:
                    enemy['just_damaged'] = True

                return True

        return False

    def _calculate_damage(self, agent, distance):
        """
        Calculate the damage dealt by an agent at a given distance.

        Args:
            agent (dict): The agent dealing damage.
            distance (float): The distance to the target.

        Returns:
            int: The amount of damage dealt.
        """
        if agent['type'] == 'artillery':
            far_damage, near_damage = agent['damage']
            return far_damage if distance > 2 else near_damage
        else:
            return agent['damage']

    def _get_observation(self):
        """
        Get the current observation of the environment.

        Returns:
            dict: Observations for all agents, including a global observation for each team's commander.
        """
        observations = {}

        def get_agent_observation(agent):
            obs = np.zeros((self.grid_height, self.grid_width, 3), dtype=np.uint8)

            # Mark the agent's position
            obs[agent["position"][0], agent["position"][1]] = [0, 255, 0] if agent["team"] == "friendly" else [255, 0,
                                                                                                               0]

            # Mark observed area
            for row in range(max(0, agent["position"][0] - agent["observation_range"]),
                             min(self.grid_height, agent["position"][0] + agent["observation_range"] + 1)):
                for col in range(max(0, agent["position"][1] - agent["observation_range"]),
                                 min(self.grid_width, agent["position"][1] + agent["observation_range"] + 1)):
                    distance = abs(row - agent["position"][0]) + abs(col - agent["position"][1])
                    if distance <= agent["observation_range"]:
                        # Check if there's an agent at this position
                        observed_agent = next((a for a in self.friendly_agents + self.enemy_agents
                                               if a["position"] == [row, col] and a["health"] > 0), None)
                        if observed_agent:
                            obs[row, col] = [0, 255, 0] if observed_agent["team"] == "friendly" else [255, 0, 0]
                        else:
                            obs[row, col] = [100, 100, 100]  # Observed empty space

            return obs.flatten()  # Flatten the observation

        def get_commander_observation(team):
            obs = np.zeros((self.grid_height, self.grid_width, 3), dtype=np.uint8)
            team_agents = self.friendly_agents if team == "friendly" else self.enemy_agents

            for agent in team_agents:
                if agent["health"] > 0:
                    agent_obs = get_agent_observation(agent)
                    obs |= agent_obs.reshape(self.grid_height, self.grid_width, 3)

            return obs.flatten()

        for agent in self.friendly_agents + self.enemy_agents:
            if agent["health"] > 0:
                observations[agent["id"]] = get_agent_observation(agent)

        # Add commander observations
        observations["friendly_commander"] = get_commander_observation("friendly")
        observations["enemy_commander"] = get_commander_observation("enemy")

        return observations

    def _calculate_reward(self):
        """
        Calculate rewards for all agents and teams in a unified manner.

        Returns:
            dict: A dictionary containing all reward-related information, including individual agent rewards,
                  team rewards, and performance metrics (strength and destruction) for both teams.
        """
        reward_info = {
            'friendly': {
                'individual': [0] * len(self.friendly_agents),
                'team': 0,
                'strength': 0,
                'destruction': 0
            },
            'enemy': {
                'individual': [0] * len(self.enemy_agents),
                'team': 0,
                'strength': 0,
                'destruction': 0
            }
        }

        def calculate_agent_reward(agent):
            reward = 0
            # Base step penalty
            reward -= 0.0005  # Changed from 0.001 to potentially encourage more exploration.

            # Detection reward
            if agent["detected"]:
                reward += 0.1  # Changed from 0.05 to encourage scouts to actively search - first step in kill chain

            # Firing reward/penalty
            if agent["fired"]:
                reward += 1.0 if agent["hit"] else -0.1  # Changed from 0.5 and -0.05 to emphasize offensive actions

            # Communication reward
            if agent["type"] == "scout":
                # Reward for successful communication (i.e., if the scout's ID is in any commander's network)
                if any(agent["id"] in commander_network
                       for commander_network in self.communication_data["network"].values()):
                    reward += 0.2  # Changed from 0.1
            elif agent["type"] == "commander":
                # Reward for successful communication (i.e., if the commander's ID is in any artillery's network)
                if any(agent["id"] in artillery_network
                       for artillery_network in self.communication_data["network"].values()):
                    reward += 0.3  # Changed from 0.2
            elif agent["type"] == "artillery" and agent["fired"] and agent["hit"]:
                # Check if the hit target was in the communicated targets
                if agent["target_pos"] in self.communication_data["targets"].values():
                    reward += 0.5  # Changed from 0.3 - this is the culminating step in the kill chain.

            # Destruction reward
            if agent["health"] == 0 and agent.get("just_died", True):
                reward -= 2.0  # Changed from -1.0 to emphasize destroy the enemy before being destroyed.

            return reward

        def calculate_team_strength(team):
            current_health = sum(agent['health'] for agent in team)
            initial_health = sum(AGENT_ATTRIBUTES[agent['type']]['health'] for agent in team)
            return current_health / initial_health

        def calculate_team_destruction(opponent_team):
            initial_health = sum(AGENT_ATTRIBUTES[agent['type']]['health'] for agent in opponent_team)
            destroyed_health = sum(
                AGENT_ATTRIBUTES[agent['type']]['health'] - agent['health'] for agent in opponent_team)
            return destroyed_health / initial_health

        def calculate_team_reward(team_strength, opponent_destruction):
            strength_weight = 0.3  # Changed from 0.4
            destruction_weight = 0.7  # Changed from 0.6 to emphasize more destruction of enemy
            return (strength_weight * team_strength + destruction_weight * opponent_destruction) * 1.5  # Changed from 2

        # Calculate individual rewards for this step
        for i, agent in enumerate(self.friendly_agents):
            reward_info['friendly']['individual'][i] = calculate_agent_reward(agent)

        for i, agent in enumerate(self.enemy_agents):
            reward_info['enemy']['individual'][i] = calculate_agent_reward(agent)

        # Calculate team metrics
        reward_info['friendly']['strength'] = calculate_team_strength(self.friendly_agents)
        reward_info['enemy']['strength'] = calculate_team_strength(self.enemy_agents)
        reward_info['friendly']['destruction'] = calculate_team_destruction(self.enemy_agents)
        reward_info['enemy']['destruction'] = calculate_team_destruction(self.friendly_agents)

        # Calculate team rewards for this step
        reward_info['friendly']['team'] = calculate_team_reward(
            reward_info['friendly']['strength'],
            reward_info['friendly']['destruction']
        )
        reward_info['enemy']['team'] = calculate_team_reward(
            reward_info['enemy']['strength'],
            reward_info['enemy']['destruction']
        )

        # Update episode rewards
        for team in ['friendly', 'enemy']:
            for i, reward in enumerate(reward_info[team]['individual']):
                self.episode_rewards[team]['individual'][i] += reward
            self.episode_rewards[team]['team'] += reward_info[team]['team']
            self.episode_rewards[team]['total'] = sum(self.episode_rewards[team]['individual']) + self.episode_rewards[
                team]['team']

        return reward_info

    def _check_done(self):
        """
        Check if the episode is finished.

        Returns:
            bool: True if the episode is done, False otherwise.
        """
        friendly_alive = sum(1 for agent in self.friendly_agents if agent["health"] > 0)
        enemy_alive = sum(1 for agent in self.enemy_agents if agent["health"] > 0)
        return self.current_step >= self.max_steps or friendly_alive == 0 or enemy_alive == 0

    def reset(self):
        """
        Reset the environment to its initial state.

        Returns:
            dict: Initial observation of the environment.
        """
        self.friendly_agents = self._create_agents(self.friendly_config, team="friendly")
        self.enemy_agents = self._create_agents(self.enemy_config, team="enemy")

        self.friendly_agent_ids = [agent["id"] for agent in self.friendly_agents]
        self.enemy_agent_ids = [agent["id"] for agent in self.enemy_agents]

        logging.info(f"Reset friendly agent IDs: {self.friendly_agent_ids}")
        logging.info(f"Reset enemy agent IDs: {self.enemy_agent_ids}")

        self.current_step = 0
        self.total_steps = 0

        # Reset communication-related attributes
        self.communication_data = {
            "network": defaultdict(list),
            "detected_enemies": defaultdict(list),
            "last_communicated": {},
            "targets": {},
            "artillery_destructions": {agent['id']: 0 for agent in self.friendly_agents if
                                       agent['type'] == 'artillery'},
            "episode_destructions": {agent['id']: 0 for agent in self.friendly_agents if agent['type'] == 'artillery'}
        }

        # Reset episode_comm_stats
        self.episode_comm_stats = {
            'scout_to_commander': {agent['id']: 0 for agent in self.friendly_agents if agent['type'] == 'scout'},
            'commander_to_artillery': {agent['id']: 0 for agent in self.friendly_agents if
                                       agent['type'] == 'commander'},
            'artillery_destroyed_enemies': {agent['id']: 0 for agent in self.friendly_agents if
                                            agent['type'] == 'artillery'}
        }

        # Reset reward-related attributes
        self.episode_rewards = {
            'friendly': {
                'individual': [0] * len(self.friendly_agents),
                'team': 0,
                'total': 0
            },
            'enemy': {
                'individual': [0] * len(self.enemy_agents),
                'team': 0,
                'total': 0
            }
        }

        # Reset agent-specific attributes
        for agent in self.friendly_agents + self.enemy_agents:
            agent["detected"] = False
            agent["detected_enemies"] = []
            agent["fired"] = False
            agent["hit"] = False
            agent["just_died"] = False
            agent["just_damaged"] = False
            if agent.get("can_communicate", False):
                agent["last_communicated"] = -1
            if 'destroyed_by' in agent:
                del agent['destroyed_by']

        # Reset the fired_agents list
        self.fired_agents.clear()

        logging.info("Environment reset. Starting new episode.")
        return self._get_observation()

    def render(self):
        """
        Render the current state of the environment.

        This function visualizes the current state of the battlefield, including:
        - The positions of all agents (friendly and enemy)
        - The observation and engagement ranges of each agent
        - Any ongoing actions (movement, firing, communication)
        - Statistical information about the current state of the battle

        The rendering is done using Pygame and includes:
        - A grid representing the battlefield
        - Different colors and shapes for different types of units
        - Visual indicators for agent actions and states
        - Text displays for relevant statistics and information

        The render function can be called after each step to create an animation of the battle progression.
        It's particularly useful for debugging and for gaining intuition about the agents' behavior.
        """

        if self.render_mode != 'human':
            return

        def draw_observation_range(agent):
            observation_cells = set()
            for row in range(max(0, agent["position"][0] - agent["observation_range"]),
                             min(self.grid_height, agent["position"][0] + agent["observation_range"] + 1)):
                for col in range(max(0, agent["position"][1] - agent["observation_range"]),
                                 min(self.grid_width, agent["position"][1] + agent["observation_range"] + 1)):
                    distance = abs(row - agent["position"][0]) + abs(col - agent["position"][1])
                    if distance <= agent["observation_range"]:
                        observation_cells.add((row, col))

            surface = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            for row, col in observation_cells:
                rect_x = self.padding + col * self.cell_size
                rect_y = self.padding + row * self.cell_size
                rect = pygame.Rect(rect_x, rect_y, self.cell_size, self.cell_size)
                cell_count = sum(1 for other_agent in self.friendly_agents + self.enemy_agents
                                 if other_agent["health"] > 0
                                 and abs(row - other_agent["position"][0]) + abs(col - other_agent["position"][1]) <=
                                 other_agent["observation_range"])
                if cell_count == 1:
                    pygame.draw.rect(surface, (255, 255, 150, 64), rect)  # Light yellow shade with alpha 64
                else:
                    saturation = max(64, 255 - (cell_count - 1) * 32)
                    pygame.draw.rect(surface, (255, saturation, 0, 64), rect)  # Darker yellow shade with alpha 64

            self.screen.blit(surface, (0, 0))

        def draw_engagement_range(agent):
            center_x = self.padding + agent["position"][1] * self.cell_size + self.cell_size // 2
            center_y = self.padding + agent["position"][0] * self.cell_size + self.cell_size // 2
            radius = agent["engagement_range"] * self.cell_size

            surface = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            pygame.draw.circle(surface, (139, 0, 0, 128), (center_x, center_y), radius,
                               2)  # Dark red ring with alpha 128

            self.screen.blit(surface, (0, 0))

        def draw_agent(agent):
            agent_rect = pygame.Rect(
                self.padding + agent["position"][1] * self.cell_size + self.cell_size // 4,
                self.padding + agent["position"][0] * self.cell_size + self.cell_size // 4,
                self.cell_size // 2,
                self.cell_size // 2,
            )
            if agent["health"] > 0:
                if agent["team"] == "friendly":
                    pygame.draw.rect(self.screen, (0, 0, 255), agent_rect)  # Blue for friendly agents
                else:
                    pygame.draw.polygon(self.screen, (255, 0, 0), [
                        (agent_rect.topleft[0], agent_rect.topleft[1]),
                        (agent_rect.topright[0], agent_rect.topright[1]),
                        (agent_rect.bottomright[0], agent_rect.bottomright[1]),
                        (agent_rect.bottomleft[0], agent_rect.bottomleft[1]),
                    ])  # Red diamond for enemy agents

                # Draw the agent type designation
                font = pygame.font.Font(None, 16)
                text = font.render(agent["type"][0].upper(), True, (255, 255, 255))
                text_rect = text.get_rect(center=agent_rect.center)
                self.screen.blit(text, text_rect)
            else:
                pygame.draw.rect(self.screen, (0, 0, 0), agent_rect)  # Black for destroyed agents

        def draw_firing_effects():
            for fired_agent in self.fired_agents:
                target_agent = next(
                    (a for a in self.friendly_agents + self.enemy_agents if a["position"] == fired_agent["target_pos"]),
                    None)
                if target_agent:
                    start_pos = (self.padding + fired_agent["position"][1] * self.cell_size + self.cell_size // 2,
                                 self.padding + fired_agent["position"][0] * self.cell_size + self.cell_size // 2)
                    end_pos = (self.padding + target_agent["position"][1] * self.cell_size + self.cell_size // 2,
                               self.padding + target_agent["position"][0] * self.cell_size + self.cell_size // 2)

                    # Green for friendly fire, Red for enemy fire
                    fire_color = (0, 255, 0) if fired_agent["team"] == "friendly" else (255, 0, 0)
                    pygame.draw.line(self.screen, fire_color, start_pos, end_pos, 2)

                    # Draw hit effect if target is destroyed
                    if fired_agent["hit"]:
                        hit_pos = end_pos
                        for _ in range(8):
                            angle = random.uniform(0, 2 * math.pi)
                            radius = random.uniform(5, 10)
                            x = hit_pos[0] + math.cos(angle) * radius
                            y = hit_pos[1] + math.sin(angle) * radius
                            pygame.draw.line(self.screen, (255, 165, 0), hit_pos, (x, y), 2)  # Orange starburst effect

        def draw_communication_links():
            for commander in self.friendly_agents:
                if commander["type"] == "commander":
                    commander_pos = (self.padding + commander["position"][1] * self.cell_size + self.cell_size // 2,
                                     self.padding + commander["position"][0] * self.cell_size + self.cell_size // 2)

                    # Draw scout to commander links
                    for scout_id in self.communication_data["network"].get(commander["id"], []):
                        scout = next((s for s in self.friendly_agents if s["id"] == scout_id), None)
                        if scout and scout["type"] == "scout":
                            scout_pos = (self.padding + scout["position"][1] * self.cell_size + self.cell_size // 2,
                                         self.padding + scout["position"][0] * self.cell_size + self.cell_size // 2)
                            pygame.draw.line(self.screen, (0, 255, 255), scout_pos, commander_pos, 2)

                    # Draw commander to artillery links
                    for artillery in self.friendly_agents:
                        if (artillery["type"] == "artillery" and
                                commander["id"] in self.communication_data["network"].get(artillery["id"], [])):
                            artillery_pos = (
                                self.padding + artillery["position"][1] * self.cell_size + self.cell_size // 2,
                                self.padding + artillery["position"][0] * self.cell_size + self.cell_size // 2)
                            pygame.draw.line(self.screen, (255, 255, 0), commander_pos, artillery_pos, 2)

        def draw_detected_enemies():
            for agent_id, enemies in self.communication_data["detected_enemies"].items():
                agent = next((a for a in self.friendly_agents if a["id"] == agent_id), None)
                if agent:
                    for enemy in enemies:
                        pygame.draw.circle(self.screen, (255, 0, 0),
                                           (self.padding + enemy["position"][1] * self.cell_size + self.cell_size // 2,
                                            self.padding + enemy["position"][0] * self.cell_size + self.cell_size // 2),
                                           self.cell_size // 4, 1)

        def draw_communicated_targets():
            for enemy_id, position in self.communication_data["targets"].items():
                pygame.draw.circle(self.screen, (255, 165, 0),
                                   (self.padding + position[1] * self.cell_size + self.cell_size // 2,
                                    self.padding + position[0] * self.cell_size + self.cell_size // 2),
                                   self.cell_size // 3, 2)

        def draw_stats():
            font = pygame.font.Font(None, 24)
            friendly_units = sum(1 for agent in self.friendly_agents if agent["health"] > 0)
            enemy_units = sum(1 for agent in self.enemy_agents if agent["health"] > 0)

            stats_text = [
                f"Friendly Units: {friendly_units}, Enemy Units: {enemy_units}",
                f"Total Steps: {self.total_steps}",
                f"Current Total Team Reward - Friendly: {self.episode_rewards['friendly']['total']:.2f}, Enemy: {self.episode_rewards['enemy']['total']:.2f}",
            ]

            if self.reward_info:
                stats_text.extend([
                    f"Team Reward - Friendly: {self.episode_rewards['friendly']['team']:.2f}, Enemy: {self.episode_rewards['enemy']['team']:.2f}",
                    f"Team Strength - Friendly: {self.reward_info['friendly']['strength']:.2f}, Enemy: {self.reward_info['enemy']['strength']:.2f}",
                    f"Team Destruction - Friendly: {self.reward_info['friendly']['destruction']:.2f}, Enemy: {self.reward_info['enemy']['destruction']:.2f}"
                ])

            for i, line in enumerate(stats_text):
                stats_surface = font.render(line, True, (0, 0, 0))
                self.screen.blit(stats_surface,
                                 (self.screen_width - 1600, self.screen_height - 24 * (len(stats_text) - i)))

            # Display individual agent rewards
            friendly_rewards = self.episode_rewards['friendly']['individual']
            enemy_rewards = self.episode_rewards['enemy']['individual']

            # Title for individual agent rewards
            title_surface = font.render("Individual Agent Rewards", True, (0, 0, 0))
            self.screen.blit(title_surface, (
                self.screen_width - 1000, self.screen_height - 24 * (len(friendly_rewards) + len(enemy_rewards) + 2)))

            # Friendly agent rewards
            for i, reward in enumerate(friendly_rewards):
                text = f"F{i}: {reward:.2f}"
                surface = font.render(text, True, (0, 0, 255))
                self.screen.blit(surface, (
                    self.screen_width - 1000,
                    self.screen_height - 24 * (len(friendly_rewards) + len(enemy_rewards) - i)))

            # Enemy agent rewards
            for i, reward in enumerate(enemy_rewards):
                text = f"E{i}: {reward:.2f}"
                surface = font.render(text, True, (255, 0, 0))
                self.screen.blit(surface, (
                    self.screen_width - 1000, self.screen_height - 24 * (len(enemy_rewards) - i)))

        # Main rendering logic
        self.screen.fill((255, 255, 255))

        # Draw observation and engagement ranges
        for agent in self.friendly_agents + self.enemy_agents:
            if agent["health"] > 0:
                draw_observation_range(agent)
                draw_engagement_range(agent)

        # Draw grid
        for row in range(self.grid_height):
            for col in range(self.grid_width):
                rect = pygame.Rect(
                    self.padding + col * self.cell_size,
                    self.padding + row * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

        # Draw agents
        for agent in self.friendly_agents + self.enemy_agents:
            draw_agent(agent)

        # Draw fire effects
        draw_firing_effects()

        # Draw communication-related visualizations
        draw_communication_links()
        draw_detected_enemies()
        draw_communicated_targets()

        # Draw stats
        draw_stats()

        pygame.display.flip()
        time.sleep(self.render_speed)

    def close(self):
        """
        Close the environment and release resources.
        """
        if self.render_mode == 'human':
            pygame.quit()
        print("Environment closed")


# Example usage
if __name__ == "__main__":
    friendly_config = {
        "artillery": {"count": 2, "positions": [(3, 0), (6, 0)]},
        "scout": {"count": 3, "positions": [(2, 2), (4, 4), (6, 6)]},
        "commander": {"count": 1, "positions": [(4, 1)]}
    }

    enemy_config = {
        "artillery": {"count": 2, "positions": [(3, 63), (6, 63)]},
        "scout": {"count": 2, "positions": [(3, 60), (6, 60)]},
        "infantry": {"count": 1, "positions": [(5, 62)]},
        "light_tank": {"count": 3, "positions": [(1, 61), (4, 61), (8, 61)]},
        "commander": {"count": 1, "positions": [(4, 62)]}
    }

    env = CombinedArmsRLEnvMAComms(friendly_config, enemy_config, render_speed=0.05, max_steps=2000,
                                   render_mode='human')

    observation = env.reset()
    done = False

    while not done:
        actions = [env.action_space.sample() for _ in range(len(env.friendly_agents) + len(env.enemy_agents))]
        observation, reward, done, _ = env.step(actions)
        env.render()

    env.close()
