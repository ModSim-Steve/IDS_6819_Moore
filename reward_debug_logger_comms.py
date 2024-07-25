import numpy as np
from combined_arms_rl_env_MA_comms import CombinedArmsRLEnvMAComms


class RewardDebugEnvComms(CombinedArmsRLEnvMAComms):
    def __init__(self, friendly_config, enemy_config, render_speed=0.05, max_steps=20, render_mode='human'):
        super().__init__(friendly_config, enemy_config, render_speed, max_steps, render_mode)

    def step(self, actions):
        self.current_step += 1
        self.total_steps += 1

        print(f"\nStep {self.current_step} - Agent Positions:")
        for agent in self.friendly_agents + self.enemy_agents:
            print(f"  {agent['id']}: {agent['position']}")

        print(f"\nStep {self.current_step} actions:")
        for agent, action in zip(self.friendly_agents + self.enemy_agents, actions):
            print(f"  Agent {agent['id']}: action {action}")
            if agent["health"] > 0:
                self._detect_enemies(agent)
                if action < 8:  # Move actions
                    self._move_agent(agent, action)
                elif action == 8:  # Fire action
                    self._fire_weapon(agent)
                elif action == 9:  # Communicate action
                    if agent["can_communicate"] and (self.current_step - agent["last_communicated"]) > agent[
                        "communication_cooldown"]:
                        self._communicate(agent)
                        agent["last_communicated"] = self.current_step
                elif action == 10:  # Hold position
                    pass

        self._process_communication()

        # Calculate rewards using our debug version
        reward_info = self._calculate_reward()

        # Update agent states
        for agent in self.friendly_agents + self.enemy_agents:
            if agent['health'] <= 0:
                agent['health'] = 0
                if not agent.get('destroyed', False):
                    agent['destroyed'] = True
                    print(f"Agent {agent['id']} has been destroyed")

        # Check if the episode is done
        done = self._check_done()

        # Get the new observation
        observation = self._get_observation()

        print(f"\nStep {self.current_step} Reward Breakdown:")
        print("Individual Rewards:")
        for team in ['friendly', 'enemy']:
            for i, reward in enumerate(reward_info['individual_rewards'][team]):
                agent_id = f"{team}_{i}"
                print(f"  {agent_id}: {reward}")

        print("\nCumulative Rewards:")
        for team in ['friendly', 'enemy']:
            for i, reward in enumerate(reward_info['cumulative_rewards'][team]):
                agent_id = f"{team}_{i}"
                print(f"  {agent_id}: {reward}")

        print(f"\nTeam Rewards:")
        print(f"  Friendly: {reward_info['team_rewards']['friendly']}")
        print(f"  Enemy: {reward_info['team_rewards']['enemy']}")

        print(f"\nTeam Metrics:")
        print(f"  Friendly Strength: {reward_info['strength']['friendly']}")
        print(f"  Friendly Destruction: {reward_info['destruction']['friendly']}")
        print(f"  Enemy Strength: {reward_info['strength']['enemy']}")
        print(f"  Enemy Destruction: {reward_info['destruction']['enemy']}")

        total_friendly_reward = sum(reward_info['individual_rewards']['friendly']) + reward_info['team_rewards'][
            'friendly']
        total_enemy_reward = sum(reward_info['individual_rewards']['enemy']) + reward_info['team_rewards']['enemy']

        print(f"\nTotal Team Rewards:")
        print(f"  Friendly: {total_friendly_reward}")
        print(f"  Enemy: {total_enemy_reward}")

        print("\nAgent health status after step:")
        for agent in self.friendly_agents + self.enemy_agents:
            print(f"  {agent['id']} health: {agent['health']}")

        return observation, reward_info['individual_rewards'], done, {
            "friendly_reward": total_friendly_reward,
            "enemy_reward": total_enemy_reward,
            "friendly_strength": reward_info['strength']['friendly'],
            "friendly_destruction": reward_info['destruction']['friendly'],
            "enemy_strength": reward_info['strength']['enemy'],
            "enemy_destruction": reward_info['destruction']['enemy'],
            "cumulative_rewards": reward_info['cumulative_rewards']
        }


# Example usage
if __name__ == "__main__":
    from combined_arms_dqn_agent_comms import DQNAgentComms
    from combined_arms_TRN_configs_comms import OpposingConfigComms

    friendly_config = {
        "artillery": {"count": 1, "positions": [(1, 1)]},
        "scout": {"count": 1, "positions": [(0, 0)]},
        "commander": {"count": 1, "positions": [(1, 0)]}
    }
    enemy_config = {
        "artillery": {"count": 1, "positions": [(8, 8)]},
        "scout": {"count": 1, "positions": [(9, 9)]},
        "commander": {"count": 1, "positions": [(8, 9)]}
    }

    env = RewardDebugEnvComms(friendly_config, enemy_config, max_steps=20, render_mode='human')
    opposing_config = OpposingConfigComms()

    # Initialize agents
    state = env.reset()
    friendly_agents = [DQNAgentComms(np.prod(state[agent_id].shape), env.action_space.n, agent_id)
                       for agent_id in env.friendly_agent_ids]

    episode_rewards = {
        'friendly': {
            'individual': [0] * len(env.friendly_agents),
            'team': 0,
            'total': 0
        },
        'enemy': {
            'individual': [0] * len(env.enemy_agents),
            'team': 0,
            'total': 0
        }
    }

    for step in range(20):  # Run for 20 steps
        print(f"\n--- Step {step + 1} ---")

        # Get actions
        actions_friendly = [agent.act(state[agent.agent_id].flatten()) for agent in friendly_agents]
        actions_opposing = opposing_config.get_actions(len(env.enemy_agents))
        actions = actions_friendly + actions_opposing

        # Take a step in the environment
        next_state, rewards, done, info = env.step(actions)

        reward_info = info['step_rewards']

        # Update episode rewards
        for team in ['friendly', 'enemy']:
            for i, reward in enumerate(reward_info[team]['individual']):
                episode_rewards[team]['individual'][i] += reward
            episode_rewards[team]['team'] += reward_info[team]['team']
            episode_rewards[team]['total'] = sum(episode_rewards[team]['individual']) + episode_rewards[team]['team']

        # Store experience for friendly agents
        for i, agent in enumerate(friendly_agents):
            if agent.agent_id in state and agent.agent_id in next_state:
                agent.store_experience(
                    state[agent.agent_id].flatten(),
                    actions_friendly[i],
                    rewards['friendly'][i],
                    next_state[agent.agent_id].flatten(),
                    done
                )

        state = next_state

        print(f"Step Rewards:")
        print(f"  Friendly: {reward_info['friendly']}")
        print(f"  Enemy: {reward_info['enemy']}")
        print(f"Cumulative Episode Rewards:")
        print(f"  Friendly: {episode_rewards['friendly']}")
        print(f"  Enemy: {episode_rewards['enemy']}")

        if done:
            break

    print(f"\nSimulation finished.")
    print(f"Final Episode Rewards:")
    print(f"  Friendly: {episode_rewards['friendly']}")
    print(f"  Enemy: {episode_rewards['enemy']}")

    env.close()
