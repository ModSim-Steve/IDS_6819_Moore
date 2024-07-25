import os
import numpy as np
import logging
import time
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from combined_arms_TRN_configs_comms import adjust_difficulty_comms

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def train_model(env, episodes, opposing_config, base_difficulty=0, train_friendly=True, agent_class=None, patience=20,
                min_episodes=400, debug=False):
    """
    Train a multi-agent reinforcement learning model in the Combined Arms environment with communication.

    This function manages the training process for multiple agents in a simulated combat environment.
    It handles episode iterations, reward calculations, model updates, and logging of various statistics.

    Args:
        env (CombinedArmsRLEnvMAComms): The environment instance for training.
        episodes (int): The maximum number of episodes to train for.
        opposing_config (OpposingConfigComms): Configuration for the opposing team's behavior.
        base_difficulty (int, optional): Starting difficulty level. Defaults to 0.
        train_friendly (bool, optional): If True, train friendly agents; otherwise, train enemy agents. Defaults to True.
        agent_class (class, optional): The class to use for creating agent instances. Defaults to None.
        patience (int, optional): Number of episodes without improvement before early stopping. Defaults to 20.
        min_episodes (int, optional): Minimum number of episodes to train before allowing early stopping. Defaults to 400.
        debug (bool, optional): If True, enable debug logging. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - friendly_rewards (list): List of dictionaries containing reward information for friendly agents per episode.
              Each dictionary includes 'individual' (list of individual agent rewards), 'team' (team-level reward),
              'total' (total team reward), 'strength' (team strength metric), and 'destruction' (team destruction metric).
              Format: [{'total': float, 'strength': float, 'destruction': float, 'individual': [float, ...]}, ...]
            - enemy_rewards (list): Similar to friendly_rewards, but for enemy agents.
            - communication_stats (dict): Dictionary tracking communication-related statistics throughout training.
              Format: {
              'scout_to_commander': {scout_id: [int, ...], ...},
              'commander_to_artillery': {commander_id: [int, ...], ...},
              'artillery_destroyed_enemies': {artillery_id: [int, ...], ...},
              'artillery_comm_destroyed_enemies': {artillery_id: [int, ...], ...}}
            - episode_lengths (list): List of integers representing the length of each episode.
            - best_models (list): List of agent instances with the best performance.
            - last_models (list): List of agent instances from the last episode.
            - save_folder (str): Path to the folder where models are saved.
            - plot_dir (str): Path to the folder where plots are saved.

        The return value would look like this:
        (
            [{'total': float, 'strength': float, 'destruction': float, 'individual': [float, ...]}, ...],  # friendly_rewards
            [{'total': float, 'strength': float, 'destruction': float, 'individual': [float, ...]}, ...],  # enemy_rewards
            [int, int, ...],  # episode_lengths
            [DQNAgentComms, DQNAgentComms, ...],  # best_models
            [DQNAgentComms, DQNAgentComms, ...],  # last_models
            'path/to/saved/models',  # save_folder
            'path/to/saved/plots',  # plot_dir
            {  # communication_stats
                'scout_to_commander': {scout_id: [int, ...], ...},
                'commander_to_artillery': {commander_id: [int, ...], ...},
                'artillery_destroyed_enemies': {artillery_id: [int, ...], ...},
                'artillery_comm_destroyed_enemies': {artillery_id: [int, ...], ...}
            }
        )

    The function uses the unified reward structure from the environment, which includes individual agent rewards,
    team rewards, and performance metrics (strength and destruction). It accumulates these rewards over each episode
    and uses them for agent learning and performance evaluation.

    Side effects:
        - Saves model checkpoints to disk.
        - Generates and saves performance plots.
        - Logs training progress and statistics.
    """

    logging.info(f"Starting training process with {episodes} episodes")
    logging.info(f"Base difficulty: {base_difficulty}")
    logging.info(f"Training {'friendly' if train_friendly else 'enemy'} agents")
    logging.info(f"Patience: {patience}")
    logging.info(f"Minimum episodes: {min_episodes}")
    logging.info(f"Debug mode: {'ON' if debug else 'OFF'}")

    batch_size = 32
    friendly_rewards = []
    enemy_rewards = []
    episode_lengths = []

    best_reward = float('-inf')
    best_models = None
    last_models = None
    episodes_without_improvement = 0

    # Create directories for saving models and plots
    save_folder, timestamp = create_timestamped_dir('saved_models')
    plot_dir, _ = create_timestamped_dir('saved_plots')

    # Initialize communication statistics tracking
    communication_stats = {
        'scout_to_commander': {agent['id']: [] for agent in env.friendly_agents if agent['type'] == 'scout'},
        'commander_to_artillery': {agent['id']: [] for agent in env.friendly_agents if agent['type'] == 'commander'},
        'artillery_destroyed_enemies': {agent['id']: [] for agent in env.friendly_agents if
                                        agent['type'] == 'artillery'},
        'artillery_comm_destroyed_enemies': {agent['id']: [] for agent in env.friendly_agents if
                                             agent['type'] == 'artillery'}
    }

    start_time = time.time()

    for e in tqdm(range(episodes), desc="Training Progress"):
        log_section(f"Starting episode {e + 1}/{episodes}")

        # Adjust difficulty based on current episode
        adjusted_opposing_config = adjust_difficulty_comms(
            opposing_config,
            base_difficulty=base_difficulty,
            episode=e,
            total_episodes=episodes,
            against_friendly=train_friendly
        )

        # Reset the environment and initialize agents
        state = env.reset()
        friendly_agents = [agent_class(np.prod(state[agent_id].shape), env.action_space.n, agent_id)
                           for agent_id in env.friendly_agent_ids]

        episode_start_time = time.time()
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

        for time_step in range(env.max_steps):
            # Render the first 100 steps of the first episode
            if e == 0 and time_step < 100:
                env.render()
            elif e == 0 and time_step == 100:
                env.close()

            # Get actions for friendly agents
            actions_friendly = []
            for agent in friendly_agents:
                if agent.agent_id in state:
                    agent_state = state[agent.agent_id].flatten()
                    action = agent.act(agent_state)
                    actions_friendly.append(action)
                else:
                    actions_friendly.append(None)

            # Get actions for opposing agents
            actions_opposing = adjusted_opposing_config.get_actions(len(env.enemy_agents))

            # Combine actions and take a step in the environment
            actions = actions_friendly + actions_opposing
            next_state, rewards, done, info = env.step(actions)

            reward_info = info['step_rewards']

            # Update episode rewards
            for team in ['friendly', 'enemy']:
                for i, reward in enumerate(reward_info[team]['individual']):
                    episode_rewards[team]['individual'][i] += reward
                episode_rewards[team]['team'] += reward_info[team]['team']
                episode_rewards[team]['total'] = sum(episode_rewards[team]['individual']) + episode_rewards[team][
                    'team']

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

            # Train friendly agents
            for agent in friendly_agents:
                agent.replay(batch_size)

            if done:
                break

        episode_duration = time.time() - episode_start_time

        # Record rewards for this episode
        friendly_rewards.append({
            'total': episode_rewards['friendly']['total'],
            'strength': info["friendly_strength"],
            'destruction': info["friendly_destruction"],
            'individual': episode_rewards['friendly']['individual']
        })
        enemy_rewards.append({
            'total': episode_rewards['enemy']['total'],
            'strength': info["enemy_strength"],
            'destruction': info["enemy_destruction"],
            'individual': episode_rewards['enemy']['individual']
        })
        episode_lengths.append(time_step + 1)

        # Log episode results
        logging.info(f"Episode {e + 1} completed in {time_step + 1} steps ({episode_duration:.2f} seconds)")
        logging.info(
            f"Friendly reward: {friendly_rewards[-1]['total']:.2f}, Enemy reward: {enemy_rewards[-1]['total']:.2f}")
        logging.info(
            f"Friendly strength: {friendly_rewards[-1]['strength']:.2f}, Friendly destruction: {friendly_rewards[-1]['destruction']:.2f}")

        logging.info("\nIndividual friendly rewards:")
        for i, (agent, reward) in enumerate(zip(env.friendly_agents, episode_rewards['friendly']['individual'])):
            agent_type = agent['type']
            agent_id = agent['id']
            logging.info(f"  {agent_type}_{i} ({agent_id}): {reward:.2f}")

        # Update communication statistics
        logging.info("\nCommunication Stats:")
        episode_comm_stats = env.episode_comm_stats.copy()

        # Scout communications
        logging.info("  Scout communications:")
        for scout_id, comm_count in episode_comm_stats['scout_to_commander'].items():
            logging.info(f"    {scout_id}: {comm_count}")
            communication_stats['scout_to_commander'][scout_id].append(comm_count)

        # Commander communications
        logging.info("  Commander communications:")
        for commander_id, comm_count in episode_comm_stats['commander_to_artillery'].items():
            logging.info(f"    {commander_id}: {comm_count}")
            communication_stats['commander_to_artillery'][commander_id].append(comm_count)

        # Artillery destroyed enemies
        logging.info("  Artillery destroyed enemies:")
        for artillery_id, destroyed_count in episode_comm_stats['artillery_destroyed_enemies'].items():
            logging.info(f"    {artillery_id}: {destroyed_count}")
            communication_stats['artillery_destroyed_enemies'][artillery_id].append(destroyed_count)

        # Artillery comm destroyed enemies
        logging.info("  Artillery comm destroyed enemies:")
        if "comm_destructions" in info:
            for artillery_id, count in info["comm_destructions"].items():
                logging.info(f"    {artillery_id}: {count}")
                communication_stats['artillery_comm_destroyed_enemies'][artillery_id].append(count)
        else:
            logging.info("  No communication-based destructions reported")
            for artillery_id in communication_stats['artillery_comm_destroyed_enemies']:
                communication_stats['artillery_comm_destroyed_enemies'][artillery_id].append(0)

        # Check if this is the best model so far
        if friendly_rewards[-1]['total'] > best_reward:
            best_reward = friendly_rewards[-1]['total']
            best_models = friendly_agents
            episodes_without_improvement = 0

            # Save the best models
            for agent in best_models:
                model_name = f'{agent.agent_id}_best.pth'
                model_path = os.path.join(save_folder, model_name)
                agent.save(model_path)
            logging.info(f"New best models saved with reward: {best_reward:.2f}")
        else:
            episodes_without_improvement += 1

        # Save last models
        last_models = friendly_agents
        for agent in last_models:
            model_name = f'{agent.agent_id}_last.pth'
            model_path = os.path.join(save_folder, model_name)
            agent.save(model_path)

        # Check for early stopping
        if e >= min_episodes and episodes_without_improvement >= patience:
            logging.info(f"Early stopping triggered. No improvement for {patience} episodes.")
            break

    total_duration = time.time() - start_time
    logging.info(f"\nTraining completed in {total_duration / 60:.2f} minutes")
    logging.info(f"Best models and last models saved in: {save_folder}")

    # Generate and save plots
    plot_results(friendly_rewards, enemy_rewards, episode_lengths, friendly_agents, plot_dir, patience, min_episodes)
    plot_communication_stats(communication_stats, episodes, plot_dir)
    plot_moving_average_rewards(episodes, friendly_rewards, enemy_rewards)
    plot_cumulative_rewards(episodes, friendly_rewards, enemy_rewards)
    plot_team_metrics(episodes, friendly_rewards, enemy_rewards)
    plot_episode_length(episodes, episode_lengths)

    logging.info(f"Plots saved in: {plot_dir}")

    return friendly_rewards, enemy_rewards, episode_lengths, best_models, last_models, save_folder, plot_dir


def log_section(message):
    logging.info(f"\n{'=' * 80}\n{message}\n{'=' * 80}")


def create_timestamped_dir(base_path):
    """
    Create a timestamped directory for saving models and plots.

    Args:
        base_path (str): The base path where the directory should be created.

    Returns:
        tuple: A tuple containing:
            - dir_path (str): Path to the created directory.
            - timestamp (str): The timestamp used for the directory name.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_path = os.path.join(base_path, f"{timestamp}")
    os.makedirs(dir_path, exist_ok=True)
    return dir_path, timestamp


def plot_results(friendly_rewards, enemy_rewards, episode_lengths, friendly_agents, plot_dir, patience, min_episodes):
    """
    Generate and save plots visualizing the training results.

    Args:
        friendly_rewards (list): List of reward dictionaries for friendly agents.
        enemy_rewards (list): List of reward dictionaries for enemy agents.
        episode_lengths (list): List of episode lengths.
        friendly_agents (list): List of friendly agent instances.
        plot_dir (str): Directory to save the plots.
        patience (int): Patience value used in training.
        min_episodes (int): Minimum episodes value used in training.

    This function creates several plots:
    - Average of friendly agents' rewards per episode
    - Individual friendly agent rewards per episode
    - Team total rewards per episode
    - Team strength and destruction scores per episode
    - Episode lengths over time

    The plots are saved in the specified plot_dir.
    """
    fig = plt.figure(figsize=(12, 28))
    gs = fig.add_gridspec(5, 1, height_ratios=[1, 1, 1, 1, 0.3])
    axes = [fig.add_subplot(gs[i]) for i in range(4)]

    episodes = range(1, len(friendly_rewards) + 1)

    # Plot average of friendly agents' rewards per episode
    avg_friendly_rewards = [np.mean(rewards['individual']) for rewards in friendly_rewards]
    axes[0].plot(episodes, avg_friendly_rewards, label='Average Friendly Reward', color='blue')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Average Reward')
    axes[0].set_title('Average Friendly Agent Rewards per Episode')
    axes[0].legend()

    # Plot individual friendly agent rewards
    num_agents = len(friendly_rewards[0]['individual'])
    colors = plt.cm.rainbow(np.linspace(0, 1, num_agents))
    for i in range(num_agents):
        agent_rewards = [rewards['individual'][i] for rewards in friendly_rewards]
        axes[1].plot(episodes, agent_rewards, label=f'Agent {i}', color=colors[i])
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Reward')
    axes[1].set_title('Individual Friendly Agent Rewards per Episode')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Plot team total rewards
    axes[2].plot(episodes, [r['total'] for r in friendly_rewards], label='Friendly', color='blue')
    axes[2].plot(episodes, [r['total'] for r in enemy_rewards], label='Enemy', color='red')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Team Reward')
    axes[2].set_title('Team Total Rewards per Episode')
    axes[2].legend()

    # Plot team strength and destruction scores
    axes[3].plot(episodes, [r['strength'] for r in friendly_rewards], label='Friendly Strength', color='green')
    axes[3].plot(episodes, [r['destruction'] for r in friendly_rewards], label='Friendly Destruction', color='orange')
    axes[3].plot(episodes, [r['strength'] for r in enemy_rewards], label='Enemy Strength', color='red')
    axes[3].plot(episodes, [r['destruction'] for r in enemy_rewards], label='Enemy Destruction', color='purple')
    axes[3].set_xlabel('Episode')
    axes[3].set_ylabel('Score')
    axes[3].set_title('Team Strength and Destruction Scores per Episode')
    axes[3].legend()

    # Add hyperparameters text
    hyperparams = f"Learning Rate: {friendly_agents[0].learning_rate}\n"
    hyperparams += f"Gamma: {friendly_agents[0].gamma}\n"
    hyperparams += f"Epsilon Decay: {friendly_agents[0].epsilon_decay}\n"
    hyperparams += f"Replay Buffer Size: {friendly_agents[0].memory.capacity}\n"
    hyperparams += f"Batch Size: 32\n"
    hyperparams += f"Target Network Update: Every {friendly_agents[0].target_update_frequency} steps\n"
    hyperparams += f"Patience: {patience}\n"
    hyperparams += f"Minimum Episodes: {min_episodes}"

    fig.text(0.1, 0.02, hyperparams, fontsize=9, verticalalignment='bottom')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(os.path.join(plot_dir, 'training_results.png'), bbox_inches='tight')
    plt.close()

    # Plot episode lengths
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, episode_lengths, label='Episode Length', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Episode Length over Time')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'episode_lengths.png'))
    plt.close()

    print(f"Plots saved in {os.path.abspath(plot_dir)}")


def plot_communication_stats(comm_stats, episodes, plot_dir):
    """
    Generate and save plots visualizing the communication statistics.

    Args:
        comm_stats (dict): Dictionary containing communication statistics.
        episodes (int): Total number of episodes.
        plot_dir (str): Directory to save the plots.

    This function creates several plots:
    - Scout to Commander Communication
    - Commander to Artillery Communication
    - Enemies Destroyed by Artillery (All)
    - Enemies Destroyed by Artillery (Communication-based)

    The plots are saved in the specified plot_dir.
    """
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 24))

    episodes_range = range(1, episodes + 1)

    def plot_with_offset(ax, data, label, offset=0):
        ax.plot(episodes_range[:len(data)], [x + offset for x in data], label=f'{label} (offset: {offset})')

    # Scout to Commander Communication
    for i, (scout_id, comms) in enumerate(comm_stats['scout_to_commander'].items()):
        plot_with_offset(ax1, comms, f'Scout {scout_id}', i * 0.1)
    ax1.set_title('Scout to Commander Communication')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Communication Count')
    ax1.legend()
    ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Commander to Artillery Communication
    for i, (commander_id, comms) in enumerate(comm_stats['commander_to_artillery'].items()):
        plot_with_offset(ax2, comms, f'Commander {commander_id}', i * 0.1)
    ax2.set_title('Commander to Artillery Communication')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Communication Count')
    ax2.legend()
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Enemies Destroyed by Artillery (All)
    for i, (artillery_id, destroyed) in enumerate(comm_stats['artillery_destroyed_enemies'].items()):
        plot_with_offset(ax3, destroyed, f'{artillery_id} (All)', i * 0.1)
    ax3.set_title('Enemies Destroyed by Artillery (All)')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Destroyed Enemy Count')
    ax3.legend()
    ax3.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Enemies Destroyed by Artillery (Communication-based)
    for i, (artillery_id, destroyed) in enumerate(comm_stats['artillery_comm_destroyed_enemies'].items()):
        plot_with_offset(ax4, destroyed, f'{artillery_id} (Comm-based)', i * 0.1)
    ax4.set_title('Enemies Destroyed by Artillery (Communication-based)')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Destroyed Enemy Count')
    ax4.legend()
    ax4.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'communication_stats.png'))
    plt.close()

    print(f"Communication stats plot saved in {os.path.abspath(plot_dir)}")


def plot_moving_average_rewards(episodes, friendly_rewards, enemy_rewards, window_size=100):
    import matplotlib.pyplot as plt
    import numpy as np

    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size), 'valid') / window_size

    friendly_ma = moving_average([r['total'] for r in friendly_rewards], window_size)
    enemy_ma = moving_average([r['total'] for r in enemy_rewards], window_size)

    plt.figure(figsize=(12, 6))
    plt.plot(range(window_size, len(episodes) + 1), friendly_ma, label='Friendly')
    plt.plot(range(window_size, len(episodes) + 1), enemy_ma, label='Enemy')
    plt.xlabel('Episode')
    plt.ylabel(f'Average Reward (Window Size: {window_size})')
    plt.title('Moving Average of Total Rewards')
    plt.legend()
    plt.grid(True)
    plt.savefig('moving_average_rewards.png')
    plt.close()


def plot_cumulative_rewards(episodes, friendly_rewards, enemy_rewards):
    import matplotlib.pyplot as plt
    import numpy as np

    friendly_cumulative = np.cumsum([r['total'] for r in friendly_rewards])
    enemy_cumulative = np.cumsum([r['total'] for r in enemy_rewards])

    plt.figure(figsize=(12, 6))
    plt.plot(episodes, friendly_cumulative, label='Friendly')
    plt.plot(episodes, enemy_cumulative, label='Enemy')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Rewards Over Episodes')
    plt.legend()
    plt.grid(True)
    plt.savefig('cumulative_rewards.png')
    plt.close()


def plot_team_metrics(episodes, friendly_rewards, enemy_rewards):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(episodes, [r['strength'] for r in friendly_rewards], label='Friendly Strength')
    plt.plot(episodes, [r['destruction'] for r in friendly_rewards], label='Friendly Destruction')
    plt.plot(episodes, [r['strength'] for r in enemy_rewards], label='Enemy Strength')
    plt.plot(episodes, [r['destruction'] for r in enemy_rewards], label='Enemy Destruction')
    plt.xlabel('Episode')
    plt.ylabel('Metric Value')
    plt.title('Team Strength and Destruction Over Episodes')
    plt.legend()
    plt.grid(True)
    plt.savefig('team_metrics.png')
    plt.close()


def plot_episode_length(episodes, episode_lengths):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(episodes, episode_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')
    plt.title('Episode Length Over Time')
    plt.grid(True)
    plt.savefig('episode_length.png')
    plt.close()