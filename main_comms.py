import argparse
import time
import logging
import torch
from combined_arms_rl_env_MA_comms import CombinedArmsRLEnvMAComms
from combined_arms_TRN_configs_comms import OpposingConfigComms
from combined_arms_TRN_process_comms import train_model
from combined_arms_dqn_agent_comms import DQNAgentComms


def main():
    """
    Main function to set up and run the training process.

    This function parses command-line arguments, initializes the environment and agents,
    and starts the training process. It also handles logging of training progress and results.
    """
    parser = argparse.ArgumentParser(description="Train agents using DQN with communication")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Total episodes for training")
    parser.add_argument("--base-difficulty", type=int, default=0,
                        help="Base difficulty level")
    parser.add_argument("--train-friendly", action="store_true",
                        help="Train friendly agents (default is to train enemy)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Number of episodes without improvement before early stopping")
    parser.add_argument("--min-episodes", type=int, default=400,
                        help="Minimum number of episodes before early stopping")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if not args.debug else logging.DEBUG,
                        format='%(message)s')

    file_handler = logging.FileHandler('saved_plots/20240723_175329_Test_wComms/training.log')
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    logging.getLogger().addHandler(file_handler)

    def log_section(message):
        logging.info(f"\n{'=' * 80}\n{message}\n{'=' * 80}")

    log_section("Starting training with communication enabled")
    log_section(f"Total episodes: {args.episodes}")
    log_section(f"Base difficulty: {args.base_difficulty}")
    log_section(f"Training {'friendly' if args.train_friendly else 'enemy'} agents")
    log_section(f"Patience: {args.patience}")
    log_section(f"Minimum episodes: {args.min_episodes}")
    log_section(f"Debug mode: {'ON' if args.debug else 'OFF'}")

    friendly_config = {
        "artillery": {
            "count": 2,
            "positions": [(3, 0), (6, 0)]
        },
        "scout": {
            "count": 2,
            "positions": [(3, 3), (6, 3)]
        },
        # "infantry": {
        #     "count": 1,
        #     "positions": [(5, 1)]
        # },
        # "light_tank": {
        #     "count": 3,
        #     "positions": [(1, 2), (4, 2), (8, 2)]
        # },
        "commander": {
            "count": 1,
            "positions": [(4, 1)]
        }
    }

    enemy_config = {
        "artillery": {
            "count": 2,
            "positions": [(3, 64), (6, 64)]
        },
        "scout": {
            "count": 2,
            "positions": [(3, 61), (6, 61)]
        },
        "infantry": {
            "count": 1,
            "positions": [(5, 63)]
        },
        "light_tank": {
            "count": 3,
            "positions": [(1, 62), (4, 62), (8, 62)]
        },
        "commander": {
            "count": 1,
            "positions": [(4, 63)]
        }
    }

    env = CombinedArmsRLEnvMAComms(friendly_config, enemy_config, render_speed=0.1, max_steps=2000, render_mode='human')

    initial_state = env.reset()
    state_shape = initial_state[list(initial_state.keys())[0]].shape
    action_size = env.action_space.n

    log_section(f"State shape: {state_shape}")
    log_section(f"Action size: {action_size}")
    log_section(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    opposing_config = OpposingConfigComms()

    start_time = time.time()

    friendly_rewards, enemy_rewards, episode_lengths, best_models, last_models, save_folder, plot_dir = train_model(
        env,
        args.episodes,
        opposing_config,
        base_difficulty=args.base_difficulty,
        train_friendly=args.train_friendly,
        agent_class=DQNAgentComms,
        patience=args.patience,
        min_episodes=args.min_episodes,
        debug=args.debug
    )

    total_time = time.time() - start_time
    log_section(f"\nTraining completed in {total_time / 60:.2f} minutes")
    log_section(f"Best models and last models saved in: {save_folder}")
    log_section(f"Plots saved in: {plot_dir}")

    # Optional: Add code here to evaluate the trained models or visualize results


if __name__ == "__main__":
    main()
