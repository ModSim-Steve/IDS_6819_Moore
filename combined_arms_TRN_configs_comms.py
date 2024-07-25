import random
import os
import tensorflow as tf


class OpposingConfigComms:
    """
    Configuration class for the opposing team in the Combined Arms environment with communication.

    This class defines the behavior of the opposing team, including movement patterns,
    engagement rules, and the option to use a trained model for decision-making.

    Attributes:
        movement_type (str): The type of movement pattern ('static', 'right_to_left', or 'free').
        engagement_type (str): The type of engagement behavior ('hold' or 'free').
        use_trained_model (bool): Whether to use a trained model for decision-making.
        trained_model (object): The trained model to use, if applicable.
        communication_enabled (bool): Whether communication is enabled for the opposing team.

    Methods:
        get_actions(num_agents, state=None): Get actions for the opposing team agents.
    """
    def __init__(self):
        """Initialize the OpposingConfigComms with default values."""
        self.movement_type = 'static'
        self.engagement_type = 'hold'
        self.use_trained_model = False
        self.trained_model = None
        self.communication_enabled = False  # Default to disabled

    def get_actions(self, num_agents, state=None):
        """
        Get actions for the opposing team agents.

        Args:
            num_agents (int): The number of agents to generate actions for.
            state (optional): The current state, used when a trained model is active.

        Returns:
            list: A list of actions for each agent.
        """
        actions = []
        for _ in range(num_agents):
            if self.use_trained_model and self.trained_model is not None:
                if state is None:
                    raise ValueError("State must be provided when using a trained model")
                action = self.trained_model.predict(state)[0]
            else:
                # Initialize movement_action and engagement_action with default values
                movement_action = 10  # Default to hold position
                engagement_action = 10  # Default to hold position

                # Determine movement action
                if self.movement_type == 'static':
                    movement_action = 10  # Hold position
                elif self.movement_type == 'right_to_left':
                    movement_action = random.choice([2, 4, 6])  # Movement from right to left (actions 2, 4, or 6)
                elif self.movement_type == 'free':
                    movement_action = random.randint(0, 7)  # Random movement

                # Determine engagement action
                if self.engagement_type == 'hold':
                    engagement_action = 10  # Hold position
                elif self.engagement_type == 'free':
                    engagement_action = 8  # Fire

                # Choose between movement and engagement
                action = random.choice([movement_action, engagement_action])

                # Communication action (9) is only available if communication is enabled
                if self.communication_enabled:
                    action = random.choice([action, 9])  # Add possibility of communication

            actions.append(action)

        return actions


def adjust_difficulty_comms(opposing_config, base_difficulty, episode, total_episodes, against_friendly=True):
    """
    Adjust the difficulty of the opposing team based on the current episode.

    This function modifies the opposing team's configuration to progressively
    increase difficulty as training progresses.

    Args:
        opposing_config (OpposingConfigComms): The current configuration of the opposing team.
        base_difficulty (int): The starting difficulty level.
        episode (int): The current episode number.
        total_episodes (int): The total number of episodes in the training run.
        against_friendly (bool): Whether the opposing team is playing against the friendly team.

    Returns:
        OpposingConfigComms: The updated opposing team configuration.
    """
    difficulty_levels = ['static_wpns_hold', 'static_wpns_free', 'R-L_MTC', 'trained_model']

    # For very short runs, just use the base difficulty
    if total_episodes < len(difficulty_levels):
        current_level_index = base_difficulty
    else:
        # Calculate the number of episodes per difficulty level
        episodes_per_level = max(1, total_episodes // len(difficulty_levels))
        # Determine the current difficulty level based on the episode number and base_difficulty
        current_level_index = min(base_difficulty + (episode // episodes_per_level), len(difficulty_levels) - 1)

    current_difficulty = difficulty_levels[current_level_index]

    return update_opposing_config_comms(opposing_config, current_difficulty, against_friendly)


def update_opposing_config_comms(opposing_config, difficulty, against_friendly):
    """
    Update the opposing team's configuration based on the specified difficulty level.

    This function sets specific behaviors and capabilities for the opposing team
    according to the current difficulty level.

    Args:
        opposing_config (OpposingConfigComms): The current configuration of the opposing team.
        difficulty (str): The difficulty level to set ('static_wpns_hold', 'static_wpns_free', 'R-L_MTC', or 'trained_model').
        against_friendly (bool): Whether the opposing team is playing against the friendly team.

    Returns:
        OpposingConfigComms: The updated opposing team configuration.
    """
    if difficulty == 'static_wpns_hold':
        opposing_config.movement_type = 'static'
        opposing_config.engagement_type = 'hold'
        opposing_config.use_trained_model = False
        opposing_config.communication_enabled = False
    elif difficulty == 'static_wpns_free':
        opposing_config.movement_type = 'static'
        opposing_config.engagement_type = 'free'
        opposing_config.use_trained_model = False
        opposing_config.communication_enabled = False
    elif difficulty == 'R-L_MTC':
        opposing_config.movement_type = 'right_to_left'
        opposing_config.engagement_type = 'free'
        opposing_config.use_trained_model = False
        opposing_config.communication_enabled = False
    elif difficulty == 'trained_model':
        opposing_config.use_trained_model = True
        model_path = 'friendly_model.h5' if against_friendly else 'enemy_model.h5'
        opposing_config.trained_model = load_trained_model(model_path)
        opposing_config.communication_enabled = True  # Enable communication only for trained model
        if opposing_config.trained_model is None:
            # Fallback to previous difficulty if model isn't available
            opposing_config.movement_type = 'right_to-left'
            opposing_config.engagement_type = 'free'
            opposing_config.use_trained_model = False
            opposing_config.communication_enabled = False

    return opposing_config


def load_trained_model(model_path):
    """
    Load a trained model from a file.

    This function attempts to load a pre-trained model to be used by the opposing team.

    Args:
        model_path (str): The file path of the trained model.

    Returns:
        object: The loaded model if successful, None otherwise.
    """
    if not os.path.exists(model_path):
        print(f"Warning: Model file {model_path} not found. Using default behavior.")
        return None
    return tf.keras.models.load_model(model_path)
