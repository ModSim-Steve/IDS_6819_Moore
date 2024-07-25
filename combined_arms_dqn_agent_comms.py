import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import random

# Check if PyTorch is available
print("PyTorch version: ", torch.__version__)

# Check if CUDA is available
print("Is CUDA available: ", torch.cuda.is_available())

if torch.cuda.is_available():
    # Print the CUDA version
    print("CUDA version:", torch.version.cuda)

    # Check if cuDNN is available
    print("Is cuDNN available:", cudnn.is_available())

    if cudnn.is_available():
        # Print the cuDNN version
        print("cuDNN version:", cudnn.version())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class DQN(nn.Module):
    """
    Deep Q-Network (DQN) model.

    This class defines the neural network architecture used for Q-value approximation
    in the DQN algorithm. It consists of several fully connected layers with ReLU activations.

    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Third fully connected layer.
        fc4 (nn.Linear): Output layer.
    """
    def __init__(self, state_shape, action_size):
        """
        Initialize the DQN model.

        Args:
            state_shape (tuple): The shape of the input state.
            action_size (int): The number of possible actions.
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(np.prod(state_shape), 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, action_size)

    def forward(self, x):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor representing the state.

        Returns:
            torch.Tensor: Output tensor representing Q-values for each action.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


class DQNAgentComms:
    """
    A Deep Q-Network (DQN) agent with communication capabilities for reinforcement learning in a combined arms environment.

    This agent implements the DQN algorithm with several enhancements, including a target network,
    prioritized experience replay, and epsilon-greedy exploration. It is designed to handle
    complex state spaces and action selections in a multi-agent communication-enabled environment.

    Attributes:
        state_shape (tuple): The shape of the input state.
        action_size (int): The number of possible actions.
        agent_id (str): Unique identifier for the agent.
        memory (PrioritizedReplayBuffer): Buffer for storing and sampling experiences.
        gamma (float): Discount factor for future rewards.
        epsilon (float): Exploration rate for epsilon-greedy policy.
        epsilon_min (float): Minimum exploration rate.
        epsilon_decay (float): Decay rate for epsilon.
        learning_rate (float): Learning rate for the optimizer.
        model (DQN): The main DQN model.
        target_model (DQN): The target network for stable Q-value estimation.
        optimizer (torch.optim.Optimizer): The optimizer for the DQN.
        steps (int): Counter for total steps taken.
        target_update_frequency (int): Frequency of target network updates.
    """
    def __init__(self, state_shape, action_size, agent_id):
        """
        Initialize the DQNAgentComms.

        Args:
            state_shape (tuple): The shape of the input state.
            action_size (int): The number of possible actions.
            agent_id (str): Unique identifier for the agent.
        """
        self.state_shape = state_shape
        self.action_size = action_size
        self.agent_id = agent_id
        self.memory = PrioritizedReplayBuffer(50000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.3
        self.epsilon_decay = 0.999
        self.learning_rate = 0.0001
        self.model = DQN(state_shape, action_size).to(device)
        self.target_model = DQN(state_shape, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()
        self.steps = 0
        self.target_update_frequency = 250

    def update_target_model(self):
        """
        Update the target network with the weights of the main network.
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def store_experience(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer.

        Args:
            state (np.array): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (np.array): The resulting state.
            done (bool): Whether the episode has ended.
        """
        self.memory.push(state, action, reward, next_state, done)

    def act(self, state):
        """
        Choose an action using an epsilon-greedy policy.

        Args:
            state (np.array): The current state.

        Returns:
            int: The chosen action.
        """
        if np.random.rand() <= self.epsilon:
            if self.agent_id.startswith(('friendly_scout', 'friendly_commander', 'friendly_artillery')):
                return random.randrange(self.action_size)  # All actions available
            else:
                return random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 10])  # All actions except communicate
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            act_values = self.model(state)
        return np.argmax(act_values.cpu().data.numpy())

    def replay(self, batch_size):
        """
        Train the network on a batch of experiences.

        Args:
            batch_size (int): The size of the batch to train on.
        """
        if len(self.memory) < batch_size:
            return  # Not enough experiences to learn from

        samples, indices, weights = self.memory.sample(batch_size)

        states, actions, rewards, next_states, dones = samples

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        weights = torch.FloatTensor(weights).to(device)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = (current_q_values.squeeze() - target_q_values).pow(2) * weights
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        new_priorities = np.abs((target_q_values - current_q_values.squeeze()).detach().cpu().numpy()) + 1e-5
        self.memory.update_priorities(indices, new_priorities)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.steps += 1
        if self.steps % self.target_update_frequency == 0:
            self.update_target_model()

    def load(self, name):
        """
        Load the neural network weights from a file.

        Args:
            name (str): The name of the file to load the weights from.
        """
        self.model.load_state_dict(torch.load(name, map_location=device))

    def save(self, name):
        """
        Save the neural network weights to a file.

        Args:
            name (str): The name of the file to save the weights to.
        """
        torch.save(self.model.state_dict(), name)


class PrioritizedReplayBuffer:
    """
    A prioritized experience replay buffer for reinforcement learning.

    This buffer stores experiences and assigns them priorities for sampling.
    Higher priority experiences have a higher probability of being sampled
    for training, which can lead to more efficient learning.

    Attributes:
        capacity (int): Maximum number of experiences that can be stored.
        memory (list): List of stored experiences.
        priorities (np.array): Array of priorities for each experience.
        position (int): Current position in the buffer for adding new experiences.
    """
    def __init__(self, capacity):
        """
        Initialize the PrioritizedReplayBuffer.

        Args:
            capacity (int): Maximum number of experiences that can be stored.
        """
        self.capacity = capacity
        self.memory = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer.

        If the buffer is full, the new experience replaces the oldest one.

        Args:
            state: The current state.
            action: The action taken.
            reward: The reward received.
            next_state: The resulting state.
            done: Boolean indicating if the episode has ended.
        """
        max_priority = self.priorities.max() if self.memory else 1.0
        if len(self.memory) < self.capacity:
            self.memory.append((state, action, reward, next_state, done))
        else:
            self.memory[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch of experiences from the buffer.

        The sampling is based on the priorities of the experiences,
        with higher priority experiences being more likely to be sampled.

        Args:
            batch_size (int): Number of experiences to sample.
            beta (float): Factor for importance sampling correction.

        Returns:
            tuple: Contains:
                - Batch of experiences (state, action, reward, next_state, done)
                - Indices of sampled experiences
                - Importance sampling weights
        """
        if len(self.memory) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]

        probs = priorities ** beta
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]

        weights = (len(self.memory) * probs[indices]) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones)), indices, weights

    def update_priorities(self, indices, priorities):
        """
        Update the priorities of specific experiences.

        Args:
            indices (list): Indices of the experiences to update.
            priorities (list): New priority values for the experiences.
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        """
        Get the current size of the buffer.

        Returns:
            int: The number of experiences currently in the buffer.
        """
        return len(self.memory)
