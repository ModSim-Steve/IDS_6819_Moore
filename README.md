# **Combined Arms MARL Environment w/DQN & Communication**
![Environment Screenshot](https://github.com/ModSim-Steve/IDS_6819_Moore/blob/main/Env_Screenshot.png)

## General Purpose
This project implements a multi-agent reinforcement learning environment simulating combined arms operations with communication capabilities. The environment models a battlefield scenario where different types of units (e.g., scouts, artillery, commanders) from two opposing teams interact. The key feature of this environment is the incorporation of a communication system that allows scouts to relay enemy positions to commanders, who then communicate this information to artillery units.

## Files
### Environment File 
#### *combined_arms_rl_env_MA_comms.py*

Implements the 'CombinedArmsRLEnvMAComms' class, which defines the state space, action space, and core environment dynamics.
Handles agent movement, combat, and communication.

### Deep Q-Network File
#### *combined_arms_dqn_agent_comms.py*

Implements the 'DQNAgentComms' class, which defines the Deep Q-Network (DQN) architecture and training process for individual agents.
Handles experience replay and target network updates.

### Opposing Side Training Configuration File
#### *combined_arms_TRN_configs_comms.py*

Implements 'OpposingConfigComms' class, which defines opponent behavior.
Provides functions for adjusting difficulty (pseudo curriculum training approach) and loading trained models for more difficult opponent.

### Training Process File
#### *combined_arms_TRN_process_comms.py*

Implements the main training loop by handling the episode iterations, reward calculations, and model updates.
Provides functions for logging and visualization of training results.

### Main File
#### *main_comms.py*

Entry point for running the training process.
Parses command-line arguments and sets up the training configuration

### Reward Debug File
#### *reward_debug_logger_comms.py*

Debugging tool for analyzing the reward structure.
Provides detailed logging of rewards and agent actions



## File Interactions

'*main_comms.py*' initializes the environment from '*combined_arms_rl_env_MA_comms.py*' and starts the training process defined in '*combined_arms_TRN_process_comms.py*'.
The training process uses agents defined in '*combined_arms_dqn_agent_comms.py*' and configurations from '*combined_arms_TRN_configs_comms.py*'.
The environment (*combined_arms_rl_env_MA_comms.py*) interacts with the agents during each step of the training process, providing observations and rewards.
'*reward_debug_logger_comms.py*' can be used independently to analyze the reward structure of the environment.

## Requirements
### Python Packages

- numpy
- tensorflow
- torch
- pygame
- matplotlib
- tqdm
- gymnasium

You can install these packages using pip:
`pip install numpy tensorflow torch pygame matplotlib tqdm gymnasium`

### Python Libraries
The following standard Python libraries are used and should be available in most Python installations:

- random
- time
- os
- logging
- argparse
- collections

### Hardware Requirements

For optimal performance, a CUDA-capable NVIDIA GPU is recommended.
Minimum 8GB RAM, 16GB or more recommended for larger simulations.
Multi-core CPU for parallel processing of agent actions.

### Software Requirements

- Python 3.7 or higher
- CUDA Toolkit (for GPU acceleration with TensorFlow and PyTorch)
- cuDNN library (for GPU acceleration)

### GPU Setup for Parallel Processing

- Install the NVIDIA GPU drivers appropriate for your GPU.
- Install CUDA Toolkit (version compatible with your TensorFlow and PyTorch installations).
- Install cuDNN library (version compatible with your CUDA Toolkit).
- Ensure your TensorFlow and PyTorch installations are GPU-enabled.

You can verify GPU availability in Python with:
```
import tensorflow as tf
print("TensorFlow GPU available:", tf.test.is_built_with_cuda())

import torch
print("PyTorch GPU available:", torch.cuda.is_available())
```

## Running the Environment
To start training, execute the following command line: 
```
python main_comms.py --episodes 1000 --base-difficulty 0 --train-friendly
```
Use `python main_comms.py --help` for a full list of command-line options.

### Note
This environment is computationally intensive. Training for a large number of episodes or with many agents may take a significant amount of time, even with GPU acceleration. Monitor system resources and adjust parameters as necessary.
