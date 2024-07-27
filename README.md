# **Combined Arms MARL Environment w/DQN & Communication**
![Environment Screenshot](https://github.com/ModSim-Steve/IDS_6819_Moore/blob/main/Env_Screenshot.png)

## General Purpose
This project implements a multi-agent reinforcement learning environment simulating combined arms operations with communication capabilities. The environment models a battlefield scenario where different types of units (e.g., scouts, artillery, commanders) from two opposing teams interact. The key feature of this environment is the incorporation of a communication system that allows scouts to relay enemy positions to commanders, who then communicate this information to artillery units.

![Project Presentation](https://youtu.be/KC1DgFCKj74)

## Files
### Environment File 
![combined_arms_rl_env_MA_comms.py](https://github.com/ModSim-Steve/IDS_6819_Moore/blob/main/combined_arms_rl_env_MA_comms.py)

Implements the 'CombinedArmsRLEnvMAComms' class, which defines the state space, action space, and core environment dynamics.

Handles agent movement, combat, and communication.

### Deep Q-Network File
![combined_arms_dqn_agent_comms.py](https://github.com/ModSim-Steve/IDS_6819_Moore/blob/main/combined_arms_dqn_agent_comms.py)

Implements the 'DQNAgentComms' class, which defines the Deep Q-Network (DQN) architecture and training process for individual agents.

Handles experience replay and target network updates.

### Opposing Side Training Configuration File
![combined_arms_TRN_configs_comms.py](https://github.com/ModSim-Steve/IDS_6819_Moore/blob/main/combined_arms_TRN_configs_comms.py)

Implements 'OpposingConfigComms' class, which defines opponent behavior.

Provides functions for adjusting difficulty (pseudo curriculum training approach) and loading trained models for more difficult opponent.

### Training Process File
![combined_arms_TRN_process_comms.py](https://github.com/ModSim-Steve/IDS_6819_Moore/blob/main/combined_arms_TRN_process_comms.py)

Implements the main training loop by handling the episode iterations, reward calculations, and model updates.

Provides functions for logging and visualization of training results.

### Main File
![main_comms.py](https://github.com/ModSim-Steve/IDS_6819_Moore/blob/main/main_comms.py)

Entry point for running the training process.

Parses command-line arguments and sets up the training configuration

### Reward Debug File
![reward_debug_logger_comms.py](https://github.com/ModSim-Steve/IDS_6819_Moore/blob/main/reward_debug_logger_comms.py)

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

## Demo
  ![Demo-gif](https://github.com/ModSim-Steve/IDS_6819_Moore/blob/main/friendly_vs_RLMTC_last-ezgif.com-video-to-gif-converter.gif)

  *A demonstration of a how trained models (friendly agents) interact within the environment against one of the opponent configurations ('R-L_MTC').*
  
- Blue Boxes represent the friendly agents 
- Red Boxes represent the enemy agents
- Letters w/in agents indicate their type S - Scout / A - Artillery / L - Light Tank / I - Infantry / C - Commander
- Yellow to Dark Orange shaded boxes represents the agents' visibility range (color scale indicates overlapping visibility)
- Red Rings indicate the agents' engagement range
- Small Red Circles indicate when an agent identifies a target (temporarily appear)
- Red Lines indicate firing from enemy agents (temporarily appear)
- Green Lines indicate firing from friendly agents (temporarily appear)
- Cyan Lines indicate communication from scouts to commander (temporarily appear)
- Yellow lines indicate communication from commander to artillery (temporarily appear)
- Black Boxes represent destroyed agents

#### Demo Video
![Demo](https://github.com/ModSim-Steve/IDS_6819_Moore/blob/main/friendly_vs_RLMTC_last.mp4)

## Future Experiments
- Communication:
  - Global Observations w/Local Observations. Instead of directing a protocol for communication [**kill chain:** scout (*detect enemy*) --> commander --> artillery (*engage enemy*)], determine if the agents can develop one based soley on sending / accessing information to / from the 'Global' observations.  The idea is that the agents will learn how to exploit each of the various agents' attributes (observation range & engagement range) given the basic reward structure (no communication rewards).  The initial experiment is essentially RIAL where communication is an action.  This additional experiment is essentially DIAL.  Metrics to monitor - computational time per episode, computational time per training session (how many episodes are required to reach optimality), difference in performance (how many agents survive the assault - average over 100 episode test).  *See Jakob Foerster's "Learning to Communicate with Deep Multi-Agent Reinforcement Learning" for details regarding the inspiration for this experiment. ![Paper](https://github.com/ModSim-Steve/IDS_6819_Moore/blob/main/03p_2016_Foerster_Learning-to-Communicate-with-Deep-MARL.pdf)*
  - Hierarchical Structure. Determine if differing groups of agents who have been trained to learn different policies (fires & manuever) can develop a "combined arms" policy through the use of graph neural networks.  The idea is that the 2 - 3 different groups of agents will learn to reduce the opposing force before conducting the assault.  *See Junji Sheng et al's "Learning Structured Communication for Multi-Agent Reinforcement Learning"*

- Environment Adjustments:
  - Reward Shaping: Determine the best method of incentivizing the agents to reach the desired outcome (without too much bias).  From the initial experiment's results (demo video above), it appeared as if the agents were not leveraging the other's unqiue attributes (scout's observation range & artillery's engagement range) and they were being too conservative (defensive).  With a few modifications to the reward structure (increased detection reward, increased firing reward, increased communication rewards, and modification to the team reward weights), the agents were able to successfully repel the assault.
    
    ![Increased Reward Demo](https://github.com/ModSim-Steve/IDS_6819_Moore/blob/main/friendly_vs_RLMTC_last_diffrewards-ezgif.com-video-to-gif-converter.gif)

    Although the results were better than the initial experiment, additional increased adjustments to the artillery's rewards might entice the agents to destroy the enemy faster.   

  - Environment Adjustments:
      - Add concealment / terrain to mask movements and extend observation range.  Determine if the agents can identify key terrain.
      - Add objectives.  Determine if the presense of objectives affects selected avenue of approach.
      - Force Ratio.  Determine if increased friendly agents affects the selection of offense / defense approach.
