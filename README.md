# Deep Reinforcement Learning - Udacity
## Project: Navigation
  
  
### Description
This repository contains the implemtation for training an agent to navigate around an environment, collecting bananas. The agent is rewarded one point for collecting a yellow banana, and negative one for collecting a blue banana. The goal of the agent is to perform in an intelligent way, collecting as many yellow bananas as possible. 

![Intro](https://github.com/Peter-Bugaj/udacity-navigation/blob/master/images/banana.gif)

#### Benchmark
The benchmark average score for the agent was set to +15 across a window of one hundred consecutive episodes. This benchmark was set slightly higher than the provided benchmark of +13 for the purpose of experimentation to see just how well the agent could perform.

#### Navigating
The agent was given the option of moving backwards, moving forwards, turning left, or turning right. It had a state space that consisted of 37 dimensions, which included the agent's velocity and a ray-based perception of objects around the agent's forward direction.
  
  
### Getting started
Start creating the environment by creating an isolated python environment (this should be done with python 3):

``bash
sudo -H pip3 install --upgrade pip
``

``bash
sudo -H pip3 install virtualenv
``

``bash
virtualenv drl_project
``

Next enable the environment and install Jupyter Notebook:

``bash
source drl_project/bin/activate
``

``bash
pip install jupyter
``

Finally run jupyter notebook as follows:

``bash
jupyter notebook
``

**Notes**
The python/requirements.txt contains the necessary libraries that need to be installed for the environment. As of writing this report, the python used for this project was downgraded to version 3.7.9, as this worked best with tensorflow. However you might need to update the requirements.txt file later in the future if libraries become deprecated, etc.
  
  
### Project Overview
The script for training and testing the agent is found in the file **Navigation.ipynb**. This file starts of by installing the appropriate libraries, starting the environment, and running a few basic experiments, like reading the state space of the environment, and doing sample steps. Later in the script the agent is tested with different epsilon values, is trained and tested with the variables leading to the best performance, and lastly the agent is tested with a double DQN implementation.

#### State space exploration
The agent had a state space of 37 dimensions, which included the agent's velocity and a ray-based perception of objects around the agent's forward direction. At any point within the environment, the agent had four actions which it could take: moving backwards, forwards, turning left or turning right.

#### Environment
The environment consisted of open space that allowed the agent to navigate around, collecting bananas. The agent was rewarded one point for collecting a yellow banana, and negative one for collecting a blue banana. This environment was downloaded from Unity, from the following list of links:

**Linux:** https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip

**Mac OSX:** https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip

**Windows (64-bit):** https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip

When running the code inside **Navigation.ipynb**, the environment is loaded using the following line:

``bash
env = UnityEnvironment(file_name="./Banana.app")
``

This was based on the assumption that the environment was downloaded and extracted for the Mac OS operating system, and placed under the root of the Jupyter project with the named **Banana.app**. Based on which environment you downloaded, you might need to change this line of code to point to the correct file/folder.

#### Model
The model was implemented based on the exercises provided by Udacity. The code can be found under the file **model.py**. It conisted of a neural network built from two hidden layers, plus an additional input layer for the taking in the states, and an output layer for providing the best estimate for each action.

#### Agent
The agent was implemented based on the exercise provided by Udacity. The code for it can be found in the file under **dqn_agent.py**. The only additional change made to this file was to support double DQN's, which was implemented based on an online article description here:

**https://medium.com/@qempsil0914/deep-q-learning-part2-double-deep-q-network-double-dqn-b8fc9212bbb2** 
  
  
### Training / Optimizations
The agent was trained by learning an optimal policy through a process called Deep Q-Learning, used for maximizing the rewards within the environment. This was done by the agent interacting within the environment hundreds of times, collecting observations along the way, and constantly updating the Q-function for mapping the states to the actions yielding the best reward.

#### 𝛆-greedy algorithm
A challenge encountered when updating this type of Q-function was deciding how the agent should choose to act while still in training. Should the agent always choose the action that best maximizes the result for each state, or should the agent explore, taking random action along the way? Choosing the action that results in the maximum value from the Q-function leads to the problem where the agent learns to ignore other alternatives. 

The 𝛆-greedy algorithm was therefore implemented by allowing the agent to do more exploration in the beginning, choosing random actions with the probability 𝛜, and only choosing the greedy options with probability 1-𝛜. As more and more episodes were iterated over, this epsilon value was decreased using a decay factor. This caused the agent to do less exploring over time and to become more greedy, as it gained more experience.

By default the starting epsilon value was set to one, meaning that the agent would always act completely at random at first. The decay factor was set to 0.995 with a minimum epsilon value set to 0.01. This allowed the agent to become greedy eventually after the two hundreth and fiftieth episode.

The act of choosing a random action over a greedy one was implemented in the **act** function in **dqn_agent.py**.

#### Experience replay
Another challenge when training the agent is that the sequential observations it was learning from had the possibility of being highly correlated. This increased the risk of the agent becoming increasingly biased, causing action values to oscilate and diverge.

The replay buffer saves the states, actions, rewards and next states as tuples, allowing them to be retrieved later on at random in the form of samples. These samples are referred to as the **experience**, hence the name. By replaying such experiences, the harmful correlations between the observations coming directly from the environment can be removed. It also observations to be re-used multiple times in training, allowing rare events to be replayed for example.

The act of choosing a random sample was implemented in the **step** function in **dqn_agent.py**. In the beginning the agent would only interact with the environment and store its observations inside a buffer. Once large enough, observations would be pulled from this buffer using the **sample()** function. These experiences would then be used for training the neural network.

#### Fixed Q-Targets
Another common problem when updating the network happened in the computation of the temporal differences and the resulting squared errors, used by the gradient optimizer.

In the simple case, the temporal difference is computed as the difference between the current predicted Q-value and the TD target, where the TD target is meant to be a replacement for the true value of the Q-function. Since the TD target is not known, it is approximated using the value output of the existing Q-function for the next state plus the current reward. This can sometimes lead to unpredictable results since the approximated target and current neural network will be updated at the same time.

To fixed this, the current and approximated target neural network were separated into a local Q-network and a target Q-network. The local Q-network was then used for predicting the best actions to take, and for populating the experience buffer. Every **n** number of steps, the experience buffer would be used for training, where the gradient error between the predicted actions and actions from the target network would be computed, and the target Q-network would be updated.

The target neural network was updated in the **learn** function in **dqn_agent.py**, and the actions from the local Q-network were prediced inside the **act** function.
  
  
#### Double DQN
Another issue addressed was the possible overestimation of action values. In the basic approach, the agent always tries to choose the best action for each given state. However in the beginning, the agent knows very little about the environment, and by choosing such actions, a lot of noise is introduced when learning, leading to over estimations in the update procedure later on.

Double DQN helps solve this problem by using two different Q-functions, Q and Q’. Function Q is used for selecting the best action with maximum value for the next state. Function Q' is then used for calculating the expected value, using the action selected by Q. Even if both Q and Q' are both noisy, their noise can be viewed as uniform distribution, as proven here:

**H. van Hasselt 2010, Section 3** https://papers.nips.cc/paper/3964-double-q-learning
  
  
### Experimentation
#### Benchmark
The benchmark was set to having an average score of +15, computed over the last one hundred episodes.

####  Test conducted.
A number of different experiments were executed, for trying out different values for the epsilon decay, as well as for the minimum value of epsilon. The minimum value of epsilon did not seem to have much impact, as was expected, however it did prevent the average score from reaching the benchmark is some cases. For example, the average score was never reached when the minimum epsilon value was set to 0.1, and instead the average score would start to fluctuate between 12 and 13.

However changing the epsilon decay did result in a noticeable difference. It was noted that by reducing the epsilon decay value to values such as 0.85, the agent became more greedy much faster over time. This also sometimes lead to the agent learning faster, receiving an average score of +13 in just 165 episodes, or an average score of +15 in 415 episodes! However such an agent did not do as well during testing.

#### Best performing agent
The best performing agent was discovered when the minimum epsilon value was kept at 0.01, with the epsilon decay factor set to 0.95. This allowed the agent to reach an average score of +15 in 292 episodes.

The graph showing how the agent trained is shown below:

![Average scores](https://github.com/Peter-Bugaj/udacity-navigation/blob/master/images/average-scores-best.png)

![Average scores graph](https://github.com/Peter-Bugaj/udacity-navigation/blob/master/images/average-scores-graph-best.png)

Further details of this performance are documented in the notebook itself, found in **Navigation.ipynb**
  
  
### Future  Work
#### Prioritized Learning
One option for improving the training the agent would be to prioritize the experience replay. Instead of choosing tuples only at random during the sampling process, it might be more helpful to choose only those most helpful for improving the agent. One way this could be accomplished is by assigning a larger probability to the tuples which resulted in a larger degree of error. This would allow the agent to pick up the experiences with bigger surprises.

#### Increasing the Q-network
Another option would be to add additional layers to the neural network, or by adding more nodes per layer. This could help the agent estimate the Q-function better. The expected result would be for the agent taking more time to learn, in exchange to it making more strategic choices.
