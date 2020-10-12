# Deep Reinforcement Learning - Udacity
## Project: Navigation

### Description
This repository contains the implemtation for training an agent to navigate around an environment, collecting bananas. The agent is rewarded one point for collecting a yellow banana, and negative one for collecting a blue banana. The goal of the agent is to perform in an intelligent way, collecting as many yellow bananas as possible. 

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
The script for training and testing the agent is found in the file **Navigation.ipynb**. This file starts of by installing the appropriate libraries, starting the environment, and running a few basic experiments, like reading the state space of the environment, and doing sample steps. Later in the script the agent is tested with different epsilon values, is trained and tested with the variables leading to the best performance, and lastly the agent is test with a double DQN implementation.

#### State space exploration
The agent had a state space of 37 dimensions, which included the agent's velocity and a ray-based perception of objects around the agent's forward direction. At any point within the environment, the agent had four action which it could take: moving backwards, forwards, turning left or turning right.

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

Double DQN helps solve this problem by using two different Q-functions, Q and Q’. Even if both Q and Q' are both noisy, their noise can be viewed as uniform distribution. The proof of this can be found here:
**H. van Hasselt 2010, Section 3** https://papers.nips.cc/paper/3964-double-q-learning

Function Q is used for selecting the best action with maximum for the next state. Function Q' is then used for calculating the expected value, using the action selected by Q.





4. Run Experiments
Now that the various components of our algorithm are in place, it's time to measure the agent's performance within the Banana environment. Performance is measured by the fewest number of episodes required to solve the environment.

The table below shows the complete set of experiments. These experiments compare different combinations of the components and hyperparameters discussed above. However, note that all agents utilized a replay buffer.



 
5. Select best performing agent
The best performing agents were able to solve the environment in 200-250 episodes. While this set of agents included ones that utilized Double DQN and Dueling DQN, ultimately, the top performing agent was a simple DQN with replay buffer.



The complete set of results and steps can be found in this notebook.

Also, here is a video showing the agent's progress as it goes from randomly selecting actions to learning a policy that maximizes rewards.



 
Future Improvements
Test the replay buffer — Implement a way to enable/disable the replay buffer. As mentioned before, all agents utilized the replay buffer. Therefore, the test results don't measure the impact the replay buffer has on performance.
Add prioritized experience replay — Rather than selecting experience tuples randomly, prioritized replay selects experiences based on a priority value that is correlated with the magnitude of error. This can improve learning by increasing the probability that rare and important experience vectors are sampled.
Replace conventional exploration heuristics with Noisy DQN — This approach is explained here in this research paper. The key takeaway is that parametric noise is added to the weights to induce stochasticity to the agent's policy, yielding more efficient exploration.