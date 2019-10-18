---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: post
title: "A gentle introduction to Reinforcement Learning, Deep Q-Learning Networks, and our Experiment"
---

{% include mathjax.html %}

# Reinforcement Learning
*Reinforcement learning* (RL) is a field in Artificial Intelligence (AI) that is concerned with agents interacting with their environment. An agent could be a robot in a maze, a player in a computer game, a programme in an automatic trading system, and many more things. In RL, there is a typical way of modelling the world. The environment of the agent is subdivided in *states*. A state can be observed by the agent, and it should contain all the information that the agent needs to base her decisions on. In every state, the agent can take some *actions*. The agent bases her actions on her *policy*, this is basically a mapping of perceived states to the taken action in the current state. Once the action is taken, the agent moves to a different state. She also receives a *reward*, which can be seen as feedback for choosing a certain action. The state and reward that are caused by the action do not need to be deterministic. If the agent takes the same action in the same state at a later point in time, the outcome may be different. Typically, agents get the opportunity to interact with the exact same environment multiple times. One interaction might end after a time limit, or when some goal is reached. The interaction is called an *episode*.

In the table below you can see an overview of the different concepts and the symbol that represents them.

| **RL concept** | **Symbol** |
|-|-|
| Current time step | $t$ |
| Time when episode ends | $T$ |
| All states, current state   | $S$, $s_t$  |
| All actions, current action | $A$, $a_t$  |
| Current reward          | $r_t$       |
| Policy          | $$\pi$$ |

# The Cookie Collector
Let's make this a bit more concrete. Let's say we have a cookie-collecting robot in a world that has cookies laying around all over the place. The robot can move in four directions (up, down, left, right). She always knows its position in the world, and uses that information to determine where to go next. When she finds a cookie, she gets happy, but she's always looking for more. Her battery lasts two minutes, after which somebody will recharge her and let her play the game again from the start.

The components of the cookie collecting robot and her environment fit nicely in the RL framework. The table shows what form each RL concept takes in this example. Because the example fits in the RL framework, it means that we can apply RL methods to make the robot as succesful as possible, in this case: to make it collect as many cookies as possible within two minutes.


| **RL concept**  | **Cookie Collector**                                      |
|-----------------|-----------------------------------------------------------|
| Agent           | Cookie collector                                          |
| States          | Observations: position (e.g. x and y coordinate)                  |
| Actions         | Possible movements: up, down, left, right                 |
| Reward          | Zero normally, 'Unit of happiness' (e.g. 1) when she moves to a state with a cookie |
| Policy          | The process of deciding on an action to take |
| Time step       | Amount of time it takes to move to a next state, e.g. 1 second |
| Episode         | Period of 2 minutes                                       |


# The best policy
As mentioned before, the agent is interested in getting as much value out of the episode as possible. In formal terms, we want to find the policy which maximizes the expected total reward in the episode. When the agent is in a certain state, she wants to choose the action of which she expects that it will deliver her the most value. This does not only depend on the immediate reward. The agent should also look ahead and choose actions that brings the agent closer to states with high rewards (e.g. closer to a cookie).

To make this decision, we want to have a state-action value function $Q(s,a)$. This function tells us how much value we can expect when we take action $a$ in state $s$. When she has this function, the agent can simply observe her state $s$, and then choose the $a$ which has highest $Q(s,a)$. What remains is the challenge of finding this state-action value function.

# DQN and the Deadly Triad
Methods to find the state-action value function are called Q-learning methods, and they work in an iterative way. First, an initial setting of the Q-values of each state-action pair is chosen. If this is used as basis for a policy, it performs very bad. In Q-learning, we iteratively update the Q-values (or parameters of our model that gives us Q-values) to make the policy better and better. If the method works, the rewards should converge to a high value if we iterate long enough.

For this blog post, we researched a popular Q-learing method, which is called Deep Q-Learning Network (DQN). You can find the original paper [here](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf). It combines three common elements to perform well: function approximation, bootstrapping and off-policy training. We will explain these elements later in more detail. For now, all you need to know that the combination of these three elements is dangerous. Using one or two is done often and works well. However, when all three are used there is a large risk that the weights of the model do not *converge* to a fixed point, but *diverge* during learning. Since the model is an estimation of the return and we cannot find good parameters, we end up with a useless Q-function. Because of this dangerous situation, the combination of these three elements is known as the *deadly triad*.

In order to make their method work, the inventors of DQN added some tricks. They claim that when we use these tricks, we don't have to worry about the deadly triad anymore.

You can imagine that this is quite a statement. Many years researchers have had to worry about not falling into this trap, and suddenly there is a paper that theoretically should be in it, but for some reason is not? Don't worry, you are not the only one who is suspicious. Surely we do not want to undermine the strength of the DQN model, we are however curious whether we could obtain similar results. This might seem evident, but reproducibility is a major problem in reinforcement learning. Remember the cookie problem from before? What if the researchers were lucky and initialized the robot with decent Q values? Or if they "made themselves" lucky by trying and initialising it several times until they succeded? How can we know if the cookie robot is indeed that good in finding  cookies or the researches might be putting cookies right under the robots nose?

We hope you are now at least a little bit skeptic before blindly accepting a newly proposed model. With this is mind, we will further investigate the DQN model in order to answer the following question:


> To what extend do the DQN tricks help to solve the deadly triad problem?

# A deep-dive into the pitfalls of DQN

We have mentioned the notion of the *deadly triad* before. Specifically, it mentions three assumptions that, in combination, could be "deadly" for the problem at hand. The term "deadly" actually refers to the concept of divergence, which is just a fancy way of saying that we did not find a solution for the problem. Note that the conditions for the deadly triad do not **necessarily** lead to divergence, it just means that a problem is **likely** to. We will now dive into more detail on how this applies to the DQN model.

First of all, we look at function approximation. This is a popular technique that overcomes the issue of having to deal with large state spaces. You can imagine that for an increased number of states, it becomes more and more difficult to determine the value of a state. Instead, we try to approximate the value function, hence the name function approximation. In DQN, this corresponds to a deep convolutional network. They have implemented such a network to process the images of Atari games, where each image can be seen as a state.

Next, we turn to bootstrapping. If we are in a current state and we wish to find the optimal action, we need to have some knowlegde of the future state that this action will lead us to. We can look at this as our 'future value'; we do not have it right now, but we need to take it into account when choosing a path. Bootstrapping is a technique that includes future value by estimating the state-action value of the state we transition to. Other methods (like Monte Carlo) simply use the discounted rewards. The future value is included in the update rule. Now how does this translate to the DQN, you wonder? Take a closer look at the image below:

<img src="{{site.baseurl}}/assets/DQN-algorithm.png" width="450">

This is the pseudocode for the DQN algorithm. Look for the line where the value for $y_j$ (corresponding to the estimation of the state-value) is set. The code next to it contains the update rule, where you will hopefully notice that the new estimate is calculated as the sum of the reward and the subsequent state-action estimate. For this reason, DQN effectively makes use of bootstrapping.

The last assumption in the triad is off-policy learning. This is somewhat trivial for DQN since Q-learning methods are generally off-policy. It refers to the learning of a policy without using the policy to generate data. We can validate the choice for off-policy learning by looking at the tricks of DQN.

# How can we make it work?

With the notion of the deadly triad and its effect on DQN, we can move on to the tricks. After having read the pitfalls of DQN you might wonder why this method can still perform so well on various games.

The strength of DQN lies in the clever way it deals with instabilities during training of reinforcement learning problems. It introduces two features: experience replay and periodic iterative update rules. The first feature, experience replay, stores the agent's experiences and subsequently draws random samples from this pool. One of the reason we do so is because the data is highly correlated. Remember that our data are merely sequential images that form the entire game we play. Naturally, one image is highly depending on the previous image. This is not a convenient property to have, so we try to break it by instead sampling from the pool of experiences. The second reason is that we feed the images through a neural network, which has many parameters to train. It would be inefficient to throw all of that work away after one step. Rather, we store it in a buffer and learn from it multiple times.

We turn to the second feature: periodic iterative update rules. We can also look at this as specifying a fixed target. The main problem is that we want to come as close as possible to the target, but as we move, the target moves as well. Basically, it is like a cat chaising its own tail. The network would show many oscillations during training, making it very unstable. In order to overcome this problem, DQN specifies two networks. We have the original Deep Q-network and a second target network. The second one is introduced to update targets periodically, smoothing the learning process.

Moreover, as the original DQN research specified a wide range of Atari games (49 to be exact) with all considerably diverse scores, they limited the range of rewards between -1 and 1. This means that all negative rewards outside the boundary are set to -1 and positive rewards that are bigger than 1 are set to 1. Especially if we want to compare results between certain games, it is convenient to equalize the range of the rewards.

It sounds more promising now, right? Let's check it out for ourselves by outlining the experiments.

# Experimental Design

We will attempt to grasp the importance of the model's assumptions as we believe these are essential for convergence. This includes experience replay, periodic iterative updates (similar to fixing the target for a previously defined number of steps) and reward clipping. We narrow our search down to the [CartPole](https://gym.openai.com/envs/CartPole-v0/) environment, available within the openAI gym. We did not have a specific preference for an environment, the openAI gym has many that are suitable for DQN. However, CartPole is a fairly simplistic one that does not take too much time to solve. Also, it has states that are represented by continuous numbers, hence we need to use function approximators (like DQN). Sadly, we could not test other environments, because of the time constraint. Therefore, we focus on the specific settings of one particular environment.

We investigate the following five experiments:

1. **Normal DQN (DQN)**: this is the normal model with experience replay, periodic iterative updates and reward clipping.
2. **No experience replay (-ER)**: the policy is updated in an "online" fashion. Particularly, we use S, A, S+1 and A+1 at the time of update.
3. **No periodic updates of target network (-TN)**: the target is defined by the current network, instead of a target network that is periodically updated.
4. **No reward clipping (-RC)**: the real rewards of the environment are used.
5. **Reward gain (R+)**: the rewards are not clipped, but are magnified by a factor 100 instead. The intuition behind this is that using magnified rewards hopefully leads to a more extreme result, making the role of reward clipping more obvious for DQN.

In the experiments, we expect our algorithm to perform worse when we leave components out. This indicates that the tricks are indeed necessary for proper training. We try to disprove this hypothesis by constantly disabling one of the tricks. If the results show that this does not significantly deteriorates performance, we have a contradiction and our hypothesis is incorrect.  

As for the architecture of our neural network, we went for something less fancy than a convolutional neural network, since we do not have enough compute and we simply do not need it to do our experiments. We have used a two layer feed-forward neural network with ReLU activation at the hidden state the size of hidden and output states of 256. Naturally, we also have to tune the hyperparameters. Fortunately, the original DQN paper states that the same hyperparameter settings work for a diverse range of tasks. For that reason, we do not necessarily need to tune them for every setting. We tune only the learning rate, for which we try 3 different values. In each experimental setting, we report the results of the optimal learning rate for that setting. The batch size is fixed at 32, which is a typical value that works well under normal conditions. The target network is updated every 10 iterations. Every experiment is run with 10 different seeds, to measure the significance of our findings.

Our main question is whether the DQN network converges to a solution under the different experimental settings. To measure convergence, we compute the parameter gradient norm in each iteration. The minimal norm (MN) is an indication of convergence. If it is close to zero, it means that the policy hardly changed at some point during training. To have intuition on the variance of this norm over seeds, we report its average, standard deviation, and its minimum and maximum over seeds.

Because convergence does not mean that a globally optimal solution is found, we also report various quantities related to the specific environments, one of which is the cumulative reward at each time step.

Our results will be summarized in a table like below for every environment:

<!-- **Results for Environment ...** -->

|      | avg MN | std MN | min MN | max MN | Task specific metrics |
|------|--------|--------|--------|--------|-----------------------|
|  DQN |        |        |        |        |                       |
| -ER  |        |        |        |        |                       |
| -TN  |        |        |        |        |                       |
| -RC  |        |        |        |        |                       |
| +R   |        |        |        |        |                       |

The implementation of the DQN model is based on [this tutorial for pytorch](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) and code written for an assignment for the course Reinforcement Learning at the University of Amsterdam (2019). You can find the code here: ...

# Results

# Conclusion and Discussion

The most unexpected result of the experiments is that the DQN without experience replay (-ER) actually performs better than the full DQN! It seems that the -ER model is the only one that converges. Interestingly, -ER is only converging for some seeds, but not all, as you can see in the figure below. This illustrates how big an influence the choice of the seeds can be on your experiments.

*TODO: FIGURE OF DIFFERENT SEEDS -ER*

How can it be that this is the only model that converges?

Code quality is another important factor which can change the performance of an algorithm and is important to take into account when testing an algorithm. One noticable difference between -ER and the other DQN adaptations, is that -ER only trains it network on the current experience in an online fashion, while the other ones update with a batch-size of 32 for a random sample from all previous experiences. 

However, all DQN models with experience replay only learn when there is enough memory and skip learning otherwise. This is a convenient way to implement experience replay, however, it comes with the catch that the learning only happens after 32 steps have been taken in the first episode. An alternative implementation could be to implement online learning until there is enough memory, however, then we would break the i.i.d. assumption for part of the episode, because the data are sequentially dependent now. We, however, do not think that this additional head start can explain the big difference between the models. The larger batch-size also means that the same experiences are used more often than once, which could result in overfitting; complicating the learning even more.

The hyperparameters of training a model, such as the learning-rate, can make the difference between good and bad learning. A too large learning rate can result in divergence in normal supervised learning. Reinforcement Learning makes this more complicated, as the model interacts with a dynamic changing environment. This makes it very difficult to find the right hyperparameters. Since we have only done a limited hyperparameter search, it could well be that for this reason we could not find any converging models for the DQN with experience replay.

*TODO: further recommendations? Hyperparameter search? smaller batch size?*
