---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: post
title: "A gentle introduction to Reinforcement Learning, Deep Q-Learning Networks, and our Experiment"
---

{% include mathjax.html %}

# Reinforcement Learning
*Reinforcement learning* (RL) is a field in Artificial Intelligence (AI) that is concerned with agents interacting with their environment. An agent could be a robot in a maze, a player in a computer game, a programme in an automatic trading system, and many more things. In RL, there is a typical way of modelling the world. The environment of the agent is subdivided in *states*. A state can be observed by the agent, and it should contain all the information that the agent needs to base her decisions on. In every state, the agent can take some *actions*. The agent bases her actions on her *policy*, this is a function that says how probable it is for the agent to take an action in a state. Once the action is taken, the agent moves to a different state. She also receives a *reward*, which indicates how valuable the action was. The state and reward that are caused by the action do not need to be deterministic. If the agent takes the same action in the same state at a later point in time, the outcome may be different. Typically, agents get the opportunity to interact with the exact same environment multiple times. One interaction might end after a time limit, or when some goal is reached. The interaction is called an *episode*.

In the table below you can see an overview of the different concepts and the symbol that represents them.

| **RL concept** | **Symbol** |
|-|-|
| Current time step | $t$ |
| Time when episode ends | $T$ |
| All tates, current state   | $S$, $s_t$  |
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

For this blog post, we researched a popular Q-learing method, which is called Deep Q-Learning Network (DQN). You can find the original paper [here](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf). It combines three common elements to perform well: function approximation, bootstrapping and off-policy training. We will explain these elements later in more detail. For now, all you need to know that the combination of these three elements is dangerous. Using one or two is done often and works well. However, when all three are used there is a large risk that the returns will *diverge* when we iterate. This means that we end up with a useless Q-function. Because of this dangerous situation, the combination of these three elements is known as the *deadly triad*.

In order to make their method work, the inventors of DQN added some tricks. They claim that when we use these tricks, we don't have to worry about the deadly triad anymore.

You can imagine that this is quite a statement. Many years researchers have had to worry about not falling into this trap, and suddenly there is a paper that theoretically should be in it, but for some reason is not? Don't worry, you are not the only one who is suspicious. Surely we do not want to undermine the strength of the DQN model, we are however curious whether we could obtain similar results. This might seem evident, but reproducibility is a major problem in reinforcement learning. Remember the cookie problem from before? Think about the possible ways you could solve this issue. Naturally, there are many paths leading to the same amount of cookies, creating equal happiness to the robot. What is then the optimal thing to do?

We hope you are now at least a little bit skeptic before blindly accepting a newly proposed model, promising the holy grail of reinforcement learning. With this is mind, we will further investigate the DQN model in order to answer the following question:


> To what extend do the DQN tricks help to solve the deadly triad problem?

# A deep-dive into the pitfalls of DQN

We have mentioned the notion of the *deadly triad* before. Specifically, it mentions three assumptions that, in combination, can be "deadly" for the problem at hand. The term "deadly" actually refers to the concept of divergence, which is just a fancy way of saying that we did not find a solution for the problem. We will now dive into more detail on how this applies to the DQN model.

First of all, we look at function approximation. This is a popular technique that overcomes the issue of having to deal with large state spaces. You can imagine that for an increased number of states, it becomes more and more difficult to determine the value of a state. Instead, we try to approximate the value function, hence the name function approximation. In DQN, this corresponds to a deep convolutional network. They have implemented such a network to process the images of Atari games, where each image can be seen as a state.

Next, we turn to bootstrapping. If we are in a current state and we wish to find the optimal action, we need to have some knowlegde of the future state that this action will lead us to. We can look at this as our 'future value'; we do not have it right now, but we need to take it into account when choosing a path. Bootstrapping is a technique that includes future value by calculating the state-action value of the state we transition to. Other methods (like Monte Carlo) simply use the discounted rewards. The future value is included in the update rule. Now how does this translate to the DQN, you wonder? Take a closer look at the image below:

<img src="{{site.baseurl}}/assets/DQN-algorithm.png" width="450">

This is the pseudocode for the DQN algorithm. Hopefully, you will notice that the new state-action value is calculated as the sum of the reward and the subsequent state-action value. For this reason, DQN effectively makes use of bootstrapping.

The last assumption in the triad is off-policy learning. This is somewhat trivial for DQN since Q-learning methods are generally off-policy. However, we can validate this assumption by looking at the tricks of DQN.

# How can we make it work?

With the notion of the deadly triad and its effect on DQN, we can move on to the tricks. After having read the pitfalls of DQN you might wonder why this method can still perform so well on various games.

The strength of DQN lies in the clever way it deals with instabilities during training of reinforcement learning problems. It introduces two features: experience replay and periodic iterative update rules. The first feature, experience replay, stores the agent's experiences and subsequently draws random samples from this pool. The weight updates are then no longer based on sequential samples, which breaks the data dependency. Furthermore, since these updates occur periodically, it is also more robust to small alterations that severely change the policy. It is worth mentioning that the inclusion of experience smooths learning by averaging over previous states, and hereby conveniently avoids divergence of parameter estimates. In summary, the combination of both circumvents the correlations that are present in the data, as well as between action-values and target values.

Moreover, as the original DQN research specified a wide range of Atari games (49 to be exact) with all considerably diverse scores, they limited the range of rewards between -1 and 1. This means that all negative rewards are set to -1 and positive rewards are set to 1, leaving zero rewards unchanged.

It sounds more promising now, right? Let's check it out for ourselves by outlining the experiments.

# Experimental Design

We will attempt to grasp the importance of the model's assumptions as we believe these are essential for convergence. This includes experience replay, periodic iterative updates (similar to fixing the target for a previously defined number of steps) and reward clipping. We narrow our search down to the cartpole environment, available within the openAI gym.  

We investigate five experimental settings:

1. **Normal DQN (DQN)**: this is the normal model with experience replay, periodic iterative updates and reward clipping.
2. **No experience replay (-ER)**: the policy is updated in an "online" fashion. Particularly, we use S, A, S+1 and A+1 at the time of update.
3. **No periodic updates of target network (-TN)**: the target is defined by the current network, instead of a target network that is periodically updated.
4. **No reward clipping (-RC)**: the real rewards of the environment are used.
5. **Reward gain (R+)**: the rewards are not clipped, but are magnified by a factor 100 instead. The intuition behind this is that using magnified rewards hopefully leads to a more extreme result, making the role of reward clipping more obvious for DQN.

In the experiments, we expect our algorithm to perform worse when we leave components out. This indicates that the tricks are indeed necessary for proper training. We try to disprove this hypothesis by constantly disabling one of the tricks. If the results show that this does not significantly deteriorates performance, we have a contradiction and our hypothesis is incorrect.  

Naturally, we also have to tune the hyperparameters. Fortunately, the original DQN paper states that the same hyperparameter settings work for a diverse range of tasks. For that reason, we do not necessarily need to tune them for every setting. We tune only the learning rate, for which we try 3 different values. In each experimental setting, we report the results of the optimal learning rate for that setting. The batch size is fixed at 32, which is a typical value that works well under normal conditions. The target network is updated every **C** iterations. Every experiment is run with 10 different seeds, to measure the significance of our findings.

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
