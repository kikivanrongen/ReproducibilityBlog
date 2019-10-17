---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---

A gentle introduction to Reinforcement Learning, Deep Q-Learning Networks and our Experiment

## Reinforcement Learning
*Reinforcement learning* (RL) is a field in Artificial Intelligence (AI) that is concerned with agents interacting with their environment. An agent could be a robot in a maze, a player in a computer game, a programme in an automatic trading system, and many more things. In RL, there is a typical way of modelling the world. The environment of the agent is subdivided in *states*. A state can be observed by the agent, and it should contain all the information that the agent needs to base her decisions on. In every state, the agent can take some *actions*. The agent bases her actions on her *policy*, this is a function that says how probable it is for the agent to take an action in a state. Once the action is taken, the agent moves to a different state. She also receives a *reward*, which indicates how valuable the action was. The state and reward that are caused by the action do not need to be deterministic. If the agent takes the same action in the same state at a later point in time, the outcome may be different. Typically, agents get the opportunity to interact with the exact same environment multiple times. One interaction might end after a time limit, or when some goal is reached. The interaction is called an *episode*.

In table **TABLENR**, you can see an overview of the different concepts and the symbol that represents them. In figure **FIGNR** you can see a diagram of the described process. Take a look at [**this link**] if you want to read more about these concepts.

| **RL concept** | **Symbol** |
|-|-|
| Current time step | $t$ |
| Time when episode ends | $T$ |
| All tates, current state   | $S$, $s_t$  |
| All actions, current action | $A$, $a_t$  |
| Current reward          | $r_t$       |
| Policy          | $\pi$ |

## The Cookie Collector
Let's make this a bit more concrete. Let's say we have a cookie-collecting robot in a world that has cookies laying around all over the place, like in figure **FIGNR**. The robot can move in four directions (up, down, left, right). She always knows its position in the world, and uses that information to determine where to go next. When she finds a cookie, she gets happy, but she's always looking for more. Her battery lasts two minutes, after which somebody will recharge her and let her play the game again from the start.

The components of the cookie collecting robot and her environment fit nicely in the RL framework. In table **TABLENR** you can see what form each RL concept takes in this example. Because the example fits in the RL framework, it means that we can apply RL methods to make the robot as succesful as possible, in this case: to make it collect as many cookies as possible withing two minutes.


| **RL concept**  | **Cookie Collector**                                      |
|-----------------|-----------------------------------------------------------|
| Agent           | Cookie collector                                          |
| States          | Observations: position (e.g. x and y coordinate)                  |
| Actions         | Possible movements: up, down, left, right                 |
| Reward          | Zero normally, 'Unit of happiness' (e.g. 1) when she moves to a state with a cookie |
| Policy          | The process of deciding on an action to take |
| Time step       | Amount of time it takes to move to a next state, e.g. 1 second |
| Episode         | Period of 2 minutes                                       |


## The best policy
As mentioned before, the agent is interested in getting as much value out of the episode as possible. In formal terms, we want to find the policy which maximizes the expected total reward in the episode. When the agent is in a certain state, she wants to choose the action of which she expects that it will deliver her the most value. This does not only depend on the immediate reward. The agent should also look ahead and choose actions that brings the agent closer to states with high rewards (e.g. closer to a cookie).

To make this decision, we want to have a state-action value function $Q(s,a)$. This function tells us how much value we can expect when we take action $a$ in state $s$. When she has this function, the agent can simply observe her state $s$, and then choose the $a$ which has highest $Q(s,a)$. What remains is the challenge of finding this state-action value function.

## DQN and the Deadly Triad
Methods to find the state-action value function are called Q-learning methods, and they work in an iterative way. First, an initial setting of the Q-values of each state-action pair is chosen. If this is used as basis for a policy, it performs very bad. In Q-learning, we iteratively update the Q-values (or parameters of our model that gives us Q-values) to make the policy better and better. If the method works, the rewards should converge to a high value if we iterate long enough.

For this blog post, we researched a popular Q-learing method, which is called Deep Q-Learning Network (DQN). It combines three common elements to perform well: function approximation, bootstrapping and off-policy training. We will explain these elements later in more detail. For now, all you need to know that the combination of these three elements is dangerous. Using one or two is done often and works well. However, when all three are used there is a large risk that the returns will *diverge* when we iterate. This means that we end up with a useless Q-function. Because of this dangerous situation, the combination of these three elements is known as the *deadly triad*.

In order to make their method work, the inventors of DQN added some tricks to their method\[**PAPER REFERENCE**\]. They claim that when we use these tricks, we don't have to worry about the deadly triad anymore. However, in the field of RL, attempts to reproduce research often lead to very different results. We wanted to see if the claims of DQN are true, and anwer the question:

> To what extend do the DQN tricks help to solve the deadly triad problem?


# Set-up of our Experiment
<!--
   Algorithms and Techniques
   1. Formal description
   2. Explanation and intuition
-->
<!--
   Experimental Design
   1. Comparison to baseline, clear relevance of baseline and environments
   2. Multiple runs and/or environments
   3. Clear how hyperparameters are chosen and which are used
-->

Intuitive: Q-learning -> Deep Q-learning? -> +tricks=DQN

Formal: line by line in algorithm, what will we influence?

~~~
( 1) Initialize replay memory $D$ to capacity $N$
( 2) Initialize action-value function $Q$ with random weights $\theta$
( 3) Initialize target action-value function $\hat{Q}$ with weights $\theta_{target} = \theta$
( 4) For episode = $1$ to $M$ do
( 5)     Initialize sequence $s_1=\{x_1\}$ g and preprocessed sequence $\phi_1=\phi(s_1)$
( 6)     For $t$ = $1$ to $T$ do
( 7)         With probability $\epsilon$ select a random action $a_t$
( 8)         otherwise select $a_t=\argmax_a Q(\phi(s_t),a;\phi)$
( 9)         Execute action $a_t$ in emulator and observe reward $r_t$ and state $x_{t+1}$
(10)         Clip reward $r_t$
(11)         Set $s_{t+1}=s_t,a_t,x_{t+1}$ and preprocess $\phi_{t+1}=\phi(s_{t+1})$
(12)         Store transition ($\phi_t,a_t,r_r,\phi_{t+1}$) in $D$
(13)         Sample random minibatch of transitions ($\phi_t,a_t,r_r,\phi_{t+1}$) from $D$
(14)         Set $y_j=r_j + \delta_{j+1} \gamma \max_{a'}\hat{Q}(\phi_{j+1},a';\theta_{target})$
(15)         Perform a gradient descent step on $(y_j-Q(\phi_j,a_j;\theta))^2$ with respect to the
(16)         network parameters $$\theta$
(17)         Every $C$ steps reset $\hat{Q}=Q$
(18)     End For
(19) End For
~~~
