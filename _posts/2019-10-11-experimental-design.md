---
layout: post
title:  "Experimental Design"
---



We have outlined earlier that the divergence of parameters for semi-gradient methods does not always occur under the assumptions of the *deadly triad*. DQN is an example of such a method and for that reason we limit our experiments to this particular algorithm. More specifically, we attempt to grasp the importance of the model's assumptions as we believe these are essential for convergence. This includes experience replay, periodic iterative updates (similar to fixing the target for a previously defined number of steps) and reward clipping. We narrow our search down to three environments:

1. Mountain Car
2. Acrobot
3. CartPole

We investigate five experimental settings:

1. **Normal DQN (DQN)**: this is the normal model with experience replay, periodic iterative updates and reward clipping.
2. **No experience replay (-ER)**: the policy is updated in an "online" fashion. Particularly, we use S, A, S+1 and A+1 at the time of update.
3. **No periodic updates of target network (-TN)**: the target is defined by the current network, instead of a target network that is periodically updated.
4. **No reward clipping (-RC)**: the real rewards of the environment are used.
5. **Reward gain (R+)**: the rewards are not clipped, but are miagnified by a factor 100 instead. The intuition behind this is that using magnified rewards hopefully leads to a more extreme result, making the role of reward clipping more obvious for DQN.

In the experiments, we expect our algorithm to perform worse when we leave components out. This indicates that the "tricks", as we may call them, are indeed necessary for proper training. We try to disprove this hypothesis by constantly disabling one of the tricks. If the results show that this does not significantly deteriorates performance, we have a contradiction and our hypothesis is incorrect.  

Naturally, we also have to tune the hyperparameters. Fortunately, the original DQN paper states that the same hyperparameter settings work for a diverse range of tasks. For that reason, we do not necessarily need to tune them for every environment separately. We tune only the learning rate, for which we try 10 different values. In each experimental setting, we report the results of the optimal learning rate for that setting. The batch size is fixed at 32, which is a typical value that works well under normal conditions. The target network is updated every **C** iterations. Every experiment is run with 10 different seeds, to measure the significance of our findings.

Our main question is whether the DQN network converges to a solution under the different experimental settings. To measure convergence, we compute the parameter gradient norm in each iteration. The minimal norm (MN) is an indication of convergence. If it is close to zero, it means that the policy hardly changed at some point during training. To have intuition on the variance of this norm over seeds, we report its average, standard deviation, and its minimum and maximum over seeds.

Because convergence does not mean that a globally optimal solution is found, we also report various quantities related to the specific environments, one of which is the cumulative reward at each time step.

Our results will be summarized in a table like below for every environment:

**Results for Environment ...**

|      | avg MN | std MN | min MN | max MN | Task specific metrics |
|------|--------|--------|--------|--------|-----------------------|
|  DQN |        |        |        |        |                       |
| -ER  |        |        |        |        |                       |
| -TN  |        |        |        |        |                       |
| -RC  |        |        |        |        |                       |
| +R   |        |        |        |        |                       |
