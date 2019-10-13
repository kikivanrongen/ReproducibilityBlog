---
layout: post
title:  "Experimental Design"
---

We have outlined earlier that the divergence of parameters for semi-gradient methods does not always occur under the assumptions of the *deadly triad*. DQN is an example of such a method and for that reason we limit our experiments to this particular algorithm. More specifically, we attempt to grasp the importance of the model's assumptions as we believe these are essential for convergence. This includes experience replay, periodic iterative updates (similar to fixing the target for a previously defined number of steps) and reward clipping. We narrow our search down to four environments (**EXPLAIN: WHY THESE FOUR?**):

1. Mountain Car
2. Bipedal Walker
3. Acrobot
4. CartPole

Every one of these environments will be tested with and without the three assumptions. In the first case, where we consider experience replay, this means that we will update the policy in an "online" fashion. Particularly, we use S, A, S+1 and A+1 at the time of update. The second property of a fixed target turns into in a learned policy. However, the loss is not propagated through the target so we still obtain a semi-gradient algorithm. Finally, we have two versions of the reward clipping that we apply: one with the original rewards and one with magnified rewards. While the first version is fairly reasonable, the second might look a bit odd. The intuition behind this is that using magnified rewards hopefully leads to a more extreme result, making the role of reward clipping more obvious for DQN.

By means of the above experiments we expect our algorithm to perform worse without any of the assumptions. This indicates that the "tricks", as we may call them, are indeed necessary for proper training. We try to disprove this hypothesis by constantly disabling one of the tricks. If the results show that this does not significantly deteriorates performance, we have a contradiction and our hypothesis is incorrect.  

Naturally, we also have to tune the hyperparameters. Fortunately, the original DQN paper states that the same hyperparameter settings work for a diverse range of tasks. For that reason, we do not necessarily need to tune them for every environment separately. First, we concentrate on the network architecture by testing 3 different sets and monitoring the results. These sets are consecutively [64, 64], [256, 256] and [128, 64, 32]. You can see that we experiment with depth as well as the number of layers, which are the two basic components of the network. Next, the batch size is fixed at 32. In general, this is considered a reasonable amount and has shown to perform well on various problems. The target updates are done every 50 episodes, during a total of 100 episodes per experiment. Lastly, the learning rate is set at 0.001. (**EXPLAIN: test with higher batch sizes/learning rate/number of episodes/etc?**)
<!-- Larger batch sizes have a risk of deteriorating the model's performance, as it can no longer generalize as well as before.  -->

During training we keep track of various quantities (most of them conventional), one of which is the cumulative reward at each time step. As reinforcement learning algorithms require an objective, we have chosen this quantity for optimization. Further statistical tests with regard to this measurement are performed afterwards.

At last, we need to regard multiple random runs. Sadly we only have limited time, so we restrict the number of runs to 10.  
