---
layout: post
title:  "Background"
---

Assignment:

> Semi-gradient methods do not always converge. DQN uses a semi- gradient version of Q-learning. Can you find an environment
> where this method diverges? How much do the tricks in DQN (e.g. experience replay, target network) help to avoid divergence?

Bootstrapping is a popular technique in reinforcement learning to reduce variance. However, it can lead to a biased estimate of the value function and cause unstable training. Moreover, bootstrapping is not considered a true gradient descent method, but actually falls under a different category called semi-gradient methods. DQN is such a method and has been quite popular since it was introduced in 2015. While these methods are very promising (they allow for continuous, online and faster learning), they have one drawback in particular: the risk of divergence. According to Sutton & Barto this should happen under three assumptions called the *deadly triad*:

1. Function approximation
2. Bootstrapping
3. Off-policy training

The interesting part is that although DQN does contain all three of the above mentioned assumptions, it still performs quite well on the majority of Atari games. This indicates that there is at least a partial gap in our understanding of the deadly triad and its influence on training convergence. Therefore we will try to fill this gap by performing DQN on multiple environments under various conditions. However, before diving into the experimental part we first take a step back and focus on the core elements of DQN. Only then we can distinguish the key components of its success.

The strength of DQN lies in the clever way it deals with instabilities during training of reinforcement learning problems. It introduces two features: experience replay and periodic iterative update rules. The first feature, experience replay, stores the agent's experiences and subsequently draws random samples from this pool. The weight updates are then no longer based on sequential samples, which breaks the data dependency. Furthermore, since these updates occur periodically, it is also more robust to small alterations that severely change the policy. It is worth mentioning that the inclusion of experience smooths learning by averaging over previous states, and hereby conveniently avoids divergence of parameter estimates. In summary, the combination of both circumvents the correlations that are present in the data, as well as between action-values and target values.

Lastly, as the original DQN research specified a wide range of Atari games (49 to be exact) with all considerably diverse scores, they limited the range of rewards between -1 and 1. This means that all negative rewards are set to -1 and positive rewards are set to 1, leaving zero rewards unchanged.
