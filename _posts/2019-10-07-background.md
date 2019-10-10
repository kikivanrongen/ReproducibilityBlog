---
layout: post
title:  "Background"
---

Assignment:
{% highlight ruby %}
Semi-gradient methods do not always converge. DQN [5] uses a semi- gradient version of Q-learning. Can you find an environment where this method diverges? How much do the tricks in DQN (e.g. experience replay, target network) help to avoid divergence?
{% endhighlight %}

Bootstrapping is a popular technique in reinforcement learning to reduce variance. However, it can lead to a biased estimate of the value function and cause unstable training. Moreover, bootstrapping is not considered a true gradient descent method, but actually falls under a different category called semi-gradient methods. DQN is such a method and has been quite popular since it was introduced in 2015. While these methods are very promising (they allow for continuous, online and faster learning), they have one drawback in particular: the risk of divergence. According to Sutton & Barto this should happen under three assumptions called the *deadly triad*:

1. Function approximation
2. Bootstrapping
3. Off-policy training

The interesting part is that DQN does actually perform quite well on the majority of Atari games, although it does contain all three of the above mentioned assumptions. This indicates that there is at least a partial gap in our understanding of the deadly triad and its influence on training convergence. Therefore we will try to fill this gap by performing DQN on various environments, in order to distinguish the key components for its success. 
