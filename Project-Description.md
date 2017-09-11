Submit on BB a link to your project repository.

At a minimum your project description should include the following 8 items:

project background description

project problem statement

inadequacy of current solutions to problem

background skills/interests applicable to problem

your project team approach to problem, including the overall goals and your team expectations for final product/demo.



# Starcraft II AI Project Description

## Team Members

Kyle Arens  - arenskyle@gmail.com  
Ryan Benner - bennerrj@outlook.com  
Jon Deibel  - dibesjr@gmail.com  

# Faculty Advisor

In the process of setting up a meeting with Dr. Ali Minai.http://www.ece.uc.edu/~aminai/

## Abstract

There has been a lot of exciting publications and advances in the AI world in the past several years. With this project, we want to explore and use conventional and cutting edge ML techniques in order to create some sort of Starcraft II (SC2) agent that can play, and hopefully defeat a basic blizzard AI.

## Scope

The scope of this project will ideally hit the bare minimum of being able to successfully play a full SC2 game, ideally beating the current SC2 AI.

## Details

There are a lot of ways we could go about doing this project. Below we outlined a few and explained the challenges and benefits to each of them. These are just some that we have been thinking about recently, certainly not an exhaustive list.

## Ideas

### Model to learn Actions -> Game outcome

Given a set of SC2 replay data, we can build a model that given a game state outputs the likelihood of winning. You could do this by mapping a game state to the game outcome and learn from that set of data.

**Challenges**

This method would require a pretty large set of SC2 replay data. Thankfully deepmind and blizzard have provided about 13 gigs (with more coming) of anonymized replay data for us to use.

**Benefits**

1. The major benefit of this model is that if properly trained, building an AI would be a simple deep search on possible X moves followed by a certain depth of moves after that and computing the overall confidence of winning at each given board state. Boom, an AI. (or something)

---

### Algorithm to determine Actions -> Game outcome

A similar approach to #1 but instead of applying a learning model to slap a value to any given game state, we come up with our own algorithm.

**Challenges**

It requires a ton of domain knowledge to figure out a proper algorithm to figure out who is in the lead given a game state. This could lead to some lackluster results, and it wouldn’t really "learn" or get better on its own.

**Benefits**

The benefit here is simple: we don’t need a massive set of data.

---

### Build a Reinforcement Learning system

We go the route of DeepMind’s AlphaGo (NOTE:  https://deepmind.com/research/alphago/) and use Reinforcement Learning to have the AI learn to be better on its own. The way this would work is to have the same AI play against each other in a simulated environment, also known as Adversarial Learning (it's the new hotness).

**Challenges**

This has quite a few challenges attached to it. The first one is to learn how in the world to build a reinforcement learning system. Then we have to hook that up to a simulated heartSC2tone environment. (NOTE:  https://github.com/HeartSC2im/SabberStone) Once those are done, then we need the compute power to actually have the damn thing to learn and get better.

**Benefits**

There are a lot of benefits here. The first and foremost is that this has proved to be the most effective way to build an AI agent yet (see AlphaGo). The next is that we don’t need a massive amount of pre-labeled and crafted data. The last benefit is that we would learn some really dope ass cutting edge stuff.

## Conclusion

Regardless of which method sounds promising, the first step would be to investigate other methods. Since these ideas are just off the top of my head, there are undoubtedly many ways to solve this problem. It would probably be good to explore more of them.
