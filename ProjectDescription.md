# Starcraft II AI Project Description

## Team Members

Kyle Arens  - arenskyle@gmail.com   
Ryan Benner - bennerrj@outlook.com  
Jon Deibel  - dibesjr@gmail.com  

## Faculty Advisor

Dr. Ali Minai - http://www.ece.uc.edu/~aminai/

## Project Background Description

There has been a lot of exciting publications and advances in the AI world in the past several years. With this project, we want to explore and use conventional and cutting edge ML techniques in order to create some sort of Starcraft II (SC2) agent that can play, and hopefully defeat a basic blizzard AI.

## Problem Statement

The problem we are attempting to address is that nobody has yet been able to create an AI for SC2 that operates on a human APM (Actions per Minute) level.

Current SC/SC2 AI abuses the fact that it is able to take an inhuman number of actions very quickly, thereby making it a poor example of actual AI capabilities.

## Finals goal of project

The scope of this project will be that our AI is able to successfully play a full SC2 game, ideally beating the current SC2 very easy AI, while limited to a human-like APM.

## Background Skills/Interests

Collectively, we have an interest in real-time strategy style games, AI, and learning systems.

## Possible Approaches

There are a lot of ways we could go about doing this project. Below we outlined a few and explained the challenges and benefits to each of them. These are just some that we have been thinking about recently, certainly not an exhaustive list.

### Model to learn Actions -> Game outcome

Given a set of SC2 replay data, we can build a model that given a game state outputs the likelihood of winning. You could do this by mapping a game state to the game outcome and learn from that set of data.

**Challenges**

This method would require a pretty large set of SC2 replay data. Thankfully deepmind and blizzard have provided about 13 gigs (with more coming) of anonymized replay data for us to use.

**Benefits**

The major benefit of this model is that if properly trained, building an AI would be a simple deep search on possible X moves followed by a certain depth of moves after that and computing the overall confidence of winning at each given board state. Boom, an AI. (or something)

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

This has quite a few challenges attached to it. The first one is to learn how in the world to build a reinforcement learning system. Then we have to hook that up to the simulated SC2 environment. Once those are done, then we need the compute power to actually have the damn thing to learn and get better.

**Benefits**

There are a lot of benefits here. The first and foremost is that this has proved to be the most effective way to build an AI agent yet (see AlphaGo). The next is that we don’t need a massive amount of pre-labeled and crafted data. The last benefit is that we would learn some really dope ass cutting edge stuff.

## Conclusion

Regardless of which method sounds promising, the first step would be to investigate other methods. Since these ideas are just off the top of my head, there are undoubtedly many ways to solve this problem. It would probably be good to explore more of them. The final demo would be to show the AI playing against either itself or a Blizzard AI opponent.
