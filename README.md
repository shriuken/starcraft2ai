# Starcraft 2 AI
This document is adapted, in part, from the original [project description](https://github.com/shriuken/starcraft2ai/edit/master/ProjectDescription.md) document.

## Table of Contents
1. [Project Abstract](https://github.com/shriuken/starcraft2ai/blob/master/README.md#abstract)
    1. [Team Members](https://github.com/shriuken/starcraft2ai/blob/master/README.md#team-members)
    1. [Advisor](https://github.com/shriuken/starcraft2ai/blob/master/README.md#faculty-advisor)
1. [Project Description](https://github.com/shriuken/starcraft2ai/blob/master/README.md#project-description)
1. [User Stories and Design Diagrams](https://github.com/shriuken/starcraft2ai/blob/master/README.md#user-stories--design-diagrams)
    1. [User Stories](https://github.com/shriuken/starcraft2ai/blob/master/README.md#user-stories)
    1. [Design Diagrams](https://github.com/shriuken/starcraft2ai/blob/master/README.md#design-diagrams)
        1. [Level 0](https://github.com/shriuken/starcraft2ai/blob/master/README.md#level-0)
        1. [Level 1](https://github.com/shriuken/starcraft2ai/blob/master/README.md#level-1)
        1. [Level 2](https://github.com/shriuken/starcraft2ai/blob/master/README.md#level-2)
1. [Project Tasks and Timeline](https://github.com/shriuken/starcraft2ai/blob/master/README.md#project-tasks-and-timelines)
1. [ABET Concerns Essay](https://github.com/shriuken/starcraft2ai/blob/master/README.md#abet-concerns-essays)
1. [Slideshow](https://github.com/shriuken/starcraft2ai/blob/master/README.md#slideshow)
    1. [Slides](https://github.com/shriuken/starcraft2ai/blob/master/README.md#slides)
    1. [Recorded Presentation](https://github.com/shriuken/starcraft2ai/blob/master/README.md#recorded-presentation)
1. [Self-Assessment Essays](https://github.com/shriuken/starcraft2ai/blob/master/README.md#self-assessment-essays)
1. [Professional Biographies](https://github.com/shriuken/starcraft2ai/blob/master/README.md#professional-biographries)
1. [Budget](https://github.com/shriuken/starcraft2ai/blob/master/README.md#budget)
1. [Appendix](https://github.com/shriuken/starcraft2ai/blob/master/README.md#appendix)

## Abstract

There have been many recent advances in game-playing AIs, such as the [Dota2 AI](https://blog.openai.com/dota-2/) and [AlphaGo](https://deepmind.com/research/alphago/). With this project, we aim to explore the use of conventional and cutting edge ML techniques to create a self-learning [Starcraft II (SC2)](https://www.starcraft2.com/en-us/) AI agent that can play against Blizzard AI.

### Team Members

Kyle Arens  - arenskyle@gmail.com   
Ryan Benner - bennerrj@outlook.com  
Jon Deibel  - dibesjr@gmail.com  

### Faculty Advisor

Dr. Ali Minai - http://www.ece.uc.edu/~aminai/

## Project Description

Inspired by recent advances in game-playing AI, we are attempting to address a similarly themed, albeit different problem - creating an SC2 AI agent that operates at human APM (actions per minute), up to ~300apm. Current SC2 AI agents abuse the fact that agents are able to take an inhuman number of actions extraordinarily quickly (such as 5000apm). Our goal is to restrict APM down to ~300apm in order to model our AI off of human capabilities and to more-effectively learn  decisions. With a hard-limit of ~300apm, any given action taken will have more importance than if the bot had a higher limit, such as ~5000apm. 

We are scoping our project in the following manner:
  * the agent will play as [Terran](http://us.battle.net/sc2/en/game/race/terran/) 
  * against a (yet-to-be-determined) stock Blizzard AI controlled target race
  * on a (yet-to-be-determined) 1v1 ladder map.
  
## User Stories & Design Diagrams

### User Stories

[User Stories](https://github.com/shriuken/starcraft2ai/blob/master/UserStories.md)

### Design Diagrams

#### Level 0
![D0 diagram](https://raw.githubusercontent.com/shriuken/starcraft2ai/master/design_diagrams/d0-diagram.png)

This diagram represents the highest level view of our architecture. Starcraft 2 game replay data is taken in, and used to generate a model for autonomously playing the game.

#### Level 1
![D1 diagram](https://raw.githubusercontent.com/shriuken/starcraft2ai/master/design_diagrams/d1-diagram.png)

This diagram elaborates on our AI engine, and gives a high level view of what is being done at each stage. First the game state is parsed, then that parsed data is used to train the model, and turn the model decisions into in-game actions.

#### Level 2
![D2 diagram](https://raw.githubusercontent.com/shriuken/starcraft2ai/master/design_diagrams/d2-diagram.png)

This diagram gives a much more in depth breakdown of what each portion of our AI engine does. It elaborates greatly on how the data is parsed, and how it's used in long and short term planning by our model.

## Project Tasks and Timelines

Combined [Task List, Timeline, and Effort Matrix](https://github.com/shriuken/starcraft2ai/blob/master/TaskList.md).

## ABET Concerns Essays

[Essay](https://github.com/shriuken/starcraft2ai/blob/master/ABETEssay.md).

## Slideshow

### Slides
[Google Presentation](https://docs.google.com/presentation/d/1Hcb6aYpbip0fVUfoEqo3Y5zmzDAUOA4cRRBE9JkhCWU/edit?usp=sharing).

### Recorded Presentation
[Video](https://drive.google.com/file/d/0BwE7gBKwLy1lMGRQa0wtWGstdmM/view).

## [Self Assessment Essays](https://github.com/shriuken/starcraft2ai/tree/master/capstone)

Each member wrote their own assesment essay.
* [Arens' Essay](https://github.com/shriuken/starcraft2ai/blob/master/capstone/KyleArens)
* [Benner's Essay](https://github.com/shriuken/starcraft2ai/blob/master/capstone/RyanBenner)
* [Deibel's Essay](https://github.com/shriuken/starcraft2ai/blob/master/capstone/JonDeibel)

## [Professional Biographries](https://github.com/shriuken/starcraft2ai/tree/master/bios)
* [Arens' Bio](https://github.com/shriuken/starcraft2ai/blob/master/bios/KyleArens.md)
* [Benner's Bio](https://github.com/shriuken/starcraft2ai/blob/master/bios/RyanBenner.md)
* [Deibel's Bio](https://github.com/shriuken/starcraft2ai/blob/master/bios/JonDeibel.md)

## Budget

There have been no expenses to date.

## Appendix

### References
Python abstraction: https://github.com/deepmind/pysc2  
Raw C++ API for starcraft: https://github.com/Blizzard/s2client-proto  
For deep model creation: https://github.com/tensorflow/tensorflow  
Silver, David, et al. "Mastering the game of go without human knowledge." Nature 550.7676 (2017): 354-359.

### Work breakdown

| Work | Time | Team Members |
| ---- | ---- | ------------ |
| Learning the basics of Reinforcement Learning | 25 Hours | Kyle Arens, Jon Deibel, Ryan Benner |
| Understanding C++/Python API | 10 Hours | Jon Deibel, Ryan Benner, Kyle Arens |
| Creation of baseline AI/First RL based AI | 10 Hours | Jon Deibel, Ryan Benner |
| Documentation | 5 Hours | Ryan Benner, Kyle Arens, Jon Deibel |
| Reading + Understanding AlphaGo | 5 Hours | Ryan Benner, Kyle Arens, Jon Deibel |
