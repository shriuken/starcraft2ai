# Starcraft 2 AI

## Table of Contents

1. [Readme](https://github.com/shriuken/starcraft2ai/blob/master/README.md)
1. [Project Description](https://github.com/shriuken/starcraft2ai/blob/master/FinalDesignReport.md#project-description)
1. [Test Plan and Results](https://github.com/shriuken/starcraft2ai/blob/master/FinalDesignReport.md#test-plan-and-results)
1. [User Manual](https://github.com/shriuken/starcraft2ai/blob/master/FinalDesignReport.md#user-manual)
    1. [Manual](https://github.com/shriuken/starcraft2ai/blob/master/FinalDesignReport.md#manual)
    1. [FAQ](https://github.com/shriuken/starcraft2ai/blob/master/FinalDesignReport.md#faq)
1. [Final PPT Presentation](https://github.com/shriuken/starcraft2ai/blob/master/FinalDesignReport.md#final-ppt-presentation)
1. [Final Expo Poster](https://github.com/shriuken/starcraft2ai/blob/master/FinalDesignReport.md#expo-poster)
1. [Self-Assessment Essays](https://github.com/shriuken/starcraft2ai/blob/master/FinalDesignReport.md#self-assessment-essays)
    1. [Initial Assessments](https://github.com/shriuken/starcraft2ai/blob/master/FinalDesignReport.md#fall-essays)
    1. [Final Assessments](https://github.com/shriuken/starcraft2ai/blob/master/FinalDesignReport.md#spring-essays)
1. [Summary of Hours and Justification](https://github.com/shriuken/starcraft2ai/blob/master/FinalDesignReport.md#summary-of-hours-and-justification)
1. [Budget](https://github.com/shriuken/starcraft2ai/blob/master/FinalDesignReport.md#budget)

## Project Description

There have been many recent advances in game-playing AIs, such as the Dota2 AI and AlphaGo. With this project, we explored the use of conventional and cutting edge ML techniques to create a self-learning Starcraft II (SC2) AI agent capable of defeating Blizzard's Very Easy AI.

We are scoping our project in the following manner:
  * the agent will play as [Terran](http://us.battle.net/sc2/en/game/race/terran/) 
  * against a Very Easy Terran AI provided in the retail version of the game
  * on the maps Simple64 and flat64.
  
## Interface Specification

## Test Plan and Results

We have two overall approaches to to testing our Starcraft 2 AI - code coverage tests and full-system tests. Due to the nature of our project, we aren’t fully able to test the learning algorithms we write, thus we will test small parts of the learning code, to ensure proper aspects are (or are not) changing. Much of our full-system tests will be verifying and inspecting the result of our learning algorithms, and spectating the bot gameplay and win conditions.

### Test Case Descriptions

| Indicator | Description |
| --------- | ----------- |
| FST 1.1 | Full-System Test 1 |
| FST 1.2 | To ensure that we have a bot capable of handling different skill levels and styles of Blizzard A.I. bots. |
| FST 1.3 | This test will have have our bot face off against a Blizzard-supplied Very Easy Terran A.I. bot. |
| FST 1.4 | Inputs: Map, Trained A.I. Bot, Blizzard A.I Bot, Trained A.I. Race, Blizzard A.I. Race. |
| FST 1.5 | Output: Trained A.I. victory |
| FST 1.6 | Normal |
| FST 1.7 | Blackbox |
| FST 1.8 | Performance Test Indication |
| FST 1.9 | Integration |
| RESULT | Pass - 80%, Fail - 20% |
| | |
| FST 2.1 | Full-System Test 2 |
| FST 2.2 | To ensure that we have a bot capable of handling different skill levels and styles of Blizzard A.I. bots. |
| FST 2.3 | This test will have have our bot face off against a Blizzard-supplied Easy Terran A.I. bot. |
| FST 2.4 | Inputs: Map, Trained A.I. Bot, Blizzard A.I Bot, Trained A.I. Race, Blizzard A.I. Race. |
| FST 2.5 | Output: Trained A.I. victory |
| FST 2.6 | Normal |
| FST 2.7 | Blackbox |
| FST 2.8 | Performance Test Indication |
| FST 2.9 | Integration |
| RESULT | Pass - 20%, Fail - 80% |
| | |
| FST 3.1 | Full-System Test 3 |
| FST 3.2 | To ensure that we have a bot capable of handling different skill levels and styles of Blizzard A.I. bots. |
| FST 3.3 | This test will have have our bot face off against a Blizzard-supplied Medium Terran A.I. bot. |
| FST 3.4 | Inputs: Map, Trained A.I. Bot, Blizzard A.I Bot, Trained A.I. Race, Blizzard A.I. Race. |
| FST 3.5 | Output: Trained A.I. victory |
| FST 3.6 | Normal |
| FST 3.7 | Blackbox |
| FST 3.8 | Performance Test Indication |
| FST 3.9 | Integration |
| RESULT | Pass - 1%, Fail - 99% |
| | |
| FST 4.1 | Full-System Test 4 |
| FST 4.2 | To ensure that we have a bot capable of handling different skill levels and styles of Blizzard A.I. bots. |
| FST 4.3 | This test will have have our bot face off against a Blizzard-supplied Very Easy Protoss A.I. bot. |
| FST 4.4 | Inputs: Map, Trained A.I. Bot, Blizzard A.I Bot, Trained A.I. Race, Blizzard A.I. Race. |
| FST 4.5 | Output: Trained A.I. victory |
| FST 4.6 | Normal |
| FST 4.7 | Blackbox |
| FST 4.8 | Performance Test Indication |
| FST 4.9 | Integration |
| RESULT | Pass - 80%, Fail - 20% |
| | |
| FST 5.1 | Full-System Test 5 |
| FST 5.2 | To ensure that we have a bot capable of handling different skill levels and styles of Blizzard A.I. bots. |
| FST 5.3 | This test will have have our bot face off against a Blizzard-supplied Easy Protoss A.I. bot. |
| FST 5.4 | Inputs: Map, Trained A.I. Bot, Blizzard A.I Bot, Trained A.I. Race, Blizzard A.I. Race. |
| FST 5.5 | Output: Trained A.I. victory |
| FST 5.6 | Normal |
| FST 5.7 | Blackbox |
| FST 5.8 | Performance Test Indication |
| FST 5.9 | Integration |
| RESULT | Pass - 12%, Fail - 88% |
| | |
| FST 6.1 | Full-System Test 6 |
| FST 6.2 | To ensure that we have a bot capable of handling different skill levels and styles of Blizzard A.I. bots. |
| FST 6.3 | This test will have have our bot face off against a Blizzard-supplied Medium Protoss A.I. bot. |
| FST 6.4 | Inputs: Map, Trained A.I. Bot, Blizzard A.I Bot, Trained A.I. Race, Blizzard A.I. Race. |
| FST 6.5 | Output: Trained A.I. victory |
| FST 6.6 | Normal |
| FST 6.7 | Blackbox |
| FST 6.8 | Performance Test Indication |
| FST 6.9 | Integration |
| RESULT | Pass - 0%, Fail - 100% |
| | |
| FST 7.1 | Full-System Test 7 |
| FST 7.2 | To ensure that we have a bot capable of handling different skill levels and styles of Blizzard A.I. bots. |
| FST 7.3 | This test will have have our bot face off against a Blizzard-supplied Very Easy Zerg A.I. bot. |
| FST 7.4 | Inputs: Map, Trained A.I. Bot, Blizzard A.I Bot, Trained A.I. Race, Blizzard A.I. Race. |
| FST 7.5 | Output: Trained A.I. victory |
| FST 7.6 | Normal |
| FST 7.7 | Blackbox |
| FST 7.8 | Performance Test Indication |
| FST 7.9 | Integration |
| RESULT | Pass - 80%, Fail - 20% |
| | |
| FST 8.1 | Full-System Test 8 |
| FST 8.2 | To ensure that we have a bot capable of handling different skill levels and styles of Blizzard A.I. bots. |
| FST 8.3 | This test will have have our bot face off against a Blizzard-supplied Easy Zerg A.I. bot. |
| FST 8.4 | Inputs: Map, Trained A.I. Bot, Blizzard A.I Bot, Trained A.I. Race, Blizzard A.I. Race. |
| FST 8.5 | Output: Trained A.I. victory |
| FST 8.6 | Normal |
| FST 8.7 | Blackbox |
| FST 8.8 | Performance Test Indication |
| FST 8.9 | Integration |
| RESULT | Pass - 9%, Fail - 91% |
| | |
| FST 9.1 | Full-System Test 9 |
| FST 9.2 | To ensure that we have a bot capable of handling different skill levels and styles of Blizzard A.I. bots. |
| FST 9.3 | This test will have have our bot face off against a Blizzard-supplied Medium Zerg A.I. bot. |
| FST 9.4 | Inputs: Map, Trained A.I. Bot, Blizzard A.I Bot, Trained A.I. Race, Blizzard A.I. Race. |
| FST 9.5 | Output: Trained A.I. victory |
| FST 9.6 | Normal |
| FST 9.7 | Blackbox |
| FST 9.8 | Performance Test Indication |
| FST 9.9 | Integration |
| RESULT | Pass - 0%, Fail - 100% |
| | |

## User Manual

### Setup
starcraft2ai is dependence upon the following libraries:
* [PySC2](https://github.com/deepmind/pysc2#quick-start-guide)
* [PyTorch](http://pytorch.org/)

### Installation
For detailed information regarding installation of starcraft2ai dependencies, please see the [Quick Start Guide](https://github.com/deepmind/pysc2#quick-start-guide) made available by PySC2 and the [PyTorch](http://pytorch.org/) documents. The PySC2 quick start guide includes:
* Directions for installying PySC2 via pip or Git
* Links to Blizzard's [documentation](https://github.com/Blizzard/s2client-proto#downloads) for downloading SC2 on Linux. (Starcraft2 is natively available for Windows and Mac from [Battle.net](https://us.battle.net/account/download/).
* [Map pack data](https://github.com/deepmind/pysc2#get-the-maps)
* Basic instructions for running and testing successful installation of PySC2.

1. Install PySC2
  * Run `$ pip install pysc2`
2. Install Starcraft2
  * Linux
    * For Linux, follow Blizzard's [documentation](https://github.com/Blizzard/s2client-proto#downloads). PySC2 expects StarCraftII to exist in `~/StarCraftII/` directory. Download the latest version of StarCraft2, then unzip to `~/StarCraftII/`.
  * Windows / Mac
    * Download and install the [StarCraft II](https://us.battle.net/account/download/) installer. 
    * Use default directories for installation purposes.
3. Install PyTorch.
  * `$ pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl 
pip3 install torchvision`
  * If you want to customize your installation, please visit [PyTorch](http://pytorch.org/).
  
### Running starcraft2ai
The starcraft2ai project is executed from the command line, or can be run from an IDE. If you wish to run it out of the box from our repository, there are a few major flags that should be set, and steps that should be taken.

1. First off, if you wish to run using a model that we have already begun, ensure that the flags at the top of QAgent.py are set as follows below. If you wish to use a new model, set 'resume' to FALSE.
  * DATA_FILE = 'sparse_agent_data'
  * USE_CUDA = True
  * IS_RANDOM = False
  * resume = True
  * resume_best = False
  * evaluate = False
2. After ensuring these flags are set correctly, navigate to the src folder within your starcraft2ai directory.
3. From there, run the following command:
  * python3 agent.py --map Simple64 --agent QAgent.RLAgent --agent_race T --max_agent_steps 0 --game_steps_per_episode 25000 --difficulty 1
4. This will launch the game in a new window, and you will be able to follow the progress of the model in both that new window, and the command line.
  

### FAQ
* Q: Do I need to buy SC2?
* A: No! The starter edition can be downloaded and used for free.
* Q: Can I play against the AI?
* A: Not yet! Maybe one day. Maybe you are the AI.


## Final PPT Presentation
See attached powerpoint.

## Expo Poster
See attached poster PDF.

## Self Assessment Essays

Each member wrote their own assessment essays.

### Fall Essays

* [Arens' Essay](https://github.com/shriuken/starcraft2ai/blob/master/capstone/KyleArens.md#fall-essay)
* [Benner's Essay](https://github.com/shriuken/starcraft2ai/blob/master/capstone/RyanBenner.md#fall-essay)
* [Deibel's Essay](https://github.com/shriuken/starcraft2ai/blob/master/capstone/JonDeibel.md#fall-essay)

### Spring Essays

* [Arens' Essay](https://github.com/shriuken/starcraft2ai/blob/master/capstone/KyleArens.md#spring-essay)
* [Benner's Essay](https://github.com/shriuken/starcraft2ai/blob/master/capstone/RyanBenner.md#spring-essay)
* [Deibel's Essay](https://github.com/shriuken/starcraft2ai/blob/master/capstone/JonDeibel.md#spring-essay)

## Summary of Hours and Justification

| Work | Time | Team Members |
| ---- | ---- | ------------ |
| **Fall Semester** |
| Learning the basics of Reinforcement Learning | 25 Hours | Kyle Arens, Jon Deibel, Ryan Benner |
| Understanding C++/Python API | 10 Hours | Jon Deibel, Ryan Benner, Kyle Arens |
| Creation of baseline AI/First RL based AI | 10 Hours | Jon Deibel, Ryan Benner |
| Documentation | 10 Hours | Ryan Benner, Kyle Arens, Jon Deibel |
| Reading + Understanding AlphaGo | 5 Hours | Ryan Benner, Kyle Arens, Jon Deibel |
| | | |
| **Spring Semester** |
| Model Improvements to AI | 50 Hours | Jon Deibel |
| Running Test Cases/Debugging | 25 Hours | Kyle Arens, Ryan Benner, Jon Deibel |
| Senior Design Expo | 4 Hours | Kyle Arens, Ryan Benner, Jon Deibel |
| Senior Design Poster | 25 Hours | Ryan Benner, Jon Deibel |
| Final Design Report | 4 Hours | Ryan Benner, Kyle Arens |
| Self Assessments | 2 Hours | Kyle Arens, Ryan Benner, Jon Deibel |
| Test Plans | 6 Hours | Kyle Arens, Ryan Benner, Jon Deibel |
| User Docs & FAQ | 10 Hours | Kyle Arens, Ryan Benner, Jon Deibel |

## Budget

There have been no expenses to date.
