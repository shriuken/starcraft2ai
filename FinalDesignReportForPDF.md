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

As our project is run from the command line, there is no user interface.

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
See attached presentation slide deck pdf.

## Expo Poster
See attached poster PDF.

## Self Assessment Essays

Each member wrote their own assessment essays.

### Ryan Benner
#### Fall Essay
Our project is about developing an AI bot to learn and play the real-time strategy game, StarCraft II. I think this is an academically
interesting problem due to the real-time nature of Starcraft II and how this varies compared to previous work that's been done in the
field, such as AlphaGo and and OpenAI's Dota2 bot (which has severe limitations). Compared to AlphaGo, Starcraft II requires more long
term memory, since the game is both real-time and board state isn't as nearly as transparent as it is in AlphaGo. Thus, there's an
interesting perfomant problem and memory problem, which AI are known to be bad at long-term strategy. The most interesting academic part
of our project to me will be teaching and watching how the AI learns. Previous research done by DeepMind has indicated that there are
issues with properly training the AI to explore and then exploit what it discovers. The basic idea of "economy" in Starcraft II is to
mine minerals and vespene gas (in-game currencies) and then spend these resources on building units and buildings - but how you spend is
just as important as when you spend it (which is dependent, somewhat upon long term strategy and what's already happened).

I think some courses in my curriculum will guide development on my senior project. While UC does offer classes on AI, ML, and Data
analysis, I haven't yet taken these courses. However, I am presently taking Learning Probabilistic Models and Natural Language
Processing which I believe will help with this project. Probabilistic Models covers much of the fundamental math and some algorithms
that are popular in the AI/ML field, thus providing a possible base to work from when developing the training algorithms for the bot.
Natural Language Processing, while not directly related to the project, involves another group project where we are again encourage to
use ideas from the AI/ML sphere and apply them to solving a Natural Language Processing problem. This will again afford me a base from
which to work from in developing the AI. Other courses, such as Cyber Defense Overview, Data Encoding, and Cyber Security Vulnerability
Assessment are all classes which have actively encouraged and taught how to explore and investigate unknown entities.

Interning in Japan at Fusion Systems Group will also help with developing our AI. While I was at Fusion, we frequently pivoted what 
methodology we were using to coordinate app development on the team. As such, I have a a good breadth of experience with a few different
methodologies for task planning and management to help stay on top of the project. I've also done undergraduate research in the Trust
Lab, working under Dr. Rozier. This has provided me valuable experience with creating task lists and action items to help accomlish a
goal. For example, Dr. Rozier would frequently only state the problem statement we were working with him to solve, then leave us largely
unsupervised to come up with our own solutions and ideas to solve the problem.

Blizzard and DeepMind have recently worked together to develop and release both a Starcraft II api, to provide hooks into creating a
game AI but also have released a wrapper in Python to call this API. Other community tools have also been created as well, facilitating
development of AI and reducing the amount of reinventing-the-wheel we will have to do. Our primary goal and approach will be to develop
an AI, using Reinforcement Learning, that can play SC2 and defeat, at the least, the easiest AI Blizzard launched with the Starcraft II
game. The AI provided by Blizzard follows a more recipe-defined system for how it should play against players, so I hope to develop an
AI using a more organic approach. To facilitate training, Blizzard has also released gigabytes worth of training data that we can use as
a base to train our AI on, to more quickly train the AI how to play. As mentioned in the first paragraph, I'm most excited about the 
exploration and exploitation portion of training our AI.

The expected results of our project will be to have an AI that can play, at some level, a game of Starcraft II from start to end. As
time goes on and we begin to explore development, we might need to change our goal through any of the following ways:
  - pick a specific race to train on
  - pick a specific matchup (bot's race and bot's opponent race) to train on.
  - focus on a subset of SC2 - for example, the various "mini games" within SC2
  - develop an army mico bot, to assist players in the "Archon" (2-player) mode in SC2.
Since our project has a very tangible output, we can easily measure basic success against whether or not the bot can play (and win) a
game of SC2. Afterwards, we can break this success down into more fine grained parameters such as,
  - win rate of Race X against Blizzard's AI on diffulcty level Y playing Race Z
  - spectate the bot playing against itself. 
  - spectate the bot playing against ranked players of various races to view its performance.
  
#### Spring Essay
I had individual contributions to this project, both technical and non-technical. Initially, I explored the feasibility of developing 
and building our A.I. within the C++ API provided us by Blizzard while Jon explored Python. During my exploration, I actually created 
our first non-machine learned AI that we initially used as a baseline for determining what basic parameters and limitations we wanted 
to use for our AI when we developed the machine learned version.  After our exploration was finished, we decided to utilize the Python 
wrapper provided by DeepMind.  I was also responsible for writing many of the documents for our project, particularly the final design 
poster.  The initial two versions of our poster draft were designed by me, and Jon and I collaborated to create the center most 
graphics depicting the model of our AI. Since I plan on attending graduate school, learning the process for creating a poster was 
extremely valuable to me. 

This project was also my first foray into developing a project that utilized machine learning. After we decided on doing Python, I 
spent much of my time trying to understand how to develop a reinforcement-learned AI and I became quite familiar with a Q-learned model.
Arguably, my biggest obstacle for this project was keeping up with each new update to our AI’s model. Since this was my first time 
working on a project in the ML space, I spent a large portion of my time attempting to learn about the Q-learned model we initially 
started with. I successfully was able to understand this, but due to our project’s model constantly being updated, I did quickly fall 
behind on that front.



### Kyle Arens
#### Fall Essay
The goal of this project is to create an AI to learn and play the game Starcraft 2. Although AIs have been created
previously, there have been consistent flaws in them that prevent them from being considered truly successful.
It has been well established that machines are more successful at games in which the markovian property is prevalent.
They do not display the same success when the outcome of a game is more dependent on long term strategy, and the
markovian property becomes less influential. This presents the biggest challenge of our project, creating an AI 
that can utilize long term memory to make meaningful decisions. Rather than simply abusing its ability to make an 
inhuman number of actions, we want the AI we create to have a broader view of the gameplay than the immediate.

A few courses stand out from the rest in my curriculum here at UC as good preparation for this project. The courses
that stand out as most important to me are CS 4071 (Design and Analysis of Algorithms), CS 6037 (Machine Learning),
STAT 2037 (Probability and Statistics 1), and MATH 2076 (Linear Algebra). Between them, these classes have given me
a solid basis in understanding the math behind machine learning, which is arguably one of the most important things
for succeeding with the creation of an AI. To complement this, CS6037, along with some supplement from other courses
I'm taking this semester, has given me a solid understanding in the theoretical fundamentals of machine learning.
Although UC's undergraduate curriculum does not offer many opportunities to venture into deep learning, it does offer
many opportunities to build a basis of knowledge from which to launch into deep learning from. It is my goal to use
this knowledge base to help me succeed in my use of and research into deep learning in this project.

The contributions of my co-op experience to this project differ starkly from the contributions that my courses will
provide. While my courses provided me with 'hard' technical skills that are relevant to this project, most of the
technical skills I learned on co-op aren't in the least bit applicable to this project. The most applicable would be
the general programming knowledge that I acquired in C++, Java, and Python. That being said, my co-op experiences have
provided me with many soft skills that I hope to use to help contribute to this project. My most recent co-op, working 
as a DEXcenter developer (Java and Python) at ITI, saw me saddled with a surprising amount of responsibility and freedom
around the problems that I was tasked with. This helped me grow my ability to independently analyze and design solutions
to complex problems. One example would be a bug in our pseudo-random number generation where it wasn't returning unique 
values in a specific use case. Rather than redesign the number generation, potentially creating problems in other areas
of the code, I came up with a way to reformat how our data was being stored that would guarantee unique records. 

My primary motivation behind, and interest in, this project is one of personal learning. Going into industry, knowledge
of deep learning isn't likely to have an enormous impact on my work. However, I would like to explore it out of personal
interest, and as a potential topic for masters work in the future. In addition to this, I've always enjoyed video games,
specifically strategy games. Having played them myself, the ability to teach an AI to understand long term and complex
causality is fascinating to me. Lastly, I'm excited about this project because it poses complex and difficult problems.
I enjoy solving problems, and hopefully doing our best to succeed with this AI will pose many interesting problems.

Our expected result is to create an AI that can, on some functional level, play Starcraft 2. This gives us a very
tangible and measurable output: is our AI winning its games? In addition to that, how soundly is it winning or losing?
Is it putting up a decent contest? To begin approaching this, the simplest way to begin is likely a simple logistic
regression across a set of gameplay data. From there, that model can be refined by moving into a reinforcement based system
with supervision. Ideally, the AI will be able to play a game by itself at the end of this project.

#### Spring Essay
My individual contribution to this project came in the areas of helping debug, model design, and working on the assignments required by senior design. To do this, I worked with python, pytorch, the starcraft 2 API provided jointly by Blizzard and Deepmind, diagramming software, and presentation and word processing software. This built on my skills with python, and helped to establish my skills and knowledge in the area of reinforcement learning.

The main thing that I wanted to learn about going into this project was reinforcement learning, which was mostly successful. I got to work with pytorch, a python library designed specifically for reinforcement learning, and got to look into current research and ideas in the field of reinforcement learning in order to try and find ways to further our model. While I learned a lot in the theoretical realm, I didn’t get to apply it as much as I would have liked. This due to the scope and time limitations on our project, and as well as hardware limitations. In the course of learning about our project, one major obstacle we had to overcome was adjusting our expectations for the project. We were shooting for a goal that was far too high, and after learning more, decided we should limit our scope to create a more manageable goal. Within this adjusted scope, we had a lot of success, which is described below in part B.

### Jon Deibel
#### Fall Essay

Our senior design project is centered around building an AI to play the game
Starcraft 2. This poses a lot of challenges to not just our team, but to
the AI world as a whole. The seemingly unsolved problem is dealing with long term
memory and how to learn that a small decision made early on in the process has
a very large impact on later outcomes. Currently it has been shown that AI can play
simple games that exhibit the markovian property better than humans. It has also
shown a clear defficiency in games that require any sort of long term strategy.
So our project will largely be trying to figure out some method in which we can
capture long term planning and strategy and incorporate that into an AI.

My college curriculum has prepped me in a lot of ways. The most obvious way is
that it has prepped me to be able to understand algorithms (CS 4071). While
algorithms is a central part of Machine Learning, I need a good understanding
of the math around it. I satisfied most of that need through the slew of math
courses (Calc 1 & 2, Prob and Stats, and Linear Algebra) I took. I need those
skills and knowledge to have a deeper understanding of the Machine Learning
research and the models that were built from complex systems. I still have a lot
of domain specific knowledge to obtain, but I believe that I have a solid
foundation to venture into those areas.

While my college curriculum will help in understanding a lot of the mechanics,
my Co-op with Intuit has provided me a lot of learnings around necessary soft
skills. The most important one is how to effectively work in a team. At Intuit
I spent my entire time working, learning, and planning as a team. I feel that
skill will directly transfer in order to help my and my team succeed to some
measure. Another critical skill is planning. This is a large project my all
estimates and it is important that there is some structure and planning around
it in order to try and make small steps towards the larger goal.

I am extremely excited to work on this project. I have a keen interest in AI systems
and machine learning in general. I have spent a lot of my personal time working
on tangentially related projects using machine learning systems that I built. This
gives me a good base to start learning the more complex topics like reinforcement
and deep reinforcement learning that will likely be required in our solution. I
think we chose a really hard and interesting problem that has not yet been solved.
I expect us to learn a great deal, but I do not expect us to succeed at our goal.
I believe we will have done well if we can create supervised and reinforcement
learning systems in general. Contributions are going to be much more knowledge
based rather than code based, so evaluation there will be more difficult.

I have a general idea on how to first attempt to tackle this problem. I think a
good and easy starting spot would be to take our large dataset of game replay
data which we acquired from blizzard and to build a simple logistic regression
model on top of it that maps state -> action of the player in each game that is
the winner. If that is able to somehow play a game, which I don't expect it to,
that would be considered quite a success. Going from there, we need to take the
knowledge we gained of reinforcement learning systems and use a supervised model,
similar to the previous model, to give that reinforcement learning's model or Q
function a starting point to which it can then play games against itself to refine
that function. At this point, the hope is that the agent can then play a full
game of starcraft by itself.

#### Spring Essay


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
