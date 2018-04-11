## Fall Essay

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
  
## Spring Essay

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
