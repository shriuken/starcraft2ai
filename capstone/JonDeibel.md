## Fall Essay

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

## Spring Essay
In the first semester I spent a lot of time learning about reinforcement learning and
relaying it to my team. I then found and sent out resources for my group mates to read and
learn. I spent a bit of time working with the group in order to create an initial plan of attack. From
there I built the first Q-learning prototype.
In the second semester I refactored the code to be a little more modular and easier to try
out different ideas. I spent time researching pytorch and relaying my findings so that we could
use it as a tool to build our model. I then used pytorch to convert our basic Q-learning model
into a Deep Q-learning model that was able to take a multidimensional input. After that I
implemented an actor-critic algorithm for learning.
