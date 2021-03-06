## Fall Essay
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

## Spring Essay
My individual contribution to this project came in the areas of helping debug, model design, and working on the assignments required by senior design. To do this, I worked with python, pytorch, the starcraft 2 API provided jointly by Blizzard and Deepmind, diagramming software, and presentation and word processing software. This built on my skills with python, and helped to establish my skills and knowledge in the area of reinforcement learning.

The main thing that I wanted to learn about going into this project was reinforcement learning, which was mostly successful. I got to work with pytorch, a python library designed specifically for reinforcement learning, and got to look into current research and ideas in the field of reinforcement learning in order to try and find ways to further our model. While I learned a lot in the theoretical realm, I didn’t get to apply it as much as I would have liked. This due to the scope and time limitations on our project, and as well as hardware limitations. In the course of learning about our project, one major obstacle we had to overcome was adjusting our expectations for the project. We were shooting for a goal that was far too high, and after learning more, decided we should limit our scope to create a more manageable goal. Within this adjusted scope, we had a lot of success, which is described below in part B.
