Below is outdated, please see the following Google doc in the interim: 

https://docs.google.com/document/d/1bmLdmKxgRm8E5rQYoWU0MRSmNFMHMkIyGDqbm7fOvgQ/edit?usp=sharing


# Overall Test Plan
We have two overall approaches to to testing our Starcraft 2 AI - code coverage tests and full-system tests. Due to the nature of our project, we aren’t fully able to test the learning algorithms we write, thus we will test small parts of the learning code, to ensure proper aspects are (or are not) changing. Much of our full-system tests will be verifying and inspecting the result of our learning algorithms, and spectating the bot gameplay and win conditions.

# Test Case Descriptions
1. Learning Algorithm Unit Test
  1. Assert that a learned variables changes between one step of the learning algorithm.
  1. Assert that non-learned variables do not change between one step of the learning algorithm.
  
