# Overall Test Plan
We have two overall approaches to to testing our Starcraft 2 AI - code coverage tests and full-system tests. Due to the nature of our project, we aren’t fully able to test the learning algorithms we write, thus we will test small parts of the learning code, to ensure proper aspects are (or are not) changing. Much of our full-system tests will be verifying and inspecting the result of our learning algorithms, and spectating the bot gameplay and win conditions.

# Test Case Descriptions

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
| RESULT | ??? |
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
| RESULT | ??? |
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
| RESULT | ??? |
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
| RESULT | ??? |
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
| RESULT | ??? |
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
| RESULT | ??? |
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
| RESULT | ??? |
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
| RESULT | ??? |
| | |

# Misc
This document is updated to reflect the final results of our project. The initial version used during planning is archived [here](https://docs.google.com/document/d/1bmLdmKxgRm8E5rQYoWU0MRSmNFMHMkIyGDqbm7fOvgQ/edit?usp=sharing).
