# User Docs
## Setup
starcraft2ai is dependence upon the following libraries:
* [PySC2](https://github.com/deepmind/pysc2#quick-start-guide)
* [PyTorch](http://pytorch.org/)

## Installation
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
  
## Running starcraft2ai
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
  

# FAQ
* Q: Do I need to buy SC2?
* A: No! The starter edition can be downloaded and used for free.
* Q: Can I play against the AI?
* A: Not yet! Maybe one day. Maybe you are the AI.
