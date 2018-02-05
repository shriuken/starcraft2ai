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
