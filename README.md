# Overview

This repository contains the code used for the paper [Towards Land Vehicle Ego-Velocity Estimation using Deep Learning and Automotive Radars](). The paper proposes a deep learning framework for estimating ego-velocity of land vehicles equipped with FMCW radars.

![radarscenes example](assets/radarscenes_example.png)

# Installation

The project was tested on `Ubuntu 20.04` and `Python 3.8.10`. Other used libraries can be found in the [requirements.txt](requirements.txt) file. After cloning the repository, you can create a custom environment using the following command:

```bash
python3 python3 -m venv env
```

The, proceed with the installation of the required libraries:

```bash
source env/bin/activate
pip3 install -r requirements.txt
```

# Data

Until the publication of this repository, the NavINST dataset was not publicly available. After downloading the RadarScenes dataset, please extract it into the [radarscenes](data/radarscenes/) folder so that the contents appear as follows:

![radarscenes data](assets/radarscenes_data.png)

# Code

Inside the [src](src/) folder you can find the codes for each dataset tested. The [utils](src/utils/) folder contains common scripts. The [checkpoints](checkpoints/) folder contains the checkpoints for the proposed models. Finally, the [runs](runs/) folder contains Tensorboard checkpoints for analysis.

Detailed descriptions of the scripts, in order of their utilization, are provided below.

### RadarScenes

| Script | Description |
|:----------------:|---------------|
| [meta](src/radarscenes/meta.py) | Used to compute the lever arm between the vehicle coordinate system and all radars. The acquired information is used inside the [compare](src/radarscenes/compare.py) script. |
| [preprocess](src/radarscenes/preprocess.py) | Used to define the sequences included in the training, validation and test sets used by the [main](src/radarscenes/main.py) script. |
| [dataset](src/radarscenes/dataset.py) | Contains the PyTorch Dataset classes. |
| [network](src/radarscenes/network.py) | Contains the PyTorch Network classes. |
| [main](src/radarscenes/main.py) | Used to train and test the models. By default, the test script saves the estimated outputs as CSV file used by the [compare](src/radarscenes/compare.py) script.|
| [compare](src/radarscenes/compare.py) | Used to compare the proposed models and the benchmarks. Run [main](src/radarscenes/main.py) script in test mode using the desired weights for each sequence available in the test set before comparing the results.|

# Citation
If you find this repository helpful, please consider citing:
<!-- ```bibtex

``` -->