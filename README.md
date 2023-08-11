# Deep_COTS
YOLOX custom code for the Kaggle [Crown of Thorns Startfish (COTS)](https://www.kaggle.com/competitions/tensorflow-great-barrier-reef) competition. The objective of this competition was to detect all instances of COTS automatically so that a UAV could help divers to prevent the irreversible damange to the Great Barrier Reef.

## Usage
First, if the link to the competition is still available (the competition closed in 2022), you will need to download the COTS dataset and extract the data.

Next download and extract this repository:
```console
$ cd ~
$ mkdir cots
$ cd cots
### Make a folder for datasets
$ mkdir dataset
### Clone this repo
$ git clone https://github.com/Joshuat38/Deep_COTS.git
```

### Training
Unfortunately, no pre-trained models are available so you will need to train the model. You can do this using:
```shell
$ cd ~/cots/Deep_COTS
$ python main.py --mode train --batch_size 4 --num_epochs 100 --gpu_id 0
```

### Testing
To test your trained model, use:
```shell
$ cd ~/cots/Deep_COTS
$ python main.py --mode test --batch_size 1 --gpu_id 0 --pretrained_model ./model/<name_of_your_saved_model>/model_checkpoint
```

