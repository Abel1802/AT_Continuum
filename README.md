We synthesize speech continuum based on adversarial-training(AT) approach.


## Dependancy
- python 3.6+
- pytorch 1.0+
- pyworld 
- praat-parselmouth

## Folder Structure
  ```
  AT_Continuum/
  |
  |--preprocess.py - extract features(mel,f0,F1,F2)
  |
  |--train.py - main script to start train
  |
  |--test.py - evaluation of trained model
  |
  |--exp/ saved dir
  |  |--exp1/
  |     |--checkpoints/ - saved model
  |     |--result.log - logging output
  |
  |--configs/ - configurations for training
  |  |--base.yaml - base configuration
  |
  |--data_loader/ - anything about data loading
  |  |--data_loader.py
  |
  |--model/ - model archit
  |  |--model.py
  |  |--mi_estimator.py
  |
  |--utils/ - samll utility functions
     |--util.py
  ```

## Preprocess
Our model is trained on [BLCU-SAIT Corpus](https://ieeexplore.ieee.org/abstract/document/7919008)

There is an example.(using F001/*.wav) - python preprocess.py

## Usage

### Training
You can start training by running python train.py. The arguments are listed below.
- --data_dir: the dir of training data
- --lambda_disc: the hyper-parameters

### Testing
You can inference by running python test.py. The arguments are listed below.

## Contact
If you have any question about the paper or the code, feel free to email me at lzblcu19@gmail.com.
