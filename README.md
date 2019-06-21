# Freesound Audio Tagging 2019 Competition

## How to use 
* Install fastai and librosa:
```bash
conda install -c pytorch -c fastai fastai
conda install librosa
```

* Clone the repository:
```bash
git clone https://github.com/mnpinto/audiotagging2019.git
```

* Download the competition data from kaggle (https://www.kaggle.com/c/freesound-audio-tagging-2019/data) to `audiotagging2019/data/` folder and unzip the `test.zip`, `train_curated.zip` and `train_noisy.zip` to folders `test`, `train_curated` and `train_noisy`. 

* On `audiotagging2019` folder run:
```bash
python run.py --n_epochs 1 --max_processors 8
```
If successful the script will create `train_curated_png` and `train_noisy_png` folders with the Mel spectrograms corresponding to all audio clips and train the model for 1 epochs using the default arguments. The `max_processors` argument will set how many processors to use to this preprocessing step. After the training is complete a folder `models` will be created and a weights file `stage-1.pth` will be saved their. Finally a submission file will be generated with the default name `submission.csv`.

**If you find any errors let me know by creating an Issue, the code has not yet been tested on fastai versions after 1.0.51.**

## Replicating my top scoring solution
**Important! This code has not yet been tested. I ran all experiments on Kaggle kernels and refactored the code to create this repository. After the final results of the competition are available, late submissions will be allowed so I will then test the code to check if anything is missing.**

My top scoring solution with a score of **0.742 on public LB** and **0.75421 on private LB** is the average of the following 6 runs:
```bash
python run.py --model xresnet18ssa --base_dim 128 --SZ 256 --fold_number -1 \
              --n_epochs 80 --loss_name FocalLoss --weights_file model1 --csv_name submission1.csv
              
python run.py --model xresnet34ssa --base_dim 128 --SZ 256 --fold_number -1 \
              --n_epochs 60 --loss_name FocalLoss --weights_file model2 --csv_name submission2.csv
              
python run.py --model xresnet18ssa --base_dim 128 --SZ 256 --fold_number -1 \
              --n_epochs 80 --weights_file model3 --csv_name submission3.csv
              
python run.py --model xresnet34ssa --base_dim 128 --SZ 256 --fold_number -1 \
              --n_epochs 60 --weights_file model4 --csv_name submission4.csv

python run.py --model models.xresnet34 --base_dim 128 --SZ 256 --fold_number -1 \
              --n_epochs 90 --loss_name FocalLoss --weights_file model5 --csv_name submission5.csv
              
python run.py --model models.xresnet50 --base_dim 128 --SZ 256 --fold_number -1 \
              --n_epochs 65 --weights_file model6_0
              
python run.py --model models.xresnet50 --base_dim 128 --SZ 256 --fold_number -1 \
              --n_epochs 65 --load_weights model6 --weights_file model6 --csv_name submission6.csv             

```
The penultimate run, generating `model6_0` weights is not used for the ensemble, is just to generate the weights that are used to the last identical run. If you are running locally, try a single run with more epochs, the 2x65 epochs is just to accommodate for the 9h run-time limit of Kaggle kernels.
