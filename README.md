# 6th place solution for Freesound Audio Tagging 2019 Competition

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

## Arguments
|name|type|default|description|
|---|---|---|---|
|`--path`|`str`|`data`|path to data folder| 
|`--working_path`|`str`|`.`|path to working folder where model weights and outputs will be saved|
|`--base_dim`|`int`|`128`|size to crop the images on the horizontal axis before rescaling with `SZ`| 
|`--SZ`|`int`|`128`|images will be rescaled to `SZxSZ`| 
|`--BS`|`int`|`64`|batch size| 
|`--lr`|`float`|`0.01`|maximum learning rate for `one_cycle_learning`| 
|`--n_epochs`|`int`|`80`|number of epochs to train the model| 
|`--epoch_size`|`int`|`1000`|number of episodes (with batch size `BS` each) in each epoch| 
|`--f2cl`|`int`|`1`|train only on samples with F2 score (with threshold of 0.2) less than `f2cl`| 
|`--fold_number`|`int`|`0`|KFold cross-validation fold number: (0,1,2,3,4) or -1 to train with all data| 
|`--loss_name`|`str`|`BCELoss`|loss function to use, options are `BCELoss` and `FocalLoss`| 
|`--csv_name`|`str`|`submission.csv`|name of csv file to save with test predictions| 
|`--model`|`str`|`models.xresnet18`|can be a fastai model as the default or `xresnet{18,34,50}ssa` to use simple self-attention| 
|`--weights_file`|`str`|`stage-1`|name of file to save the weights| 
|`--load_weights`|`str`||provide the name of weights file (e.g., stage-1) to load before training| 
|`--max_processors`|`int`|`8`|number cpu threads to use for converting wav files to png| 
|`--force`|`bool`|`False`|if set to `True` the pngs will be recomputed for noisy and curated train datasets| 



## Replicating my top scoring solution
**Important! This code has not yet been tested. I ran all experiments on Kaggle kernels and refactored the code to create this repository. After the final results of the competition are available, late submissions will be allowed so I will then test the code to check if anything is missing.**

My top scoring solution with a score of **0.742 on public LB** and **0.75421 on private LB** (final results pending...) is the average of the following 6 runs:
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
              --n_epochs 65 --load_weights model6_0 --weights_file model6 --csv_name submission6.csv             

```
The penultimate run, generating `model6_0` weights is not used for the ensemble, is just to generate the weights that are used to the last identical run. If you are running locally, try a single run with more epochs, the 2x65 epochs is just to accommodate for the 9h run-time limit of Kaggle kernels.

# Ablation study (in progress)*
* Fixed parameters: `--base_dim 128 --SZ 256 --fold_number -1 --n_epochs 80 --loss_name FocalLoss`

|Model | private LB scores |
|---|---|
|xresnet18ssa| [0.74211, 0.74695] |
|xresnet34ssa| [0.74545, 0.74875] |

# Citing this repository
```bibtex
@misc{mnpinto2019audio,
  author = {Pinto, M. M.},
  title = {6th place solution for Freesound Audio Tagging 2019 Competition},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/mnpinto/audiotagging2019}}
}
```
