# Vision-aided training for StyleGAN3

This repo is built upon official StyleGAN3 pytorch repo. For more details and setup please refer to [stylegan3](https://github.com/NVlabs/stylegan3).

The main dependencies are:
* 64-bit Python 3.8 and PyTorch 1.9.0 (or later). See [https://pytorch.org/](https://pytorch.org/) for PyTorch install instructions.
* Cuda toolkit 11.1 or later.
* python libraries: `pip install -r requirements.txt`


## Training

**Vision-aided Gan training with a single off-the-shelf model**

```.bash
#install if not already done
pip install vision_aided_loss

# download dataset
mkdir datasets
wget https://data-efficient-gans.mit.edu/datasets/AnimalFace-dog.zip -P datasets

# training
python train.py --outdir models/ --data datasets/AnimalFace-dog.zip --kimg 4000 --cfg stylegan3-t --gpus 2 --gamma 10 \
--batch 16 --cv input-clip-output-conv_multi_level --cv-loss multilevel_sigmoid_s --mirror 1 --aug ada --warmup 5e5 

# Diffaugment is used to augment images input to the vision-aided discriminator.
```

<details ><summary> <b>Training configuration details</b> </summary> 

Training configuration corresponding to training with our loss:

* `--cv=input-<cv_type>-output-<output_type>` pretrained network and its configuration as explained [here](https://github.com/nupurkmr9/vision_aided_module#vision-aided-discriminator-in-a-custom-gan-model).
* `--cv-loss=multilevel_sigmoid_s` what loss to use on pretrained model based discriminator as described [here](https://github.com/nupurkmr9/vision_aided_module#vision-aided-discriminator-in-a-custom-gan-model).
* `--augcv=ada` performs ADA augmentation on pretrained model based discriminator.
* `--exact-resume=1` enables resume along with optimizer and augmentation state. default is 0.
* `--warmup=0` should be number of iterations after which vision-aided loss is added (~5e5) when training from scratch. Introduces our loss after training with warmup images of training. If resuming warmup should be 0.

StyleGAN3 default configurations:
* `--outdir='models/'` directory to save training runs.
* `--data` dataset path.
* `--metrics=fid50kfull` evaluates FID calculation during training at every `snap` iterations.
* `--cfg=stylegan3-t` architecture and hyperparameter configuration for G and D. 
* `--mirror=1` enables horizontal flipping
* `--aug=ada` enables ADA augmentation in styleGAN Discriminator. 
* `--snap=25` evaluation and model saving interval

Miscellaneous configurations:
* `--wandb-log=1` enables wandb logging.
* `--clean=1` enables FID calculation using [clean-fid](https://github.com/GaParmar/clean-fid) if the real distribution statistics are pre-calculated. default is False.

Run `python train.py --help` for more details and the full list of args.
</details>
