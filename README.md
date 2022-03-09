# Vision-aided GAN


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ensembling-off-the-shelf-models-for-gan/image-generation-on-lsun-churches-256-x-256)](https://paperswithcode.com/sota/image-generation-on-lsun-churches-256-x-256?p=ensembling-off-the-shelf-models-for-gan)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ensembling-off-the-shelf-models-for-gan/image-generation-on-lsun-horse-256-x-256)](https://paperswithcode.com/sota/image-generation-on-lsun-horse-256-x-256?p=ensembling-off-the-shelf-models-for-gan)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ensembling-off-the-shelf-models-for-gan/image-generation-on-lsun-cat-256-x-256)](https://paperswithcode.com/sota/image-generation-on-lsun-cat-256-x-256?p=ensembling-off-the-shelf-models-for-gan)


### [video](https://youtu.be/oHdyJNdQ9E4) | [website](https://www.cs.cmu.edu/~vision-aided-gan/) |   [paper](https://arxiv.org/abs/2112.09130)

<br>

<div class="gif">
<p align="center">
<img src='docs/vision-aided-gan.gif' align="center" width=800>
</p>
</div>

Can the collective *knowledge* from a large bank of pretrained vision models be leveraged to improve GAN training? If so, with so many models to choose from, which one(s) should be selected, and in what manner are they most effective?

We find that pretrained computer vision models can significantly improve performance when used in an ensemble of discriminators.  We propose an effective selection mechanism, by probing the linear separability between real and fake samples in pretrained model embeddings, choosing the most accurate model, and progressively adding it to the discriminator ensemble. Our method can improve GAN training in both limited data and large-scale settings.


Ensembling Off-the-shelf Models for GAN Training <br>
[Nupur Kumari](https://nupurkmr9.github.io/), [Richard Zhang](https://richzhang.github.io/), [Eli Shechtman](https://research.adobe.com/person/eli-shechtman/), [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/)<br>
In CVPR 2022

## Quantitative Comparison

<p align="center">
<img src="docs/lsun_eval.jpg" width="800px"/><br>
</p>

Our method outperforms recent GAN training methods by a large margin, especially in limited sample setting. For LSUN Cat, we achieve similar FID as StyleGAN2 trained on the full dataset using only 0.7\% of the dataset.  On the full dataset, our method improves FID by 1.5x to 2x on cat, church, and horse categories of LSUN.

## Example Results
Below, we show visual comparisons between the baseline StyleGAN2-ADA and our model (Vision-aided GAN) for the
same randomly sample latent code.

<img src="docs/lsuncat1k_compare.gif" width="800px"/>

<img src="docs/ffhq1k_compare.gif" width="800px"/>

## Interpolation Videos
Latent interpolation results of models trained with our method on AnimalFace Cat (160 images), Dog (389 images),  and  Bridge-of-Sighs (100 photos).

<p align="center">
<img src="docs/interp.gif" width="800px"/>
</p>

## Worst sample visualzation
We randomly sample 5k images and sort them according to Mahalanobis distance using mean and variance of real samples calculated in inception feature space. Below visualization shows the bottom 30 images according to the distance for StyleGAN2-ADA (left) and our model (right).

<details open><summary>AFHQ Dog</summary>
<p>
<div class="images">
 <table width=500>
  <tr>
    <td valign="top"><img src="docs/afhqdog_worst_baseline.jpg"/></td>
    <td valign="top"><img src="docs/afhqdog_worst_ours.jpg"/></td>
  </tr>
</table>
</div>
</p>
</details>

<details><summary>AFHQ Cat</summary>
<p>
<div class="images">
 <table>
  <tr>
    <td valign="top"><img src="docs/afhqcat_worst_baseline.jpg"/></td>
    <td valign="top"><img src="docs/afhqcat_worst_ours.jpg"/></td>
  </tr>
</table>
</div>
</p>
</details>

<details><summary>AFHQ Wild</summary>
<p>
<div class="images">
 <table>
  <tr>
    <td valign="top"><img src="docs/afhqwild_worst_baseline.jpg"/></td>
    <td valign="top"><img src="docs/afhqwild_worst_ours.jpg"/></td>
  </tr>
</table>
</div>
</p>
</details>

## Requirements

* 64-bit Python 3.8 and PyTorch 1.8.0 (or later). See [https://pytorch.org/](https://pytorch.org/) for PyTorch install instructions.
* Cuda toolkit 11.0 or later.
* python libraries: see scripts/requirements.txt
* StyleGAN2 code relies heavily on custom PyTorch extensions. For detail please refer to the repo [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)

To setup conda env with all requirements and pretrained networks run the following command:
```.bash
conda create -n vgan python=3.8
conda activate vgan
git clone https://github.com/nupurkmr9/vision-aided-gan.git
cd vision-aided-gan
bash docs/setup.sh
```

For details on off-the-shelf models please see [MODELS.md](docs/MODELS.md)

## Using Pretrained Models
Our final trained models can be downloaded at this [link](https://www.cs.cmu.edu/~vision-aided-gan/models/)

**To generate images**: 


```.bash
# random image generation from LSUN Church model

python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 --network=https://www.cs.cmu.edu/~vision-aided-gan/models/main_paper_table2_fulldataset/vision-aided-gan-lsunchurch-ada-3.pkl
```
The above command generates 4 images using the provided seed values and saves it in `out` directory controlled by `--outdir`. Our generator architecture is same as styleGAN2 and can be similarly used in the Python code as described in [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/README.md#using-networks-from-python).

**model evaluation**:
```.bash
python calc_metrics.py --network https://www.cs.cmu.edu/~vision-aided-gan/models/main_paper_table2_fulldataset/vision-aided-gan-lsunchurch-ada-3.pkl --metrics fid50k_full --data lsunchurch --clean 1
```
We use [clean-fid](https://github.com/GaParmar/clean-fid) library to calculate FID metric. We calclate the full real distribution statistics for FID calculation. For details on calculating the statistics, please refer to [clean-fid](https://github.com/GaParmar/clean-fid).
For default FID evaluation of StyleGAN2-ADA use `clean=0`. The above command will return the FID `~1.72`


## Datasets

Dataset preparation is same as given in [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/README.md#preparing-datasets).
Example setup for 100-shot AnimalFace Dog and LSUN Church

**AnimalFace Dog**
```.bash
mkdir datasets
wget https://data-efficient-gans.mit.edu/datasets/AnimalFace-dog.zip -P datasets
```

**LSUN Church**
```.bash
cd ..
git clone https://github.com/fyu/lsun.git
cd lsun
python3 download.py -c church_outdoor
unzip church_outdoor_train_lmdb.zip
cd ../vision-aided-gan
mkdir datasets
python dataset_tool.py --source ../lsun/church_outdoor_train_lmdb/ --dest datasets/church1k.zip --max-images 1000  --transform=center-crop --width=256 --height=256
```

All other datasets can be downloaded from their repsective websites:

[FFHQ](https://github.com/NVlabs/ffhq-dataset), [LSUN Categories](http://dl.yf.io/lsun/objects/), [AFHQ](https://github.com/clovaai/stargan-v2), [AnimalFace Dog](https://data-efficient-gans.mit.edu/datasets/AnimalFace-dog.zip), [AnimalFace Cat](https://data-efficient-gans.mit.edu/datasets/AnimalFace-cat.zip), [100-shot Bridge-of-Sighs](https://data-efficient-gans.mit.edu/datasets/100-shot-bridge_of_sighs.zip)


## Training new networks

**Vision-aided GAN training with multiple pretrained networks**:
```.bash
python vision-aided-gan.py --outdir models/ --data datasets/AnimalFace-dog.zip --cfg paper256_2fmap  --mirror 1 \
--aug ada --augpipe bgc --augcv ada --batch 16 --gpus 2 --kimgs-list '1000,1000,1000'  --num 3
```

The network, sample generated images, and logs are saved at regular intervals (controlled by `--snap` flag) in `models/<exp-folder>` dir, where `<exp-folder>` name is based on input args. Network with each progressive additin of pretrained model is saved in a different directory. Logs are saved as TFevents by default. Wandb logging can be enabled by `--wandb-log` flag and setting wandb `entity` in `training.training_loop`.

If fine-tuning a baseline trained model with vision-aided adversarial loss include `--resume <network.pkl>` in the above command. 

`--kimgs-list` controls the number of iterations after which next off-the-shelf model is added. It is a comma separated list of iteration numbers. For dataset with training samples 1k, we initialize `--kimgs-list` to '4000,1000,1000', and for training samples >1k '8000,2000,2000'.


**Vision-aided Gan training with a specific pretrained network**

```.bash
python train.py --outdir models/ --data datasets/AnimalFace-dog.zip --kimg 10000 --cfg paper256_2fmap --gpus 2 \
--cv input-clip-output-conv_multi_level --cv-loss multilevel_s --augcv ada --mirror 1 --aug ada --warmup 1 
```

**model selection**: returns the computer vision model with highest linear probe accuracy for the best FID model in a folder or the given network file.

```.bash
python model_selection.py --data mydataset.zip --network  <mynetworkfolder or mynetworkpklfile>
```

**To add you own pretrained Model**:
create the class file to extract pretrained features inside `vision_model` folder. Add the class path in the `class_name_dict` in `vision_model.cvmodel.CVWrapper` class. Update the architecture of trainable classifier head over pretrained features in `training.cv_discriminator`.



<details ><summary> <b>Training configuration details</b> </summary> 

Training configuration corresponding to training with our loss:

* `--cv=input-dino-output-conv_multi_level` pretrained network and its configuration.
* `--warmup=0` should be enabled when training from scratch. Introduces our loss after training with 500k images.
* `--cv-loss=multilevel` what loss to use on pretrained model based discriminator.
* `--augcv=ada` performs ADA augmentation on pretrained model based discriminator.
* `--augcv=diffaugment-<policy>` performs DiffAugment on pretrained model based discriminator with given poilcy e.g. `color,translation,cutout`
* `--augpipecv=bgc` ADA augmentation strategy. Note: cutout is always enabled. 
* `--ada-target-cv=0.3` adjusts ADA target value for pretrained model based discriminator.
* `--exact-resume=1` enables resume along with optimizer and augmentation state. default is 0.

StyleGAN2 configurations:
* `--outdir='models/'` directory to save training runs.
* `--data` data directory created after running `dataset_tool.py`.
* `--metrics=fid50kfull` evaluates FID calculation during training at every `snap` iterations.
* `--cfg=paper256` architecture and hyperparameter configuration for G and D. 
* `--mirror=1` enables horizontal flipping
* `--aug=ada` enables ADA augmentation in trainable D. 
* `--diffaugment=color,translation,cutout` enables DiffAugment in trainable D.
* `--augpipe=bgc` ADA augmentation strategy in trainable D.
* `--snap=25` evaluation and model saving interval

Miscellaneous configurations:
* `--wandb-log=1` enables wandb logging.
* `--clean=1` enables FID calculation using [clean-fid](https://github.com/GaParmar/clean-fid) if the real distribution statistics are pre-calculated. default is false.

Run `python train.py --help` for more details and the full list of args.
</details>


## References

```
@article{kumari2021ensembling,
  title={Ensembling Off-the-shelf Models for GAN Training},
  author={Kumari, Nupur and Zhang, Richard and Shechtman, Eli and Zhu, Jun-Yan},
  journal={arXiv preprint arXiv:2112.09130},
  year={2021}
}
```

## Acknowledgments
We thank Muyang Li, Sheng-Yu Wang, Chonghyuk (Andrew) Song for proofreading the draft. We are also grateful to Alexei A. Efros, Sheng-Yu Wang, Taesung Park, and William Peebles for helpful comments and discussion. Our codebase is built on [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) and [ DiffAugment](https://github.com/mit-han-lab/data-efficient-gans).
