# Vision-aided training for StyleGAN2

This repo is built upon official StyleGAN2 pytorch repo. For detail please refer to [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch).

The main dependencies are:
- 64-bit Python 3.8 and PyTorch 1.8.0 (or later). See [https://pytorch.org/](https://pytorch.org/) for PyTorch install instructions.
- Cuda toolkit 11.0 or later.
- python libraries: `pip install -r docs/requirements.txt`


## Pretrained Models


#### Large scale datasets

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Dataset</th>
<th valign="bottom">Resolution</th>
<th valign="bottom">download</th>
<th valign="bottom">FID (Inception)</th>
<th valign="bottom">FID (SwAV)</th>

<!-- TABLE BODY -->
<tr><td align="left">LSUN Horse</td>
<td align="left">256</td>
<td align="center"><a href="https://www.cs.cmu.edu/~vision-aided-gan/models/main_paper_table2_fulldataset/vision-aided-gan-lsunhorse-ada-3.pkl">model</a></td>
<td align="center">2.11</td>
<td align="center">0.71</td>
</tr>

<tr><td align="left">LSUN Cat</td>
<td align="left">256</td>
<td align="center"><a href="https://www.cs.cmu.edu/~vision-aided-gan/models/main_paper_table2_fulldataset/vision-aided-gan-lsuncat-ada-3.pkl">model</a></td>
<td align="center">3.98</td>
<td align="center">1.03</td>
</tr>


<tr><td align="left">LSUN Church</td>
<td align="left">256</td>
<td align="center"><a href="https://www.cs.cmu.edu/~vision-aided-gan/models/main_paper_table2_fulldataset/vision-aided-gan-lsunchurch-ada-3.pkl">model</a></td>
<td align="center">1.72</td>
<td align="center">0.58</td>
</tr>

<tr><td align="left">FFHQ</td>
<td align="left">1024</td>
<td align="center"><a href="https://www.cs.cmu.edu/~vision-aided-gan/models/main_paper_table2_fulldataset/vision-aided-gan-ffhq-ada-2.pkl">model</a></td>
<td align="center">3.01</td>
<td align="center">0.38</td>
</tr>

</tbody></table>


#### Limited and Few-shot datasets


<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Dataset</th>
<th valign="bottom">Resolution</th>
<th valign="bottom">download</th>
<th valign="bottom">FID (Inception)</th>
<th valign="bottom">FID (SwAV)</th>

<tr><td align="left">AFHQ Dog</td>
<td align="left">512</td>
<td align="center"><a href="https://www.cs.cmu.edu/~vision-aided-gan/models/main_paper_table3_afhq/vision-aided-gan-afhqcat-ada-3.pkl">model</a></td>
<td align="center">4.73</td>
<td align="center">1.04</td>
</tr>



<tr><td align="left">AFHQ Cat</td>
<td align="left">512</td>
<td align="center"><a href="https://www.cs.cmu.edu/~vision-aided-gan/models/main_paper_table3_afhq/vision-aided-gan-afhqdog-ada-3.pkl">model</a></td>
<td align="center">2.53</td>
<td align="center">0.62</td>
</tr>



<tr><td align="left">AFHQ Wild</td>
<td align="left">512</td>
<td align="center"><a href="https://www.cs.cmu.edu/~vision-aided-gan/models/main_paper_table3_afhq/vision-aided-gan-afhqwild-ada-3.pkl">model</a></td>
<td align="center">2.36</td>
<td align="center">1.10</td>
</tr>



<tr><td align="left">AnimalFace-Dog</td>
<td align="left">256</td>
<td align="center"><a href="https://www.cs.cmu.edu/~vision-aided-gan/models/main_paper_table4_fewshot/vision-aided-gan-animalface_dog-ada-3.pkl">model</a></td>
<td align="center">32.56</td>
<td align="center">6.47</td>
</tr>



<tr><td align="left">AnimalFace-Cat</td>
<td align="left">256</td>
<td align="center"><a href="https://www.cs.cmu.edu/~vision-aided-gan/models/main_paper_table4_fewshot/vision-aided-gan-animalface_cat-ada-3.pkl">model</a></td>
<td align="center">27.35</td>
<td align="center">5.18</td>
</tr>


<tr><td align="left">100-shot Bridge-of-Sighs</td>
<td align="left">256</td>
<td align="center"><a href="https://www.cs.cmu.edu/~vision-aided-gan/models/main_paper_table4_fewshot/vision-aided-gan-bridge-diffaug-3.pkl">model</a></td>
<td align="center">34.35</td>
<td align="center">3.46</td>
</tr>



</tbody></table>


Other pre-trained models including experiments with varying training samples can be downloaded at this [link](https://www.cs.cmu.edu/~vision-aided-gan/models/).

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

**worst sample analysis**

```.bash
python calc_metrics.py --metrics sort_likelihood --name afhq_dog --split train --network https://www.cs.cmu.edu/~vision-aided-gan/models/main_paper_table3_afhq/vision-aided-gan-afhqdog-ada-3.pkl --data afhqdog
```
Example command to create similar visualization as shown [here](https://github.com/nupurkmr9/vision-aided-gan#worst-sample-visualzation). The output image is saved in `out` directory for the above command. 

## Datasets

Dataset preparation is same as given in [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/README.md#preparing-datasets).

Example 100-shot AnimalFace Dog dataset
```.bash
mkdir datasets
wget https://data-efficient-gans.mit.edu/datasets/AnimalFace-dog.zip -P datasets
```

All datasets can be downloaded from their repsective websites:

[FFHQ](https://github.com/NVlabs/ffhq-dataset), [LSUN Categories](http://dl.yf.io/lsun/objects/), [AFHQ](https://github.com/clovaai/stargan-v2), [AnimalFace Dog](https://data-efficient-gans.mit.edu/datasets/AnimalFace-dog.zip), [AnimalFace Cat](https://data-efficient-gans.mit.edu/datasets/AnimalFace-cat.zip), [100-shot Bridge-of-Sighs](https://data-efficient-gans.mit.edu/datasets/100-shot-bridge_of_sighs.zip)


## Training 

**Vision-aided GAN training with multiple off-the-shelf models**:
```.bash
python vision-aided-gan.py --outdir models/ --data datasets/AnimalFace-dog.zip --cfg paper256_2fmap  --mirror 1 \
--aug ada --augpipe bgc --augcv ada --batch 16 --gpus 2 --kimgs-list '1000,1000,1000'  --num 3
```

The network, sample generated images, and logs are saved at regular intervals (controlled by `--snap` flag) in `<outdir>/<exp-folder>` dir, where `<exp-folder>` name is based on input args. Network with each progressive additin of pretrained model is saved in a different directory. Logs are saved as TFevents by default. Wandb logging can be enabled by `--wandb-log` flag and setting wandb `entity` in `training.training_loop`. If fine-tuning a baseline model with vision-aided adversarial loss include `--resume <network.pkl>` in the above command. 

`--kimgs-list` controls the number of iterations after which next off-the-shelf model is added. It is a comma separated list of iteration numbers. For dataset with training samples 1k, we initialize `--kimgs-list` to '4000,1000,1000', and for training samples >1k '8000,2000,2000'.


**Vision-aided Gan training with a single off-the-shelf model**

```.bash
python train.py --outdir models/ --data datasets/AnimalFace-dog.zip --kimg 10000 --cfg paper256_2fmap --gpus 2 \
--cv input-clip-output-conv_multi_level --cv-loss multilevel_sigmoid_s --augcv ada --mirror 1 --aug ada --warmup 5e5
```

**model selection**: returns the computer vision model with highest linear probe accuracy for the best FID model in a folder or the given network file.

```.bash
python model_selection.py --data mydataset.zip --network  <mynetworkfolder or mynetworkpklfile>
```


**To add you own pretrained Model**:
create the class file to extract pretrained features as `vision_module/<custom_model>.py`. Add the class path in the `class_name_dict` in `vision_module.cvmodel.CVBackbone` class. Update the architecture of trainable classifier head over pretrained features in `vision_module.cv_discriminator`.



<details ><summary> <b>Training configuration details</b> </summary> 

Training configuration corresponding to training with our loss:
* `--cv=input-<cv_type>-output-<output_type>` pretrained network and its configuration.
* `--warmup=0` should be number of iterations after which vision-aided loss is added (~5e5) when training from scratch. Introduces our loss after training with warmup images of training. 
* `--cv-loss=multilevel_sigmoid_s` what loss to use on pretrained model based discriminator as described [here](https://github.com/nupurkmr9/vision-aided-gan#vision-aided-discriminator-in-a-custom-gan-model).
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
* `--clean=1` enables FID calculation using [clean-fid](https://github.com/GaParmar/clean-fid) if the real distribution statistics are pre-calculated. default is False.

Run `python train.py --help` for more details and the full list of args.
</details>





