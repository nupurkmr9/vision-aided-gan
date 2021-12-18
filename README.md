# Vision-aided GAN

### [video (3m)](https://youtu.be/oHdyJNdQ9E4) | [website](https://www.cs.cmu.edu/~vision-aided-gan/) |   [paper](https://arxiv.org/abs/2112.09130)
<br>

<div class="gif">
<img src='images/vision-aided-gan.gif' align="right" width=1000>
</div>

<br><br><br><br><br>


Can the collective *knowledge* from a large bank of pretrained vision models be leveraged to improve GAN training? If so, with so many models to choose from, which one(s) should be selected, and in what manner are they most effective?

We find that pretrained computer vision models can significantly improve performance when used in an ensemble of discriminators.  We propose an effective selection mechanism, by probing the linear separability between real and fake samples in pretrained model embeddings, choosing the most accurate model, and progressively adding it to the discriminator ensemble. Our method can improve GAN training in both limited data and large-scale settings.




Ensembling Off-the-shelf Models for GAN Training <br>
[Nupur Kumari](https://nupurkmr9.github.io/), [Richard Zhang](https://richzhang.github.io/), [Eli Shechtman](https://research.adobe.com/person/eli-shechtman/), [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/)<br>
arXiv 2112.09130, 2021

## Quantitative Comparison
<img src="images/lsun_eval.jpg" width="800px"/><br>
Our method outperforms recent GAN training methods by a large margin, especially in limited sample setting. For LSUN Cat, we achieve similar FID as StyleGAN2 trained on the full dataset using only $0.7\%$ of the dataset.  On the full dataset, our method improves FID by 1.5x to 2x on cat, church, and horse categories of LSUN.

## Example Results
Below, we show visual comparisons between the baseline StyleGAN2-ADA and our model (Vision-aided GAN) for the
same randomly sample latent code.

<img src="images/lsuncat1k_compare.gif" width="800px"/>

<img src="images/ffhq1k_compare.gif" width="800px"/>

## Interpolation Videos
Latent interpolation results of models trained with our method on AnimalFace Cat (160 images), Dog (389 images),  and  Bridge-of-Sighs (100 photos).

<img src="images/interp.gif" width="800px"/><br>


## Requirements

* 64-bit Python 3.8 and PyTorch 1.8.0 (or later). See [https://pytorch.org/](https://pytorch.org/) for PyTorch install instructions.
* Cuda toolkit 11.0 or later.
* python libraries: see requirements.txt
* StyleGAN2 code relies heavily on custom PyTorch extensions. For detail please refer to the repo [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)


## Setting up Off-the-shelf Computer Vision models


**[CLIP(ViT)](https://github.com/openai/CLIP)**: we modify the model.py function to return intermediate features of the transformer model. To set up follow these steps.

```.bash
git clone https://github.com/openai/CLIP.git
cp vision-aided-gan/training/clip_model.py CLIP/clip/model.py
cd CLIP
python setup.py install
```

**[DINO(ViT)](https://github.com/facebookresearch/dino)**: model is automatically downloaded from torch hub.

**[VGG-16](https://github.com/adobe/antialiased-cnns)**: model is automatically downloaded.


**[Swin-T(MoBY)](https://github.com/SwinTransformer/Transformer-SSL)**: Create a `pretrained-models` directory and save the downloaded [model](https://drive.google.com/file/d/1PS1Q0tAnUfBWLRPxh9iUrinAxeq7Y--u/view?usp=sharing) there.


**[Swin-T(Object Detection)](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection)**: follow the below step for setup. Download the model [here](https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_tiny_patch4_window7_512x512.pth) and save it in the `pretrained-models` directory.
```.bash
git clone https://github.com/SwinTransformer/Swin-Transformer-Object-Detection
cd Swin-Transformer-Object-Detection
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
python setup.py install
```


**[Swin-T(Segmentation)](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation)**: follow the below step for setup. Download the model [here](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_tiny_patch4_window7.pth) and save it in the `pretrained-models` directory.
```.bash
git clone https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation.git
cd Swin-Transformer-Semantic-Segmentation
remove assert statement from __init__.py
python setup.py install
```

**[Face Parsing](https://github.com/switchablenorms/CelebAMask-HQ)**:download the model [here](https://drive.google.com/file/d/1o1m-eT38zNCIFldcRaoWcLvvBtY8S4W3/view?usp=sharing) and save in the `pretrained-models` directory.

**[Face Normals](https://github.com/boukhayma/face_normals)**:download the model [here](https://drive.google.com/file/d/1Qb7CZbM13Zpksa30ywjXEEHHDcVWHju_) and save in the `pretrained-models` directory.

## Pretrained Models
Our final trained models can be downloaded at this [link](https://www.cs.cmu.edu/~vision-aided-gan/models/)

**To generate images**: 

```.bash
python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 --network=<network.pkl>
```
The output is stored in `out` directory controlled by `--outdir`. Our generator architecture is same as styleGAN2 and can be similarly used in the Python code as described in [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/README.md#using-networks-from-python).

**model evaluation**:
```.bash
python calc_metrics.py --network <network.pkl> --metrics fid50k_full --data <dataset> --clean 1
```
We use [clean-fid](https://github.com/GaParmar/clean-fid) library to calculate FID metric. For LSUN Church and LSUN Horse, we calclate the full real distribution statistics. For details on calculating the real distribution statistics, please refer to [clean-fid](https://github.com/GaParmar/clean-fid).
For default FID evaluation of StyleGAN2-ADA use `clean=0`. 

## Datasets

Dataset preparation is same as given in [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/README.md#preparing-datasets).
Example setup for LSUN Church


**LSUN Church**
```.bash
git clone https://github.com/fyu/lsun.git
cd lsun
python3 download.py -c church_outdoor
unzip church_outdoor_train_lmdb.zip
cd ../vision-aided-gan
python dataset_tool.py --source <path-to>/church_outdoor_train_lmdb/ --dest <path-to-datasets>/church1k.zip --max-images 1000  --transform=center-crop --width=256 --height=256
```

datasets can be downloaded from their repsective websites:

[FFHQ](https://github.com/NVlabs/ffhq-dataset), [LSUN Categories](http://dl.yf.io/lsun/objects/), [AFHQ](https://github.com/clovaai/stargan-v2), [AnimalFace Dog](https://data-efficient-gans.mit.edu/datasets/AnimalFace-dog.zip), [AnimalFace Cat](https://data-efficient-gans.mit.edu/datasets/AnimalFace-cat.zip), [100-shot Bridge-of-Sighs](https://data-efficient-gans.mit.edu/datasets/100-shot-bridge_of_sighs.zip)


## Training new networks

**model selection**: returns the computer vision model with highest linear probe accuracy for the best FID model in a folder or the given network file.

```.bash
python model_selection.py --data mydataset.zip --network  <mynetworkfolder or mynetworkpklfile>
```

**example training command for training with a single pretrained network from scratch**

```.bash
python train.py --outdir=training-models/ --data=mydataset.zip --gpus 2 --metrics fid50k_full --kimg 25000 --cfg paper256 --cv input-dino-output-conv_multi_level --cv-loss multilevel_s --augcv ada --ada-target-cv 0.3 --augpipecv bgc --batch 16 --mirror 1 --aug ada --augpipe bgc --snap 25 --warmup 1  
```

Training configuration corresponding to training with vision-aided-loss:

* `--cv=input-dino-output-conv_multi_level` pretrained network and its configuration.
* `--warmup=0` should be enabled when training from scratch. Introduces our loss after training with 500k images.
* `--cv-loss=multilevel` what loss to use on pretrained model based discriminator.
* `--augcv=ada` performs ADA augmentation on pretrained model based discriminator.
* `--augcv=diffaugment-<policy>` performs DiffAugment on pretrained model based discriminator with given poilcy.
* `--augpipecv=bgc` ADA augmentation strategy. Note: cutout is always enabled. 
* `--ada-target-cv=0.3` adjusts ADA target value for pretrained model based discriminator.
* `--exact-resume=0` enables exact resume along with optimizer state.


Miscellaneous configurations:
* `--appendname=''` additional string to append to training directory name.
* `--wandb-log=0` enables wandb logging.
* `--clean=0` enables FID calculation using [clean-fid](https://github.com/GaParmar/clean-fid) if the real distribution statistics are pre-calculated.

Run `python train.py --help` for more details and the full list of args.

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
