# Vision-aided-GAN Training

### [video (1m)]() | [website]() |   [paper]()
<br>

<div class="gif">
<img src='images/vision-aided-gan.gif' align="right" width=1000>
</div>

<br><br><br><br><br>


<!-- 
**Ensembling Off-the-shelf Models for GAN Training**<br>
Nupur Kumari, Richard Zhang, Eli Shechtman, Jun-Yan Zhu<br>

Abstract:*The advent of large-scale training has produced a cornucopia of powerful visual recognition models. However, generative models, such as GANs, have traditionally been trained from scratch in an unsupervised manner. Can the collective "knowledge" from a large bank of pretrained vision models be leveraged to improve GAN training? If so, with so many models to choose from, which one(s) should be selected, and in what manner are they most effective? We find that pretrained computer vision models can significantly improve performance when used in an ensemble of discriminators. Notably, the particular subset of selected models greatly affects performance. We propose an effective selection mechanism, by probing the linear separability between real and fake samples in pretrained model embeddings, choosing the most accurate model, and progressively adding it to the discriminator ensemble. Interestingly, our method can improve GAN training in both limited data and large-scale settings. Given only 10k training samples, our FID on LSUN Cat matches the StyleGAN2 trained on 1.6M images. On the full dataset, our method improves FID by 1.5 to 2 times on LSUN Cat and LSUN Church.*


 -->
## Requirements

* 64-bit Python 3.8 and PyTorch 1.8.0 (or later). See [https://pytorch.org/](https://pytorch.org/) for PyTorch install instructions.
* Cuda toolkit 11.0 or later. 
* python libraries: see requirements.txt 
* StyleGAN2 code relies heavily on custom PyTorch extensions. For detail please refer to the repo [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)


## Setting up Off-the-shelf Computer Vision models


**[CLIP(ViT)](https://github.com/openai/CLIP)**: we modify the model.py function to return intermediate features of the transformer model. To set up follow these steps.

```.bash
git clone https://github.com/openai/CLIP.git
mv vision-aided-gan/training/clip_model.py CLIP/clip/model.py
cd CLIP
python setup.py install
```

**[DINO(ViT)](https://github.com/facebookresearch/dino)**: model is automatically downloaded from torch hub.

**[VGG-16](https://github.com/adobe/antialiased-cnns)**: model is automatically downloaded.


**[Swin-T(MoBY)](https://github.com/SwinTransformer/Transformer-SSL)**: Create a "pretrained-models" directory and save the downloaded [model](https://drive.google.com/file/d/1PS1Q0tAnUfBWLRPxh9iUrinAxeq7Y--u/view?usp=sharing) there. 


**[Swin-T(Object Detection)](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection)**: follow the below step for setup. Download the model [here](https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_tiny_patch4_window7_512x512.pth) and save it in the "pretrained-models" directory.
```.bash
git clone https://github.com/SwinTransformer/Swin-Transformer-Object-Detection
cd Swin-Transformer-Object-Detection
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
python setup.py install
```


**[Swin-T(Segmentation)](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation)**: follow the below step for setup. Download the model [here](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_tiny_patch4_window7.pth) and save it in the "pretrained-models" directory.
```.bash
git clone https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation.git
cd Swin-Transformer-Semantic-Segmentation
remove assert statement from __init__.py
python setup.py install
```

**[Face Parsing](https://github.com/switchablenorms/CelebAMask-HQ)**:download the model [here](https://drive.google.com/file/d/1o1m-eT38zNCIFldcRaoWcLvvBtY8S4W3/view?usp=sharing) and save in the "pretrained-models" directory.

**[Face Normals](https://github.com/boukhayma/face_normals)**:download the model [here](https://drive.google.com/file/d/1Qb7CZbM13Zpksa30ywjXEEHHDcVWHju_) and save in the "pretrained-models" directory. 



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
python train.py --outdir=training-models/ --data=mydataset.zip --gpus 2 --metrics fid50k_full --kimg 25000 --cfg ffhq1k --cv input-dino-output-conv_multi_level --cv-loss multilevel_s --augcv ada --ada-target-cv 0.3 --augpipecv bgc --batch 16 --mirror 1 --aug ada --augpipe bgc --snap 25 --warmup 1  
```

Training configuration corresponding to training with vision-aided-loss:

* `--cv=input-dino-output-conv_multi_level` pretrained network and its configuration.
* `--warmup=0` should be enabled when training from scratch. Introduces our loss after training with 500k images.
* `--cv-loss=multilevel` what loss to use on pretrained model based discriminator.
* `--augcv=ada` performs ADA augmentation on pretrained model based discriminator.
* `--augpipecv=bgc` ADA augmentation strategy.
* `--ada-target-cv=0.3` adjusts ADA target value for pretrained model based discriminator.
* `--exact-resume=0` enables exact resume along with optimizer state.


Miscellaneous configurations:
* `--appendname=''` additional string to append to training directory name.
* `--wandb-log=0` enables wandb logging.
* `--clean=0` enables FID calculation using [clean-fid](https://github.com/GaParmar/clean-fid) if the real distribution statistics are pre-calculated.


**Pretrained Models** can be downloaded at this [link](https://www.cs.cmu.edu/~vision-aided-gan/models/)


## Acknowledgments

Our codebase is built over [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch).

## ToDos
- [ ] add a script for training with multiple models

