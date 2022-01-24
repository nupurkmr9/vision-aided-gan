## Setting up Off-the-shelf Computer Vision models

If `docs/setup.sh` not used, individually setup each model by following the below steps:

**[CLIP(ViT)](https://github.com/openai/CLIP)**: we modify the model.py function to return intermediate features of the transformer model. 

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
pip install mmcv-full==1.3.0 -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
python setup.py install
```

for more details on mmcv installation please refer [here](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md)

**[Swin-T(Segmentation)](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation)**: follow the below step for setup. Download the model [here](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_tiny_patch4_window7.pth) and save it in the `pretrained-models` directory.
```.bash
git clone https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation.git
cd Swin-Transformer-Semantic-Segmentation
python setup.py install
```

**[Face Parsing](https://github.com/switchablenorms/CelebAMask-HQ)**:download the model [here](https://drive.google.com/file/d/1o1m-eT38zNCIFldcRaoWcLvvBtY8S4W3/view?usp=sharing) and save in the `pretrained-models` directory.

**[Face Normals](https://github.com/boukhayma/face_normals)**:download the model [here](https://drive.google.com/file/d/1Qb7CZbM13Zpksa30ywjXEEHHDcVWHju_) and save in the `pretrained-models` directory.

