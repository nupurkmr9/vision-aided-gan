pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r docs/requirements.txt

##CLIP###
cd ..
git clone https://github.com/openai/CLIP.git
cp vision-aided-gan/vision_model/clip_model.py CLIP/clip/model.py
cd CLIP
python setup.py install

##Swin-T object detection###
cd ..
git clone https://github.com/SwinTransformer/Swin-Transformer-Object-Detection
cd Swin-Transformer-Object-Detection
pip install mmcv-full==1.3.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.1/index.html
python setup.py install
cd ..

##Swin-T object segmentation###
git clone https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation.git
cd Swin-Transformer-Semantic-Segmentation
python setup.py install
cd ..

##Model download###
cd vision-aided-gan
mkdir pretrained-models
cd pretrained-models
#swin-T MoBY model#
gdown --id 1PS1Q0tAnUfBWLRPxh9iUrinAxeq7Y--u
#swin-T detection model#
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_tiny_patch4_window7_512x512.pth
#swin-T segmentation model#
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_tiny_patch4_window7.pth
#face parsing model#
gdown --id 1o1m-eT38zNCIFldcRaoWcLvvBtY8S4W3
mv model.pth celeba_parsing_model.pth
#face normals model#
gdown --id 1Qb7CZbM13Zpksa30ywjXEEHHDcVWHju_
mv model.pth normal_model.pth
