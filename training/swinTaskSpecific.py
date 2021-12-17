import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.apis import init_segmentor
from mmdet.apis import init_detector

class Seg(torch.nn.Module):

    def __init__(self, cv_type = 'attention' ):
        super().__init__(
        )

        self.cv_type = cv_type
        
        if 'object' in self.cv_type:
            self.model =  init_detector('../Swin-Transformer-Semantic-Segmentation/configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py.py', 
                                   'pretrained-models/cascade_mask_rcnn_swin_tiny_patch4_window7.pth')

        else:
            self.model = init_segmentor('../Swin-Transformer-Semantic-Segmentation/configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k.py', 
                               'pretrained-models/upernet_swin_tiny_patch4_window7_512x512.pth')

       
        self.model.eval()
        self.model.requires_grad = False

        self.image_mean = torch.tensor([0.485, 0.456, 0.406])
        self.image_std = torch.tensor([0.229, 0.22399999999999998, 0.225])
        self.pool  = nn.AdaptiveAvgPool2d((1,1))
        
    def __call__(self, image , return_features=False):
        image = F.interpolate(image*0.5+0.5, size=(256,256), mode='area')
        image -= self.image_mean[:, None, None].to(image.device)
        image /= self.image_std[:, None, None].to(image.device)

        if 'pool' in self.cv_type:
            seg_logit = self.model.backbone(image)[-1]
            seg_logit = self.pool(seg_logit).reshape(-1,768)
        else:
            seg_logit = self.model.backbone(image)[-1]
                
        return seg_logit 
   