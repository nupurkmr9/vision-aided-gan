#code from https://github.com/boukhayma/face_normals
import copy
import os
import gdown
import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import resnet18

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class ResNetUNet(nn.Module):
    def __init__(self, n_class):
        super(ResNetUNet,self).__init__()

        self.base_model = resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 256 + 512, 1, 0)
              
        self.layer0_2 = copy.deepcopy(self.layer0)
        self.layer1_2 = copy.deepcopy(self.layer1)
        self.layer2_2 = copy.deepcopy(self.layer2)
        self.layer3_2 = copy.deepcopy(self.layer3)
        self.layer4_2 = copy.deepcopy(self.layer4)                


        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 128 + 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512,  64 + 256, 3, 1)
        self.conv_up1 = convrelu( 64 + 256,  64 + 256, 3, 1)
        self.conv_up0 = convrelu( 64 + 256,  64 + 128, 3, 1)
        
        self.conv_up3_2 = convrelu(512, 512, 3, 1)
        self.conv_up2_2 = convrelu(512, 256, 3, 1)
        self.conv_up1_2 = convrelu(256, 256, 3, 1)
        self.conv_up0_2 = convrelu(256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)
        
        self.conv_original_size0_2 = convrelu(3, 64, 3, 1)
        self.conv_original_size1_2 = convrelu(64, 64, 3, 1)
        self.conv_original_size2_2 = convrelu(128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)
        
        self.conv_last_2 = nn.Conv2d(64, n_class, 1)

    def forward(self, input, flag=0, get_feat=False):
    
        if (flag == 0): #'im-input'
        
          # Image Encoder
          x_original = self.conv_original_size0(input)
          x_original = self.conv_original_size1(x_original)

          layer0 = self.layer0(input)
          layer1 = self.layer1(layer0)
          layer2 = self.layer2(layer1)
          layer3 = self.layer3(layer2)
          layer4 = self.layer4(layer3)
        
          
          if get_feat:
              return layer4
            
          # Normal decoder
          layer4_1 = self.layer4_1x1(layer4)                 
          x_1 = self.upsample(layer4_1)
          layer3_1 = self.layer3_1x1(layer3)
          x_1 = torch.cat([x_1[:,:512,:,:], torch.max(x_1[:,512:,:,:] , layer3_1)], dim=1)           
          x_1 = self.conv_up3(x_1)
        

          x_1 = self.upsample(x_1)
          layer2_1 = self.layer2_1x1(layer2)
          x_1 = torch.cat([x_1[:,:512,:,:], torch.max(x_1[:,512:,:,:] , layer2_1)], dim=1)
          x_1 = self.conv_up2(x_1)

          x_1 = self.upsample(x_1)
          layer1_1 = self.layer1_1x1(layer1)
          x_1 = torch.cat([x_1[:,:256,:,:], torch.max(x_1[:,256:,:,:] , layer1_1)], dim=1)
          x_1 = self.conv_up1(x_1)

          x_1 = self.upsample(x_1)
          layer0_1 = self.layer0_1x1(layer0)
          x_1 = torch.cat([x_1[:,:256,:,:], torch.max(x_1[:,256:,:,:] , layer0_1)], dim=1)
          x_1 = self.conv_up0(x_1)

          x_1 = self.upsample(x_1)
          x_1 = torch.cat([x_1[:,:128,:,:], torch.max(x_1[:,128:,:,:] , x_original)], dim=1)          
          x_1 = self.conv_original_size2(x_1)

          out_1 = self.conv_last(x_1)
        
          return out_1
          
             
          # Image decoder
          x_2 = self.upsample_2(layer4)  
          x_2 = self.conv_up3_2(x_2)

          x_2 = self.upsample_2(x_2)
          x_2 = self.conv_up2_2(x_2)

          x_2 = self.upsample_2(x_2)
          x_2 = self.conv_up1_2(x_2)

          x_2 = self.upsample_2(x_2)
          x_2 = self.conv_up0_2(x_2)

          x_2 = self.upsample_2(x_2)
          x_2 = self.conv_original_size2_2(x_2)

          out_2 = self.conv_last_2(x_2)
                  
          return out_1, out_2 
                    
          
        if (flag == 1): #'norm-input'
        
          # Normal Encoder
          x_original = self.conv_original_size0_2(input)
          x_original = self.conv_original_size1_2(x_original)

          layer0 = self.layer0_2(input)
          layer1 = self.layer1_2(layer0)
          layer2 = self.layer2_2(layer1)
          layer3 = self.layer3_2(layer2)
          layer4 = self.layer4_2(layer3)
          
          # Normal decoder
          layer4 = self.layer4_1x1(layer4)                 
          
          x_1 = self.upsample(layer4)  
          x_1 = self.conv_up3(x_1)

          x_1 = self.upsample(x_1)
          x_1 = self.conv_up2(x_1)

          x_1 = self.upsample(x_1)
          x_1 = self.conv_up1(x_1)

          x_1 = self.upsample(x_1)
          x_1 = self.conv_up0(x_1)

          x_1 = self.upsample(x_1)
          x_1 = self.conv_original_size2(x_1)

          out_1 = self.conv_last(x_1)
          
          return out_1
              
class Normals(torch.nn.Module):

    def __init__(self, cv_type='adv'):
        super().__init__(
        )
        
        self.cv_type = cv_type

        self.model = ResNetUNet(n_class = 3)
        if not os.path.exists(os.path.join(os.environ['HOME'], '.cache', 'normal_model.pth')):
            gdown.download("https://drive.google.com/u/0/uc?id=1Qb7CZbM13Zpksa30ywjXEEHHDcVWHju_&export=download", os.path.join(os.environ['HOME'], '.cache', 'normal_model.pth'))

        self.model.load_state_dict(torch.load(os.path.join(os.environ['HOME'], '.cache', 'normal_model.pth'), map_location='cpu'))

        self.model.eval() 
        self.model.requires_grad = False
        
         
    def __call__(self, image):
      image = F.interpolate(image, size=(256,256), mode='area') 
      if 'conv' in self.cv_type:
        outs = self.model(image*0.5+0.5,get_feat=True) 
        return outs
      elif 'image' in self.cv_type:
        outs = self.model(image*0.5+0.5) 
        l2_norm1 = torch.clamp(torch.norm(outs, dim=1, keepdim=True), min=1e-5)
        outs = outs / l2_norm1
        return outs
      elif 'pool' in self.cv_type:
        outs = self.model(image*0.5+0.5,get_feat=True)
        return F.avg_pool2d(outs , 8).squeeze()
