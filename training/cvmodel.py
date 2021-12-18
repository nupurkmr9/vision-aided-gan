import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import clip
import antialiased_cnns

from torch_utils import persistence
import dnnlib


@persistence.persistent_class 
class Vgg(nn.Module):
    def __init__(self, cv_type='adv'):
        super().__init__()
        self.cv_type = cv_type
        self.model = antialiased_cnns.vgg16(pretrained=True, filter_size=4).features
        self.model.eval()
        self.model.requires_grad = False
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.image_mean = torch.tensor([0.485, 0.456, 0.406])
        self.image_std = torch.tensor([0.229, 0.224, 0.225])
        
    def forward(self, image):
        image = F.interpolate(image*0.5+0.5, size=(224, 224), mode='area')
        image = image - self.image_mean[:, None, None].to(image.device)
        image /= self.image_std[:, None, None].to(image.device)
        out = self.model(image)

        if 'pool' in self.cv_type:
            out = self.pool(out)
            out = out.view(out.shape[0],-1)
        
        return out

@persistence.persistent_class  
class Swin(torch.nn.Module):

    def __init__(self, cv_type='adv'):
        super().__init__(
        )

        self.cv_type = cv_type

        self.model = timm.create_model('swin_tiny_patch4_window7_224')
        st = torch.load('pretrained-models/moby_swin_t_300ep_pretrained.pth', map_location='cpu')
        new_st = {}
        for each in st['model'].keys():
            if 'encoder.' in each:
                newk = each.replace('encoder.','')
                new_st[newk] = st['model'][each]
        self.model.load_state_dict(new_st, strict=False)

        self.model.eval()
        self.model.requires_grad = False

        self.image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
        self.image_std = torch.tensor([0.229, 0.224, 0.225])      
        self.pool = nn.AdaptiveAvgPool2d((1,1))

    def forward_features_custom_swin(self, x, return_intermediate=False):
        x = self.model.patch_embed(x)
        if self.model.ape:
            x = x + self.model.absolute_pos_embed
        x = self.model.pos_drop(x)
        x = self.model.layers(x)     
        x = self.model.norm(x)
        if return_intermediate:
            return x.transpose(1, 2)
        
        x = self.model.avgpool(x.transpose(1, 2)) 
        x = torch.flatten(x, 1)
        return x

    def __call__(self, image):
        image = F.interpolate(image*0.5+0.5, size=(224,224), mode='area')
        image = image - self.image_mean[:, None, None].to(image.device)
        image /= self.image_std[:, None, None].to(image.device)

        if 'conv' in self.cv_type:
            final_feat = self.forward_features_custom_swin(image, return_intermediate=True)
            final_feat = final_feat.reshape(-1,768,7,7)
            return final_feat
            
        return self.model.forward_features(image)

     
@persistence.persistent_class   
class CLIP(torch.nn.Module):

    def __init__(self, cv_type='adv'):
        super().__init__(
        )

        self.cv_type = cv_type
        self.model, _ = clip.load("ViT-B/32", jit=False)
        self.model = self.model.visual
        self.model.eval()
        self.model.requires_grad = False

        self.image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
        self.image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])
 
    def __call__(self, image):
        image = F.interpolate(image*0.5+0.5, size=(224,224), mode='area')#, align_corners=True) 
        image = image - self.image_mean[:, None, None].to(image.device)
        image /= self.image_std[:, None, None].to(image.device)
            
        if 'conv_multi_level' in self.cv_type:
            image_features = self.model(image.type(self.model.conv1.weight.dtype), return_intermediate=True)
            image_features[0] = image_features[0][:,1:,:].permute(0,2,1).reshape(-1,768,7,7).float()
            image_features[1] = image_features[1][:,1:,:].permute(0,2,1).reshape(-1,768,7,7).float()
            image_features[2] = image_features[2].float()
        else:
            image_features = self.model(image.type(self.model.conv1.weight.dtype)).float()
            
        return image_features
            
  
@persistence.persistent_class   
class DINO(torch.nn.Module):

    def __init__(self, cv_type='adv'):
        super().__init__(
        )

        self.cv_type = cv_type
        self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        self.model.eval()
        self.model.requires_grad = False
        self.input_resolution = 224
        self.image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
        self.image_std = torch.tensor([0.229, 0.224, 0.225])
       
    def forward_custom(self, x, return_intermediate=False):
        # See note [TorchScript super()]
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x1 = self.model.layer2(x)
        x2 = self.model.layer3(x1)
        x3 = self.model.layer4(x2)

        if return_intermediate:
            return [x1, x2, x3]
        return x3
    
    def __call__(self, image):
        image = F.interpolate(image*0.5+0.5, size=(224,224), mode='area')#, align_corners=True) 
        image = image - self.image_mean[:, None, None].to(image.device)
        image /= self.image_std[:, None, None].to(image.device)
        
        if 'conv_multi_level' in self.cv_type:
            image_features = self.model.get_intermediate_layers(image, n=8)
            image_features = [image_features[x] for x in [0,4,-1]]
            image_features[0] = image_features[0][:,1:,:].permute(0,2,1).reshape(-1,768,14,14)
            image_features[1] = image_features[1][:,1:,:].permute(0,2,1).reshape(-1,768,14,14)
            image_features[2] = image_features[2][:,0,:]
        else:
            image_features = self.model(image)
            
        return image_features



@persistence.persistent_class       
class CVWrapper(torch.nn.Module):

    def __init__(self, device, cv_type):
        super().__init__(
        )
        cv_lists = cv_type.split('input-')[1].split('-output-')[0].split('-')
        output = cv_type.split('output-')[1]
        class_name_dict = {
                'seg_ade': 'training.swinTaskSpecific.Seg',
                'det_coco': 'training.swinTaskSpecific.Seg',
                'face_parsing': 'training.face_parsing.Parsing',
                'face_normals': 'training.face_normals.Normals',
                'clip': 'training.cvmodel.CLIP',
                'dino': 'training.cvmodel.DINO',
                'vgg':'training.cvmodel.Vgg',
                'swin':'training.cvmodel.Swin',
            }
    
        self.cv_type = cv_type
        self.models = []
        for each in cv_lists:
            cv_specs = {
                        'cv_type': each + '_' + output
                    }

            cv_kwargs = dnnlib.EasyDict(class_name=class_name_dict[each], **cv_specs)
            model = dnnlib.util.construct_class_by_name(**cv_kwargs).requires_grad_(False).to(device) 
            self.models.append(model)
       
    def __call__(self, image_list):
        if len(self.models) == 1:
            return self.models[0](image_list)
        image_features = []
        for i, each in enumerate(self.models):
            image_features.append(each(image_list[i]))
        return image_features
        



