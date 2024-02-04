import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models

class VGGLoss(nn.Module):
    def __init__(self, device):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().to(device)
        self.device = device
        self.criterion = nn.L1Loss(reduction='none')
        self.weights = [1.0/16, 1.0/8, 1.0/4, 1.0]#[1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]     

    def forward(self, x, y,delta=0):            
        # x = x.permute(2,0,1).float()
        # y = y.permute(2,0,1).float()
        # x shape: N,C,H,W
        H,W=x.shape[-2:]
        if not delta:
            x = x.unsqueeze(1).broadcast_to(x.shape[0],3,H,W)
            y = y.unsqueeze(1).broadcast_to(x.shape[0],3,H,W)
        else:
            x = x.broadcast_to(x.shape[0],3,H,W)
            y = y.broadcast_to(x.shape[0],3,H,W)         
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = torch.zeros((x.shape[0],H,W)).to(self.device)
        for i in range(len(x_vgg)-1):
            feat_x, feat_y = x_vgg[i], y_vgg[i].detach()
            if(i > 0):
                feat_x = F.upsample(x_vgg[i], mode='bilinear', size=(H,W), align_corners=True)
                feat_y = F.upsample(y_vgg[i], mode='bilinear', size=(H,W), align_corners=True)
                feat_x = feat_x.squeeze(0)
                feat_y = feat_y.squeeze(0)

            f_loss = self.weights[i] * self.criterion(feat_x, feat_y)
            loss += f_loss.mean(0).mean(0)
        return loss
        
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(21, 30):
        #     self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        #h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4] #, h_relu5]
        return out