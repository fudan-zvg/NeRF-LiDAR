from ctypes import create_unicode_buffer
from xml.dom.domreg import registered
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from data_loading import BasicDataset, RayDropDataset
from unet import UNet
from tqdm import tqdm
import numpy as np
import os
import torch
import torch.nn.functional as F
import yaml
from torchvision import transforms as transforms
from .VGG import VGGLoss
from .darknet import FeatureLoss
class ray_drop_learning:
    def __init__(self,data_depends=None,
                        n_channels=6,batch_size=1,val_percent=0.2,epoch_num = 100,
                        bilinear = True,regression = False,transform = True,early_stop=True,
                        mask_loss=True,delta = False,vgg=False,vgg_weights=0.5,roll = False,
                        device=None,proj_xyz=False,feature_loss = False,feature_loss_weights = 0.5):
        # 1. Create dataset
        # if not proj_xyz:
        #     imgs,masks,ranges = data_depends
        # else:
        #     imgs,masks,ranges,proj_points,gt_proj_points = data_depends
        if transform:
            train_transform=transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(15),
                        # transforms.Resize(image_size),
                        # transforms.CenterCrop(image_size),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
        else:
            train_transform = None
        if data_depends is not None:
            # import pdb;pdb.set_trace()
            dataset = RayDropDataset(data_depends,transforms = train_transform,scale=1.0,proj = proj_xyz)
            self.dataset = dataset
            print('Create dataset')
            # 2. Split into train / validation partitions
            n_val = int(len(dataset) * val_percent)
            n_train = len(dataset) - n_val
            self.train_set, self.val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
            loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
            self.dataloader = DataLoader(self.train_set, shuffle=True, **loader_args)
            self.val_loader = DataLoader(self.val_set, shuffle=False, drop_last=True, **loader_args)
        self.epochs = epoch_num
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UNet(n_channels=n_channels,n_classes=2,bilinear=bilinear,regression=regression).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters())
        self.regression = regression
        self.early_stop = early_stop
        self.mask_loss = mask_loss
        self.delta = delta
        # self.weights = 0.5 if self.regression else 1.
        self.weights = 1.
        self.vgg_weights = vgg_weights 
        self.vgg = vgg
        self.vgg_loss =VGGLoss(device = self.device)
        if feature_loss:
            self.feature_loss =FeatureLoss(self.device)
        else:
            self.feature_loss = None
        self.feature_loss_weights = feature_loss_weights
        self.roll = roll
        # ARCH = yaml.safe_load(open('/SSD_DISK/users/zhangjunge/lidar-bonnetal/train/tasks/semantic/config/arch/darknet53-1024px.yaml', 'r'))
        # self.backbone  = Backbone(params=ARCH["backbone"])
    def train(self,savepath = './'):
        print('start training')
        current_loss = 10;best_loss = 10
        for epoch in tqdm(range(self.epochs+1)):
            Loss = []
            Feature_loss = []
            for batch in self.dataloader:
                img = batch['image'].permute(0,3,1,2)
                gt_mask = batch['mask']
                gt_range = batch['range']
                img = img.to(self.device)  # [N, 1, H, W]
                gt_mask = gt_mask.to(self.device)  # [N, H, W] with class indices (0, 1)
                gt_range = gt_range.to(self.device)
                if self.roll:
                    displacement = int(torch.randint(0,1024,(1,)))
                    img = img.roll(displacement, dims = 3)
                    gt_mask = gt_mask.roll(displacement, dims = 2)
                    gt_range = gt_range.roll(displacement, dims = 2)

                loss=0
                if not self.regression:
                    prediction = self.model(img)  # [N, 2, H, W]
                else:
                    prediction,pred_range = self.model(img)
                    # loss += F.cross_entropy(prediction, gt_mask)
                    if not self.delta:
                        diff = (pred_range - gt_range)[gt_mask.unsqueeze(1)==1].abs()
                    else:
                        # pred_range is a delta correction for depth
                        diff = (torch.clamp(2*pred_range-1+ img[:,0,...],min=0.,max=1.)- gt_range)[gt_mask.unsqueeze(1)==1].abs()
                    loss += diff.mean()
                if self.mask_loss:
                    loss += F.cross_entropy(prediction, gt_mask)*self.weights
                mask = torch.nn.functional.gumbel_softmax(prediction,dim=1,hard=True)
                if self.vgg:
                    # get the channel of depth range 
                    if not self.delta:
                        vgg_loss = self.vgg_loss(img[:,0,...]*mask[:,1,...],gt_range).mean()
                    else:
                        vgg_loss = self.vgg_loss(torch.clamp(2*pred_range-1+ img[:,0,...],min=0.,max=1.)*mask[:,1,...],gt_range,delta=1).mean()

                    loss+= vgg_loss * self.vgg_weights
                    Feature_loss.append(vgg_loss)
                if self.feature_loss:
                    proj_points = batch['proj_points'].to(self.device).permute(0,3,1,2)
                    gt_proj_points = batch['gt_proj_points'].to(self.device).permute(0,3,1,2)
                    feature_loss = self.feature_loss(img[:,0,...],proj_points, mask[:,1,...],gt_range,gt_proj_points)
                    loss += feature_loss * self.feature_loss_weights
                
                    Feature_loss.append(feature_loss)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                Loss.append(loss)
                
            # if epoch % 1 ==0:
            
            print('Epoch %s:  Loss on training set: %s Feature Loss: %s' % (epoch,torch.tensor(Loss).mean().item(),torch.tensor(Feature_loss).mean().item()))
            if epoch%10 == 0 and epoch>0:
                self.model.eval()
                with torch.no_grad():
                    Loss=[]
                    for batch in self.val_loader:
                        loss= 0
                        img = batch['image'].permute(0,3,1,2).to(self.device)  # [N, 1, H, W]
                        gt_mask = batch['mask'].to(self.device)  # [N, H, W] with class indices (0, 1)
                        gt_range =batch['range'].to(self.device) # 
                        if not self.regression:
                            prediction = self.model(img)  # [N, 2, H, W]
                        else:
                            prediction,pred_range = self.model(img)
                        #     if not self.delta:
                        #         diff = (pred_range - gt_range)[gt_mask.unsqueeze(1)==1].abs()
                        #     else:
                        #         # pred_range is a delta correction for depth
                        #         diff = (2*pred_range-1+ img[:,0,...]- gt_range)[gt_mask.unsqueeze(1)==1].abs()
                        #     loss = diff.mean()
                        # if self.mask_loss:
                        loss +=F.cross_entropy(prediction, gt_mask)*self.weights
                        Loss.append(loss)
                    current_loss = torch.tensor(Loss).mean()
                print('epoch_num:',epoch,torch.tensor(Loss).mean())
                if self.early_stop:
                    if current_loss < best_loss:
                        best_loss = current_loss
                    else:
                        break
                    # continue
            if epoch % 10 == 0 and epoch>0:
                #     break
                torch.save(self.model.state_dict(), os.path.join(savepath,'{:05d}.pth'.format(epoch)))
        # return model
    def return_gt(self,index,val=True):
        # if val:
        #     return self.val_loader[index]
        # else:
        #     return self.dataloader[index]
        return self.dataset[index]
    def predict(self,index):
        ########################################################### Generate prediction masks
        # model = UNet(n_channels=n_channels,n_classes=2,bilinear=True).to(device)
        # if filename is not None:
        #     self.load_ckpt(filename=filename)
        # else:
        #     self.load_ckpt('checkpoint_epoch{}.pth'.format(40))
        img =self.val_set[index]['image'][None,...].permute(0,3,1,2).to(self.device).float()  # [N, 1, H, W]
        gt_mask=self.val_set[index]['mask'][None,...].to(self.device)  # [N, H, W] with class indices (0, 1)
        if not self.regression:
            prediction = self.model(img)  # [N, 2, H, W]
            pred_range = torch.tensor(0)
        else:
            prediction,pred_range = self.model(img)
        
        # predict_mask = F.softmax(prediction[0],dim=0)
        predict_mask = prediction[0]

        predict_mask = predict_mask.detach().cpu().numpy()
        gt_mask = gt_mask.detach().cpu().numpy()
        pred_range = pred_range.detach().cpu().numpy()
        # np.save('./probability_mask.npy',predict_mask)
        # np.save('./gt_mask.npy',gt_mask)
        # # if not self.regression:
        # #     return predict_mask,gt_mask
        # # else:
        # np.save('./pred_range.npy',pred_range)
        return predict_mask,gt_mask,pred_range

    def load_ckpt(self, filename):
        self.model.load_state_dict(torch.load(filename))

    def test(self,images,filename=None):
        ########################################################### Generate prediction masks
        # model = UNet(n_channels=n_channels,n_classes=2,bilinear=True).to(device)
        # if filename is not None:
        #     self.load_ckpt(filename=filename)
        # else:
        #     self.load_ckpt('checkpoint_epoch{}.pth'.format(40))
        img =torch.tensor(images).permute(0,3,1,2).to(self.device).float()  # [N, 1, H, W]

        if not self.regression:
            pred_range = torch.tensor(0)
            prediction = self.model(img)  # [N, 2, H, W]
        else:
            prediction,pred_range = self.model(img)
        pred_range = pred_range.detach().cpu().numpy()
        predict_mask =prediction[0].detach().cpu().numpy()

        # if self.regression:
        return predict_mask,pred_range



    
   


if __name__ == '__main__':
    ray_drop_learning()