from fastai.distributed import *
import argparse
import numpy as np
import os
import pickle
from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *
from fastai.callbacks import LossMetrics,SaveModelCallback,CSVLogger
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
arg = parser.parse_args()
# print(args)
torch.cuda.set_device(arg.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')
device = torch.device("cuda:{}".format(arg.local_rank))
path=Path('/home/staff/xin/Downloads/building')
df=pd.read_csv('building.csv')
def get_y_fn(x): return x.replace('jpg','png')
codes = array(['Building'])

size = (288,288)
free = gpu_mem_get_free_no_cache()
bs=16

src=(SegmentationItemList.from_df(df,path).split_from_df(1).label_from_func(get_y_fn, classes=codes))
data = (src.transform(get_transforms(), size=size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input,target,weight):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
#         print(weight.dtype)
#         print(loss.dtype)
        loss = loss*weight       
        return loss.mean()

class SemsegLossWeighted(nn.Module):
    def __init__(self,
                 use_running_mean=False,
                 bce_weight=5.0,
                 dice_weight=0.5, #0.5
                 eps=1e-10,
                 gamma=1.8,#0.8
                 use_weight_mask=True,
                 deduct_intersection=False
                 ):
        super().__init__()

        self.use_weight_mask = use_weight_mask
        
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.eps = eps
        self.focal = FocalLoss(2.0)      
        self.use_running_mean = use_running_mean
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.deduct_intersection = deduct_intersection
        self.metric_names = ['bce','dice']
        self.focal = FocalLoss(gamma)
        
        if self.use_running_mean == True:
            self.register_buffer('running_bce_loss', torch.zeros(1))
            self.register_buffer('running_dice_loss', torch.zeros(1))
            self.reset_parameters()

    def reset_parameters(self):
        self.running_bce_loss.zero_()        
        self.running_dice_loss.zero_()            

    def forward(self,
                outputs,
                targets):
        # inputs and targets are assumed to be BxCxWxH
#         outputs=outputs[:,1,:,:].unsqueeze(1)
        targets=targets.float()
#         weights=targets[:,1,:,:].unsqueeze(1)
#         targets=targets[:,0,:,:].unsqueeze(1)
        assert len(outputs.shape) == len(targets.shape)
        assert outputs.shape == targets.shape
        if self.use_weight_mask:
            weight=get_weights(targets)
            bce_loss = self.focal(outputs,targets,weight)            
        else:
            bce_loss = self.nll_loss(input=outputs,
                                     target=targets)

        dice_target = (targets == 1).float()
        dice_output = torch.sigmoid(outputs)
        dice_target = dice_target.view(-1)
        dice_output = dice_output.view(-1)
        intersection = (dice_output * dice_target).sum()
        if self.deduct_intersection:
            union = dice_output.sum() + dice_target.sum() - intersection + self.eps
        else:
            union = dice_output.sum() + dice_target.sum() + 1.0
            
        dice_loss = (-torch.log((2 * intersection+1.0) / union))         
        
        if self.use_running_mean == False:
            bmw = self.bce_weight
            dmw = self.dice_weight
            # loss += torch.clamp(1 - torch.log(2 * intersection / union),0,100)  * self.dice_weight
        else:
            self.running_bce_loss = self.running_bce_loss * self.gamma + bce_loss.data * (1 - self.gamma)        
            self.running_dice_loss = self.running_dice_loss * self.gamma + dice_loss.data * (1 - self.gamma)

            bm = float(self.running_bce_loss)
            dm = float(self.running_dice_loss)

            bmw = 1 - bm / (bm + dm)
            dmw = 1 - dm / (bm + dm)
                
        loss = bce_loss * bmw + dice_loss * dmw
        
        outloss=[bce_loss,dice_loss]
        
        self.metrics=dict(zip(self.metric_names,outloss))
        
        return loss
#     ,bce_loss,dice_loss  

def get_weights(target):
    weights=[]
    for i in range(len(target)):
        weight=torch.from_numpy(get_weight(target[i]))
        weights.append(weight)
    weights=torch.stack(weights)
    weights = weights.to(device).float()
    return weights
def get_weight(target):
    size_weights = get_size_weights(target)
    distance_weights = distance_mask(target)
    weights = distance_weights * size_weights
    return weights

from skimage.measure import label
from scipy.ndimage.morphology import distance_transform_edt

def get_size_weights(mask):
    C = np.sqrt(mask.shape[0] * mask.shape[1]) / 2.0
    sizes = np.ones_like(mask)
    labeled = label(mask)
    for label_nr in range(1, labeled.max() + 1):
        label_size = (labeled == label_nr).sum()
        sizes = np.where(labeled == label_nr, label_size, sizes)        
    sizes_ = sizes.copy()
    sizes_[sizes == 0] = 1
    size_weights = C / sizes_
    size_weights[sizes_ == 1] = 1           
    return size_weights.astype('float32')  
def distance_mask(mask):
    d = distance_transform_edt(1 - mask)
    weights = np.ones_like(mask) + 50.0 * np.exp(-(np.power(d,2)) / (10.0 ** 2))
    weights[d == 0] = 1
    return weights

def dice(input, target):
    input = torch.sigmoid(input)    
    input = (input>0.5).float()
    target=target.float()
    return 2.0 * (input*target).sum() / ((input+target).sum() + 1.0)

def IoU(input, target):
    input = torch.sigmoid(input)    
    input = (input>0.5).float()
    target=target.float()
    intersection = (input*target).sum()
    return intersection / ((input+target).sum() - intersection + 1.0)

import lovasz_losses as L
class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, outputs, targets):
        outputs = torch.sigmoid(outputs)
        outputs = outputs.squeeze(1)
        targets = targets.squeeze(1)
        loss = L.lovasz_softmax(outputs,targets, classes=[1], ignore=255)
        return loss

from models.LinkNet import LinkNet152
model=LinkNet152(num_classes=1,pretrained=True)

# learn = unet_learner(data, models.resnet18,bottle=True,metrics=[dice,IoU],callback_fns=[BnFreeze,LossMetrics]).to_fp16()
# learn=Learner(data,model,loss_func=SemsegLossWeighted(),metrics=[dice,IoU],callback_fns=[BnFreeze,LossMetrics,CSVLogger]).to_fp16() 
learn=Learner(data,model,loss_func=SemsegLossWeighted(),metrics=[dice,IoU],callback_fns=[BnFreeze,CSVLogger]).to_fp16() 
learn=learn.to_distributed(arg.local_rank)
learn.load('models2_9')
learn.loss_func=LovaszHingeLoss()
learn.model.unfreeze()
lr=2e-5
learn.fit_one_cycle(2, slice(lr),callbacks=[SaveModelCallback(learn, every='epoch', monitor='accuracy', name='floss')])