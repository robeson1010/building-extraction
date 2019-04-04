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
        
    def forward(self, input,target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        loss = loss
        
        return loss.mean()

class SemsegLossWeighted(nn.Module):
    def __init__(self,
                 use_running_mean=False,
                 bce_weight=0.5,
                 dice_weight=5.0,
                 eps=1e-10,
                 gamma=0.9,
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
            bce_loss = self.focal(outputs,targets)            
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

from models.LinkNet import LinkNet152
model=LinkNet152(num_classes=1,pretrained=True)

# learn = unet_learner(data, models.resnet18,bottle=True,metrics=[dice,IoU],callback_fns=[BnFreeze,LossMetrics]).to_fp16()
learn=Learner(data,model,loss_func=SemsegLossWeighted(),metrics=[dice,IoU],callback_fns=[BnFreeze,LossMetrics,CSVLogger]).to_fp16() 
learn=learn.to_distributed(arg.local_rank)
learn.load('model_4')
learn.model.unfreeze()
lr=1e-6
learn.fit_one_cycle(10, slice(lr),callbacks=[SaveModelCallback(learn, every='epoch', monitor='accuracy', name='models2')])