from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
import torch
from torch import nn
import numpy as np
#The fidelity term
class Fidelity_Loss(nn.Module):
    def __init__(self):
        super(Fidelity_Loss, self).__init__()
        return

    def forward(self, preds, labels):
        length=len(preds)
        H,W=preds.shape[-2:]    
        normalization=3*length*H*W
        if(len(preds.shape)==5):
            normalization*=preds.shape[2]
        losses=[torch.linalg.norm(pred-label) for pred,label in zip(preds,labels)]
        return 1/normalization*sum(losses)**2

class Total_Loss(nn.Module):
    def __init__(self):
        super(Total_Loss, self).__init__()
        self.pearson=Neg_Pearson()
        self.fidelity=Fidelity_Loss()
        return
        
    def forward(self,signal_preds,signal_labels,video_preds,video_labels,lam=0.5):
        return 1-torch.corrcoef(torch.permute(torch.cat((signal_preds,signal_labels.unsqueeze(1)),dim=1),(1,0)))[0][1],lam*self.fidelity(video_preds,video_labels)


