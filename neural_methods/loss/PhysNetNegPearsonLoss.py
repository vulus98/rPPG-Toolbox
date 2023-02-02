from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
import argparse, os
import pandas as pd
import numpy as np
import random
import math
from torchvision import transforms
from torch import nn


class Neg_Pearson(nn.Module):
    """
    The Neg_Pearson Module is from the orignal author of Physnet.
    Code of 'Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks' 
    source: https://github.com/ZitongYu/PhysNet/blob/master/NegPearsonLoss.py
    """
    
    def __init__(self):
        super(Neg_Pearson, self).__init__()
        return


    def forward(self, preds, labels):       
        sum_x = torch.sum(preds)               
        sum_y = torch.sum(labels)             
        sum_xy = torch.sum(preds*labels)       
        sum_x2 = torch.sum(torch.pow(preds,2))  
        sum_y2 = torch.sum(torch.pow(labels,2)) 
        N = preds.shape[0]
        pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2))))
        loss = 1 -pearson
        return loss




