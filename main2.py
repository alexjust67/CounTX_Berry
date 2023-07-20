from functools import partial
import math
import copy
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import open_clip
import matplotlib.pyplot as plt
import numpy as np
from models_vit import CrossAttentionBlock
from util.pos_embed import get_2d_sincos_pos_embed
import matplotlib.image as mpimg
import time
import module1 as m1
from postproc import clustercount,postprocess,showimagefun,normalize
import matplotlib.patches as patches
import copy
import cv2
from scipy import signal

def mainf(
        model,
        image= None,
        query = "",
        sqsz=224,
        dm_save=True,
        showimage=True,
        showkern=False,
        tresh=0.4,
        text_add="",
        ground_truth="",
        mxlen=20,
        stride=[50,50],
        device="cpu",
        shownorm=False,
        norm=110,
        colorfilter=False,
        height=0,
        adap_krnl=False,
):
    
    if adap_krnl:
        sqsz=round(-174*height+1281)


    density_map,dew,deh,stridex,stridey=m1.density_map_creator(image,model,text_add,query,dm_save,device=device,sqsz=sqsz,stride=stride,showkern=showkern,norm=norm,shownorm=shownorm)
    
    dencp=copy.deepcopy(density_map)

    print("Calculating clusters...   ")
    
    strt=time.time()
    numlist=[]
    x=[]
    i=0

    for tre in tresh:
        clsnum,clsmap=clustercount(density_map,tre,image,mxlen=mxlen,colorfilter=colorfilter)
        numlist.append(clsnum)
        i+=1
        if i % 50 == 0: print("Done ",i/len(tresh)*100,"%                   ",end="\r")
    
    print("Done calculating clusters. Time: ",round(time.time()-strt,2),"s")
    
    if showimage:
        print("Showing image...   ")
        showimagefun(image,density_map,clsmap,deh,dew,ground_truth,textadd=text_add,showout=True,height=height)

    
    return stridex,stridey, numlist, tresh, sqsz

