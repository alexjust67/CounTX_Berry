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
from postproc import clustercount,postprocess,showimagefun,clustercount2
import matplotlib.patches as patches
from noise import noise_map_creator
import copy

def mainf(
        model,
        image_file_name = "./img/bacchetree.png",
        image= None,
        text = "",
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
):
    
    if stride==0:
        no_stride=True
    else:
        no_stride=False
    
    # Define preprocessing.
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    # Tokenize the text.
    enc_txt=tokenizer(text)
    
    density_map,dew,deh,stridex,stridey=m1.density_map_creator(image,model,text_add,enc_txt,dm_save,device=device,sqsz=sqsz,stride=stride,no_stride=no_stride,showkern=showkern)
    
    if dm_save: np.save(f"./img/results/density_map.npy",density_map.numpy())

    print("Calculating clusters...   ")
    
    strt=time.time()
    numlist=[]
    x=[]
    i=0
    for tre in tresh:
        clsnum,clsmap=clustercount2(density_map,tre/100,mxlen=mxlen)
        numlist.append(clsnum)
        i+=1
        if i % 50 == 0: print("Done ",i/len(tresh)*100,"%                   ",end="\r")
    print("Done calculating clusters. Time: ",round(time.time()-strt,2),"s")
    
    if showimage:
        print("Showing image...   ")
        showimagefun(density_map,image_file_name,clsmap,deh,dew,ground_truth)

    return stridex,stridey, numlist, tresh, sqsz


