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
from postproc import clustercount,postprocess,showimagefun
import matplotlib.patches as patches
from kmeans_pytorch import kmeans
from noise import noise_map_creator
import copy

def mainf(
        model,
        image_file_name = "./img/bacchetree.png",
        image= None,
        text = "the number of berries",
        sqsz=224,
        dm_save=True,
        showimage=True,
        recl=60,
        tresh=0.4,
        text_add="",
        ground_truth="",
        mxlen=20,
        no_stride=True,
        stride=[50,50],
        device="cpu",
        clusteralg="rec-find",
        iterations=0,
        noise_lvl=1,
        noise_type="gauss",
        dropoff=30,
        density_map_coeff=2,
        shownoise=False,
        rad=60,
        ):
    
    for i1 in range(iterations):
        # Define preprocessing.
        tokenizer = open_clip.get_tokenizer("ViT-B-16")
        # Tokenize the text.
        enc_txt=tokenizer(text[i1])
        
        density_map,dew,deh,stridex,stridey=m1.density_map_creator(image,model,text_add,i1,iterations,enc_txt,dm_save,device=device,sqsz=sqsz,stride=stride,no_stride=no_stride)
        
        print("Calculating clusters...   ")
            
        strt=time.time()
            
        #calculate the clusters using the chosen algorithm
        iclsc,clslst=clustercount(density_map.numpy(),tresh=tresh,tresh2=tresh,recl=recl,mxlen=mxlen,algo=clusteralg)

        print("Done calculating clusters. Time: ",round(time.time()-strt,2),"s")

        if iterations!=0: 
            print("Starting map modification...")
                
            if noise_type!="None":    
                noisetime=time.time()
                
                #calculate the noise map
                noisemap=noise_map_creator([deh,dew],noise_lvl,noise_type,density_map,clslst,rad,dropoff=dropoff,density_map_coeff=density_map_coeff,shownoise=shownoise)

                image=np.array(image)

                nos=Image.fromarray(noisemap).convert("L")
                nos=nos.resize((image.shape[1],image.shape[0]))
                noisemap=np.array(nos)
                #convert to int array
                noisemap=noisemap.astype(np.float64)
                if noise_type=="equal": 
                    noisemap/=np.full(noisemap.shape,255)
                    #positive incentive
                    noisemap+=np.full(noisemap.shape,0.999)
                #add two new axis to the noise map
                noisemap = np.repeat(noisemap[:, :, np.newaxis], 3, axis=2)
                #add the noise map to the image
                image=(image.astype(np.float64)/noisemap).astype(np.uint8)

                if (False):
                    plt.imshow(image)
                    #plt.imshow(noisemap)
                    plt.show()
                image=Image.fromarray(image.astype(np.uint8))
            


            np.save(f"./img/results/density_map{i1}.npy",density_map.numpy())
            try:
                print("Done map modification. Time: ",round(time.time()-noisetime,2),"s")
            except:
                continue



        #show the image with the clusters and the density map if needed
        if showimage and iterations==0: 
            showimagefun(density_map.numpy(),image_file_name,clslst,deh,dew,ground_truth,tresh=tresh)
            
    
    if iterations!=0:
        #create a np array with the density maps of all the noise iterations from ./img/results/density_map{i}.npy
        denmaps=np.zeros((0,deh,dew))
        #calculate the average of all the density maps

        for i in range(iterations):
            denmaps=np.append(denmaps,np.expand_dims(np.load(f"./img/results/density_map{i}.npy"),0),axis=0)
        
        #denmaps=np.mean(denmaps,axis=0)
        denmaps=np.sum(denmaps,axis=0)
        #calculate the clusters using the chosen algorithm
        iclsc,clslst=clustercount(denmaps,tresh=tresh+iterations/5,tresh2=tresh,recl=recl,mxlen=mxlen,algo=clusteralg)
        if showimage:
            showimagefun(denmaps,image_file_name,clslst,deh,dew,ground_truth,tresh=tresh+iterations/5,)

    return stridex,stridey, clslst, tresh, recl, sqsz


