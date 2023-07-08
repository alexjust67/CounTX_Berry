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
from postproc import clustercount,postprocess
import matplotlib.patches as patches
from kmeans_pytorch import kmeans


def mainf(
        model,
        image_file_name = "./img/bacchetree.png",
        text = "the number of berries",
        enc_txt = None,
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
        ):

    
    # Load and process the image.
    image = Image.open(image_file_name).convert("RGB")
    image.load()

    # Resize and center crop the image into sqsz shapes.
    if no_stride: 
        w1,h1,image=m1.split_image(image,sqsz)
        deh=int((h1/sqsz)*384)  #actual size of the final image
        dew=int((w1/sqsz)*384)
    else:
        dew,deh=image.size
        deh=math.floor((deh/sqsz)*384)  #actual size of the final image
        dew=math.floor((dew/sqsz)*384)
        stridex,stridey,w1,h1,image,coord=m1.split_image_stride(image,sqsz,stride=stride)
        
    
    w=0
    h=0
    density_map=torch.zeros((deh,dew))    #create a zero tensor with the dimension of the final image that will be no of squares * 384(standard model output)
    tot=0
    time1=[]
    time2=[]
    dens_time=time.time()

    for i in image:                       #loop through the images
        
        if no_stride: 
            density_map[h:h+384,w:w+384]=m1.runmodel(i,enc_txt,model,device=device).to(torch.device("cpu"))

        else:                               #if stride is used it will loop through the coordinates of the squares and run the model on them and then put the output in the right place in the final image using a max function between the overlapping squares
             
            inx=math.floor(coord[tot][0]*(384-(stridex/sqsz)*384))          #transform the coordinate indexes in the 384 coord space
            iny=math.floor(coord[tot][1]*(384-(stridey/sqsz)*384))
            if inx>dew-384: inx=dew-384
            if iny>deh-384: iny=deh-384
            denmap=m1.runmodel(i,enc_txt,model,device=device).to(torch.device("cpu"))
            density_map[iny:iny+384,inx:inx+384]=torch.max(density_map[iny:iny+384,inx:inx+384],denmap)      

        if (h+384!=int((h1/sqsz)*384)and no_stride) or ((not no_stride) and h//384!=coord[len(coord)-1][1]):        #check if it has arrived to the bottom (only used by no_stride)
            
            h+=384
            
            time2.insert(0,time.time())                             #calculate the time remaining
            try:
                time1.insert(0,time2[0]-time2[1])
                time2.pop(2)
                try:
                    time2.pop(3)
                except:
                    pass
            except:
                pass
            try:
                avgperpoint=sum(time1[0:10])/len(time1[0:10])
            except:
                avgperpoint=0
            
            if no_stride:                                            #print the progress
                print(h//384," :  "+str(deh//384)+"    ",w//384," :  "+str(dew//384)+"    ",(tot)," : ",image.shape[0],"  ",str(round((tot/image.shape[0])*100,2))+"%"+ "    "+"Time remaining: ",round(avgperpoint*(image.shape[0]-tot),0),"s"+"   ",text_add,"      ",end="\r")
            else:
                print(h//384," : ",coord[len(coord)-1][1],"    ",w//384," : ",coord[len(coord)-1][0],"    ",(tot)," : ",image.shape[0],"  ",str(round((tot/image.shape[0])*100,2))+"%"+ "    "+"Time remaining: ",round(avgperpoint*(image.shape[0]-tot),0),"s"+"   ",text_add,"      ",end="\r")

            tot+=1
        
        else:
            time2.insert(0,time.time())
            tot+=1
            h=0
            w=w+384
    
    print("Done calculating density map in:",round(time.time()-dens_time,2),text_add,"                                       ")
    
    mod_pred_cnt = torch.sum(density_map / 60).item()       #predicted count by the integration of the density map

    if dm_save: np.save("./img/results/density_map.npy",density_map.numpy())        #save the density map if needed

    print("calculating clusters...   ")
    
    strt=time.time()

    #calculate the clusters using the chosen algorithm
    iclsc,clslst=clustercount(density_map.numpy(),tresh=tresh,tresh2=tresh,recl=recl,mxlen=mxlen,algo=clusteralg)
    
    print("Done calculating clusters. Time: ",round(time.time()-strt,2),"s")
    
    #show the image with the clusters and the density map if needed
    if showimage: 
        
        a=postprocess(density_map.numpy(),tresh=tresh)

        img=mpimg.imread(image_file_name)
        fig,ax = plt.subplots(1,3,sharex=True,sharey=True)
        ax[0].imshow(img,extent=(0,density_map.shape[1],density_map.shape[0],0))
        ax[1].imshow(img,extent=(0,dew,deh,0))
        ax[1].imshow(a, cmap='jet', interpolation='nearest',alpha=0.85)
        ax[2].imshow(density_map.numpy(), cmap='jet', interpolation='nearest',alpha=1)

        
        for it in clslst:
            ax[1].add_patch(patches.Rectangle((it[1],it[0]),it[3],it[2],linewidth=1,facecolor='none',edgecolor='red'))

        plt.title("Pred: " + str(len(clslst)) + "G-T: "+ground_truth)
        image_file_name=image_file_name[(image_file_name.rfind("/")+1):]

        plt.savefig(f"./img/results/{image_file_name}",dpi=1500)
        plt.show()
        plt.close('all')
    
    return stridex,stridey,mod_pred_cnt, clslst, tresh, recl, sqsz

