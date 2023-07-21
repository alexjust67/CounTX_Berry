import math
import copy
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
import module1 as m1
from postproc import clustercount,showimagefun

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
        generalize_results=False,
):
    #if the adaptive kernel is used, the size of the kernel is calculated based on the height of the drone.
    if adap_krnl:
        sqsz=round(-174*height+1281)

    #Model module.
    density_map,dew,deh,stridex,stridey=m1.density_map_creator(image,model,text_add,query,dm_save,device=device,sqsz=sqsz,stride=stride,showkern=showkern,norm=norm,shownorm=shownorm)

    print("Calculating clusters...   ")
    
    strt=time.time()

    #clusters module.
    clsnum,clsmap=clustercount(density_map,tresh,image,mxlen=mxlen,colorfilter=colorfilter)

    print("Done calculating clusters. Time: ",round(time.time()-strt,2),"s")
    
    #postprocessing/showing module.
    if showimage:
        print("Showing image...   ")
        showimagefun(image,density_map,clsmap,deh,dew,ground_truth,textadd=text_add,showout=True,height=height,generalize_results=generalize_results)

    #return the stride values, the number of clusters, the treshold value, the size of the kernel.
    return stridex,stridey, clsnum, tresh, sqsz

