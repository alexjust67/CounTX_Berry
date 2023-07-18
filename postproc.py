import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import torch
from scipy import ndimage
import cv2
from PIL import Image

#apply a treshold to the density map.
def postprocess(density_map,tresh):

    #density_map = density_map/torch.max(density_map)
    density_map[density_map<=tresh] =0

    return density_map

#show the image with the density map and the clusters.
def showimagefun(img,density_map,clslst,deh,dew,ground_truth,showout=True,textadd=""):
   
    a=clslst
    fig,ax = plt.subplots(1,3,sharex=True,sharey=True)
    ax[0].imshow(img,extent=(0,density_map.shape[1],density_map.shape[0],0))
    
    #add a colorbar to the second and third axes.

    ax[1].imshow(img,extent=(0,dew,deh,0))
    a=np.clip(a,0,1)
    im=ax[1].imshow(a, cmap='jet', interpolation='nearest',alpha=0.85)
    
    plt.colorbar(im,ax=ax[1],fraction=0.036, pad=0.04)
    
    im=ax[2].imshow(density_map, cmap='jet', interpolation='nearest',alpha=1)
    plt.colorbar(im,ax=ax[2],fraction=0.036, pad=0.04)
    
    plt.title("Pred: " + str(np.max(clslst)) + "G-T: "+ground_truth+" "+str(textadd))

    if showout: plt.show()
    plt.close('all')


def clustercount(density_map, treshold, mxlen=17):
    density_map = postprocess(density_map, treshold)
    density_map = ndimage.measurements.label(density_map)[0]
    density_map = set_to_zero(density_map, mxlen)
    density_map = ndimage.measurements.label(density_map)[0]
    density_map = set_to_zero(density_map, mxlen)
    maxval = np.max(density_map)
    return maxval,density_map

def set_to_zero(arr, x):
    unique_values, counts = np.unique(arr, return_counts=True)
    values_to_zero = unique_values[counts > x]
    
    for value in values_to_zero:
        arr[arr == value] = 0
    
    return arr

def function2(x,b):
    
    bx=(255/2)+b*(255/2)
    by=(255/2)-b*(255/2)
    x = np.where(x<bx, x*(by/bx), ((255-by)/(255-bx))*(x-bx)+by)

    return x

def normalize(img,mean,show=False):
    
    img=np.array(img,dtype=np.uint8)
    if show: imgorig=img.copy()
    #calculate the mean of the graymap
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist1=cv2.calcHist([gray], [0], None, [256], [0, 256])
    # Convert the histogram numpy array to a 1D array (if needed)
    histogram1 = hist1.flatten()
    # Calculate the mean
    mean1 = np.average(np.arange(256), weights=histogram1)
    mean1c=mean1
    #calculate the ratio
    ratio=mean/mean1

    b=(1-(ratio+(1-ratio)*0.5))*2.2

    #normalize the image
    img=function2(img,b)
    img=np.array(img,dtype=np.uint8)
    

    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist1=cv2.calcHist([gray1], [0], None, [256], [0, 256])
    histogram1 = hist1.flatten()
    mean1 = np.average(np.arange(256), weights=histogram1)

    if show:
        fig, ax = plt.subplots(2,2)
        ax[1,0].imshow(img)
        ax[0,0].imshow(imgorig)
        ax[0,1].plot(cv2.calcHist([gray], [0], None, [256], [0, 256]))
        ax[0,1].set_title('Mean: '+str(round(mean1,2)))
        ax[0,1].axvline(mean1c, color='r', linestyle='dashed', linewidth=1)
        ax[0,1].axvline(mean, color='g', linestyle='dashed', linewidth=1)
        ax[1,1].plot([function2(x,b)for x in range(256)])

        plt.show()

    return Image.fromarray(img)

