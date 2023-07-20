import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import torch
from scipy import ndimage
import cv2
from PIL import Image, ImageFilter
from scipy import ndimage

#apply a treshold to the density map.
def postprocess(density_map,tresh):

    #density_map = density_map/torch.max(density_map)
    density_map[density_map<=tresh] =0

    return density_map

def boxfilter(a,shape):
    
    rag=shape//100
    while not chk_prime(rag):
        rag+=1

    a=a.filter(ImageFilter.MaxFilter(rag))
    
    return a

def chk_prime(n):
    if n>1:
        for i in range(2, n//2+1):
            if n%i==0:
                return False
                break
        else:
            return True
    else:
        return False

#show the image with the density map and the clusters.
def showimagefun(img,density_map,clslst,deh,dew,ground_truth,showout=True,textadd=""):
    
    a=np.clip(clslst*255,0,255)
    a=Image.fromarray(a)
    a=a.convert('L')
    a=a.resize((density_map.shape[1],density_map.shape[0]))
    a=boxfilter(a,density_map.shape[1])

    a=a.filter(ImageFilter.GaussianBlur(radius = 70))
    a=np.array(a)
    a_cv2 = 255//np.max(a)*a
    im_color = cv2.applyColorMap(a_cv2, cv2.COLORMAP_JET)
    cv2.imwrite(f"/home/agiustina/CounTX_Berry/img/renders/dronemap/droneres/{textadd}",im_color)

    fig,ax = plt.subplots(1,3,sharex=True,sharey=True)
    ax[0].imshow(img,extent=(0,density_map.shape[1],density_map.shape[0],0))
    
    #add a colorbar to the second and third axes.

    ax[1].imshow(img,extent=(0,dew,deh,0))
    im=ax[1].imshow(a, cmap='jet', interpolation='nearest',alpha=1)
    
    plt.colorbar(im,ax=ax[1],fraction=0.036, pad=0.04)
    
    im=ax[2].imshow(density_map, cmap='jet', interpolation='nearest',alpha=1)
    plt.colorbar(im,ax=ax[2],fraction=0.036, pad=0.04)
    
    plt.title("Pred: " + str(np.max(clslst)) + "G-T: "+ground_truth+" "+str(textadd))

    if showout: plt.show()
    plt.close('all')


def clustercount(density_map, treshold,rgb_image, mxlen=17,colorfilter=False):
    density_map = postprocess(density_map, treshold)
    density_map = ndimage.measurements.label(density_map)[0]
    density_map = set_to_zero(density_map, mxlen)
    density_map = ndimage.measurements.label(density_map)[0]
    density_map = set_to_zero(density_map, mxlen)
    if colorfilter:
        density_map = remove_nonred_clusters(density_map, rgb_image)
        density_map = ndimage.measurements.label(density_map)[0]
    maxval = np.max(density_map)
    return maxval,density_map

def remove_nonred_clusters(density_map, rgb_image):
    #convert to numpy  
    density_map = np.array(density_map)
    rgb_image = np.array(rgb_image)
    
    # Resize the RGB image to match the shape of the density map
    resized_image = cv2.resize(rgb_image, (density_map.shape[1], density_map.shape[0]))

    # Convert the resized image to HSV color space
    resized_image_hsv = cv2.cvtColor(resized_image, cv2.COLOR_RGB2HSV)

    # Thresholds for red color range in HSV
    lower_red = np.array([0, 50, 50])   # Adjust these values as needed
    upper_red = np.array([10, 255, 255])   # Adjust these values as needed

    # Create a binary mask based on the red color range
    red_mask = cv2.inRange(resized_image_hsv, lower_red, upper_red)

    # Find connected components in the density map
    _, labels, stats, _ = cv2.connectedComponentsWithStats(density_map.astype(np.uint8))

    # Iterate through each cluster
    for label in range(1, np.max(labels) + 1):
        # Check if the corresponding region in the resized image has any red pixels
        cluster_region = red_mask[labels == label]
        if np.sum(cluster_region) == 0:
            # If no red pixels, delete the cluster by setting its pixels to 0 in the density map
            density_map[labels == label] = 0

    return density_map

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
    
    b=(1-ratio)*2.2
    if b<0: b=0

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

