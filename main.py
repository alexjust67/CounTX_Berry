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

# Model and image file names.
model_file_name = "./chkp/paper-model.pth"

image_file_name = "./img/bacchetree.png"
# Specify the text description,
# (i.e., response to "what object should be counted?").
text = "the number of berries"

#size of the square (will be resized to 224*224 once in the model)
sqsz=224

# Load model.
model = m1.main_counting_network()
checkpoint = torch.load(model_file_name, map_location="cpu")
model.load_state_dict(checkpoint["model"], strict=False)

# Define preprocessing.
tokenizer = open_clip.get_tokenizer("ViT-B-16")
text1=text[:]
# Load and process the image.
image = Image.open(image_file_name).convert("RGB")
image.load()

w1,h1,image=m1.split_image(image)

w=0
h=0
deh=int((h1/sqsz)*384)  #actual size of the final image
dew=int((w1/sqsz)*384)

density_map=torch.zeros((deh,dew))    #create a zero tensor with the dimension of the final image that will be no of squares * 384(standard model output)
tot=0

time1=[]
time2=[]

for i in image:                       #loop through the image
    
    density_map[h:h+384,w:w+384]=m1.runmodel(i,text,model,tokenizer)           
    
    if h+384!=int((h1/sqsz)*384):                                #check if it has arrived to the bottom
        
        h+=384
        
        time2.insert(0,time.time())
        
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
        
        print(h//384," :  "+str(deh//384)+"    ",w//384," :  "+str(dew//384)+"    ",(tot)," : ",image.shape[0],"  ",str(round((tot/image.shape[0])*100,2))+"%"+ "    "+"Time remaining: ",round(avgperpoint*(image.shape[0]-tot),0),"s")
        tot+=1
    
    else:
        time2.insert(0,time.time())
        tot+=1
        h=0
        w=w+384

pred_cnt = torch.sum(density_map / 60).item()       #predicted count

np.save("./img/results/density_map.npy",density_map.numpy())

print("Predicted Count: " + str(pred_cnt)+"  Prompt: "+text1)

a=postprocess(density_map.numpy(),tresh=0.4)

img=mpimg.imread(image_file_name)
fig,ax = plt.subplots(1,2,sharex=True,sharey=True)
ax[0].imshow(img,extent=(0,density_map.shape[1],density_map.shape[0],0))
ax[1].imshow(img,extent=(0,dew,deh,0))
ax[1].imshow(a, cmap='jet', interpolation='nearest',alpha=0.85)

plt.title("Predicted Count: " + str(clustercount(density_map.numpy(),tresh=0.4,tresh2=0.4)))
text1.replace(" ","_")
text1=text1+"_"+image_file_name
plt.savefig(f"./img/results/resu.jpg",dpi=1500)
plt.show()

