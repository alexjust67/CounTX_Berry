from main2 import mainf
import os
import pandas as pd
import torch
import module1 as m1
import open_clip
from PIL import Image
import json 
import numpy as np

dir_path='./img/datas/images/'                                #path to the directory containing the images.
dir_path_names='./img/datas/images/'                      #path to the directory containing the images names.
cvs_path='./cvs_data/data.csv'                              #path to the cvs file.
ckp_path='./chkp/paper-model.pth'                           #path to the checkpoint.

# Load model.
device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")      #use gpu if available.
model = m1.main_counting_network()
checkpoint = torch.load(ckp_path, map_location=device)      #load the checkpoint.
model.load_state_dict(checkpoint["model"], strict=False)                    #load the model.

dirfiles=len([entry for entry in os.listdir(dir_path_names) if os.path.isfile(os.path.join(dir_path_names, entry))])    #count the number of files in the directory.

#prepare the dataframe.
d = {'img': [], 'exp_val': [], 'clus_pred': [], 'treshold': [], 'kern_size': [], 'text': [], 'clus-error': [],'delta_bacche_abs':[],'delta_bacche':[]}
df = pd.DataFrame(data=d)
df.to_csv(cvs_path)

#queryes to feed the model.
queryes=["the number of berries"]#,"the number of berries", "a photo of the raspberries","a photo of the berries", "a drone image of the raspberries","a drone image of the berries","the berries on the ground"]  #"the berry", "the berries on the ground","the red berries","the number of red berries","the number of raspberries", "the raspberries"

#kernel sizes, this is the size of the square that will be fed to the model (after being reshaped to 224*224).
sqsz=[350]

#tresh=[[y/1000 for y in range(20,80,5)]]#best
tresh=[[0.45]]

#max length of the cluster, if the cluster is bigger than this value it will be ignored.
mxlen=[50]

#strides, if 0 no stride will be used, if set to "autostride", it will set the stride automatically to evenly cover the image with a minimum set by the second value in the list.
stride=[[50,50]]

#visualization parameters.
showimage=False
#save density map as npy.
density_datasave=False
#show the kernel.
showkern=False
#show normalization
shownorm=False

#loop through the parameters.
iterat=1
for strid in stride:
    for tre in tresh:
        for mxl in mxlen:
            for sqsz1 in sqsz:
                for text in queryes:

                    for file in os.listdir(dir_path_names):
                        
                        filename = os.fsdecode(file)
                        
                        #get the ground truth value.
                        ground_truth=filename[6:(filename[6:].find("_")+6)]#(filename.find("pred")+1):(filename.find("_")-1)
                                                
                        # Load the image.
                        image = Image.open(str(dir_path+str(filename))).convert("RGB")
                        image.load()

                        stridex,stridey,clst,tre,sqz=mainf(
                        
                        model,                                                                                              #model to load.
                                                
                        image=image,                                                                                        #image.

                        text=text,                                                                                          #text prompt.
                        
                        sqsz=sqsz1,                                                                                         #size of image division kernel.
                        
                        dm_save=density_datasave,                                                                           #save density map as npy.
                        
                        showimage=showimage,                                                                                #show also the image.
                        
                        showkern=showkern,                                                                                  #show the kernel.                        
                        
                        tresh=tre,                                                                                          #image filtering treshold, values under this will not be taken into consideration by the clusterfinder.

                        text_add=f"file no:{iterat} of {dirfiles*len(sqsz)*len(tresh)*len(stride)*len(queryes)}",           #text to add to the print.

                        ground_truth=ground_truth,                                                                          #ground truth value.

                        mxlen=mxl,                                                                                          #maximum cluster length, if the cluster is bigger than this value it will be ignored.

                        stride=strid,                                                                                       #stride value.                                 

                        device=device,                                                                                      #device to use.
                        
                        shownorm=shownorm,                                                                                  #show normalization.
                        )
                        
                        ind=0

                        for clsti in clst: 
                            
                            d2 = {'img': [filename], 'exp_val': [ground_truth], 'clus_pred': [clsti], 'treshold': [tre[ind]], 'kern_size': [sqz], 'text': [str(text)], 'clus-error': [(clsti-int(ground_truth))**2],'delta_bacche_abs':[abs(int(ground_truth)-clsti)],'delta_bacche':[clsti-int(ground_truth)]}
                        
                            #append the data to the dataframe.
                            df2=pd.DataFrame(data=d2)
                            df=df.append(df2)
                            ind+=1
                        iterat+=1
                        df.to_csv(cvs_path,mode='a', header=False)
                        df = df[0:0]

