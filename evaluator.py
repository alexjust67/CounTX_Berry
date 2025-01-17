from main2 import mainf
import os
import pandas as pd
import torch
import module1 as m1
import open_clip
from PIL import Image
from PIL import ImageFilter
import numpy as np
import matplotlib.pyplot as plt

if (False):
    dir_path='./img/datas/images/'                                #path to the directory containing the images.
    dir_path_names='./img/datas/images/'                      #path to the directory containing the images names.
    cvs_path='./cvs_data/data.csv'                              #path to the cvs file.
    ckp_path='./chkp/paper-model.pth'                           #path to the checkpoint.
elif (False):
    dir_path='D:/Vstudio/Vscode/CounTX_Berry/CounTX_Berry/img/datas/images/'                                #path to the directory containing the images.
    dir_path_names='D:/Vstudio/Vscode/CounTX_Berry/CounTX_Berry/img/renders/drone/'                      #path to the directory containing the images names.
    cvs_path='D:/Vstudio/Vscode/CounTX_Berry/CounTX_Berry/cvs_data/data.csv'                              #path to the cvs file.
    ckp_path='D:/Vstudio/Vscode/CounTX_Berry/CounTX_Berry/chkp/paper-model.pth'                           #path to the checkpoint.
else:
    dir_path='/home/agiustina/CounTX_Berry/img/datas/images/'                                  #path to the directory containing the images.
    dir_path_names='/home/agiustina/CounTX_Berry/img/datas/images/'                            #path to the directory containing the images names.
    cvs_path='/home/agiustina/CounTX_Berry/cvs_data/data.csv'                                   #path to the cvs file.
    ckp_path='/home/agiustina/CounTX_Berry/chkp/paper-model.pth'                                #path to the checkpoint.

# Load model.
device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")      #use gpu if available.
model = m1.main_counting_network()

#load the checkpoint.
checkpoint = torch.load(ckp_path, map_location=device)

#load the model.      
model.load_state_dict(checkpoint["model"], strict=False)                    

#prepare the files
dirfiles=len([entry for entry in os.listdir(dir_path_names) if os.path.isfile(os.path.join(dir_path_names, entry))])    #count the number of files in the directory.
ind=0

#prepare the dataframe.
d = {'indx':[],'img': [], 'exp_val': [], 'clus_pred': [], 'treshold': [], 'kern_size': [], 'query': [], 'clus-error': [],'delta_bacche_abs':[],'delta_bacche':[]}
df = pd.DataFrame(data=d)
df.to_csv(cvs_path)

#VARIABLES
#kernel sizes, this is the size of the square that will be fed to the model (after being reshaped to 224*224).
sqsz=[350]
adap_krnl=False

#tresh=[[y/1000 for y in range(20,80,5)]]#best
tresh=[0.45]

#normalization value.
norm=0

#max length of the cluster, if the cluster is bigger than this value it will be ignored.
mxlen=[50]

#strides, if 0 no stride will be used, if set to "autostride", it will set the stride automatically to evenly cover the image with a minimum set by the second value in the list.
stride=[[50,50]]

#filter the cluster by color
colorfilter=False

#queryes to feed the model.
queryes=[["the number of berries"]]#,"the number of berries", "a photo of the raspberries","a photo of the berries", "a drone image of the raspberries","a drone image of the berries","the berries on the ground"]  #"the berry", "the berries on the ground","the red berries","the number of red berries","the number of raspberries", "the raspberries"


#visualization parameters.
showimage=True
generalize_results=False
#save density map as npy.
density_datasave=False
#show the kernel.
showkern=False
#show normalization
shownorm=True

#template.format('berries') for template in templates]
#loop through the parameters.
ind=0
for strid in stride:
    for tre in tresh:
        for mxl in mxlen:
            for sqsz1 in sqsz:
                for text in queryes:

                    for file in os.listdir(dir_path_names):
                        
                        filename = os.fsdecode(file)
                        
                        #get the height of the drone, if 0 all functions that require the height will be deactivated.
                        #height=float(filename[:filename.find("m")])
                        height=0

                        #get the ground truth value for CRAID dataset.
                        ground_truth=filename[6:(filename[6:].find("_")+6)]
                                                
                        # Load the image.
                        image = Image.open(str(dir_path+str(filename))).convert("RGB")
                        image.load()

                        stridex,stridey,clsti,tre,sqz=mainf(
                        
                        model,                                                                                              #model to load.
                                                
                        image=image,                                                                                        #image.

                        query=text,                                                                                          #text prompt.
                        
                        sqsz=sqsz1,                                                                                         #size of image division kernel.
                        
                        dm_save=density_datasave,                                                                           #save density map as npy.
                        
                        showimage=showimage,                                                                                #show also the image.
                        
                        showkern=showkern,                                                                                  #show the kernel.                        
                        
                        tresh=tre,                                                                                          #image filtering treshold, values under this will not be taken into consideration by the clusterfinder.

                        text_add=filename,                                                                                  #text to add to the print.

                        ground_truth=ground_truth,                                                                          #ground truth value.

                        mxlen=mxl,                                                                                          #maximum cluster length, if the cluster is bigger than this value it will be ignored.

                        stride=strid,                                                                                       #stride value.                                 

                        device=device,                                                                                      #device to use.
                        
                        shownorm=shownorm,                                                                                  #show normalization.
                        
                        norm=norm,                                                                                          #normalization value.
                        
                        colorfilter=colorfilter,                                                                            #filter the cluster by color.
                        
                        height=height,                                                                                      #height of the drone.
                        
                        adap_krnl=adap_krnl,                                                                                #use adaptive kernel.
                        
                        generalize_results=generalize_results,                                                              #generalize the results.
                        )
                        
                        #create the dataframe.
                        try:
                            ground_truth=int(ground_truth)
                        except:
                            ground_truth=0
                        
                        d2 = {'indx':[ind],'img': [filename], 'exp_val': [ground_truth], 'clus_pred': [clsti], 'treshold': [tre], 'kern_size': [sqz], 'query': [str(text)], 'clus-error': [(clsti-int(ground_truth))**2],'delta_bacche_abs':[abs(int(ground_truth)-clsti)],'delta_bacche':[clsti-int(ground_truth)]}
                        ind+=1

                        #append the data to the dataframe.
                        df2=pd.DataFrame(data=d2)
                        df=df.append(df2)

                        #save the dataframe to file and delete.
                        df.to_csv(cvs_path,mode='a', header=False)
                        df = df[0:0]

