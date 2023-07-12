from main2 import mainf
import os
import pandas as pd
import torch
import module1 as m1
import open_clip
from PIL import Image

dir_path='./img/datas/images/'                                #path to the directory containing the images.

# Load model.
device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")      #use gpu if available.
model = m1.main_counting_network()
checkpoint = torch.load("./chkp/paper-model.pth", map_location=device)      #load the checkpoint.
model.load_state_dict(checkpoint["model"], strict=False)                    #load the model.

dirfiles=len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))])    #count the number of files in the directory.

#check if data.cvs exist and if it does throw an error.
#if os.path.isfile('./cvs_data/data.csv'):
#    raise Exception("WARNING: ./cvs_data/data.csv already exists, please delete it before running the program.")

#prepare the dataframe.
d = {'img': [], 'exp_val': [], 'clus_pred': [], 'treshold': [], 'kern_size': [], 'text': [], 'clus-error': [],'Notes': [],'delta_bacche_abs':[],'delta_bacche':[],'stridex':[],'stridey':[]}
df = pd.DataFrame(data=d)
df.to_csv('./cvs_data/data.csv')
#prepare the parameters.

#queryes to feed the model.
queryes=["the number of berries"]#,"the number of berries", "a photo of the raspberries","a photo of the berries", "a drone image of the raspberries","a drone image of the berries","the berries on the ground"]  #"the berry", "the berries on the ground","the red berries","the number of red berries","the number of raspberries", "the raspberries"

#kernel sizes, this is the size of the square that will be fed to the model (after being reshaped to 224*224).
sqsz=[300,250,350]

#tresholds for the clusterfinder, all values outputted from the model under this treshold will be ignored.

tresh=[[]]
for l in range(0,100,1):
    tresh[0].append(l/100)

#max length of the cluster, if the cluster is bigger than this value it will be ignored.
mxlen=[17]

#strides, if 0 no stride will be used, if set to "autostride", it will set the stride automatically to evenly cover the image with a minimum set by the second value in the list.
stride=[[50,50]]

#notes to add to the dataframe.
notes=""

#visualization parameters.
showimage=False
density_datasave=False
rolling_datasave=False
shownoise=False
showkern=False

#loop through the parameters.
iterat=1
for strid in stride:
    for tre in tresh:
        for mxl in mxlen:
            for sqsz1 in sqsz:
                for text in queryes:
                    
                    if rolling_datasave:
                        df_roll=pd.DataFrame(data=d)
                    
                    for file in os.listdir(dir_path):
                        
                        filename = os.fsdecode(file)
                        
                        #get the ground truth value.
                        ground_truth=filename[6:(filename[6:].find("_")+6)]#(filename.find("pred")+1):(filename.find("_")-1)

                        # Load and process the image.
                        image = Image.open(str(dir_path+str(filename))).convert("RGB")
                        image.load()

                        stridex,stridey,clst,tre,sqz=mainf(
                        
                        model,                                                                                              #model to load.
                        
                        image_file_name = str(dir_path+str(filename)),                                                        #image file name.
                        
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
                        
                        )
                        
                        ind=0
                        for clsti in clst: 
                            
                            d2 = {'img': [filename], 'exp_val': [ground_truth], 'clus_pred': [clsti], 'treshold': [tre[ind]], 'kern_size': [sqz], 'text': [str(text)], 'clus-error': [abs(int(ground_truth)-clsti)/int(ground_truth)],'Notes': [notes],'delta_bacche_abs':[abs(int(ground_truth)-clsti)],'delta_bacche':[int(ground_truth)-clsti],'stridex':[stridex],'stridey':[stridey]}
                        
                            #append the data to the dataframe.
                            df2=pd.DataFrame(data=d2)
                            df=df.append(df2)
                            
                            if rolling_datasave:
                                #append the data to the rolling dataframe.
                                df_roll=df_roll.append(df2)
                            ind+=1
                        iterat+=1
                        df.to_csv('./cvs_data/data.csv',mode='a', header=False)
                        df = df[0:0]
                    #if rolling data save is true save the rolling dataframe.
                    if rolling_datasave:    
                        d3 = {'img': ["Error mean"], 'exp_val': [None], 'clus_pred': [None], 'treshold': [None], 'kern_size': [None], 'text': [str(text)], 'clus-error': [df_roll.iloc[:,7].mean()],'Notes': [None],'delta_bacche_abs':[df_roll.iloc[:,9].mean()],'delta_bacche':[None],'stridex':[stridex],'stridey':[stridey]}
                        df3=pd.DataFrame(data=d3)
                        with open(f'./rolling_data/out_roll{(iterat-1)/5}.txt', 'w') as f:
                            f.write(df_roll.describe().to_string())
                        df_roll=df_roll.append(df3)
                        df_roll.to_csv(f'./rolling_data/out_roll{(iterat-1)/5}.csv')
