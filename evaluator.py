from main import mainf
import os
import pandas as pd
import torch
import module1 as m1
import open_clip
from PIL import Image

dir_path='./img/datas'
iterat=1

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
d = {'img': [], 'exp_val': [], 'clus_pred': [], 'max rec': [], 'treshold': [], 'kern_size': [], 'text': [], 'clus-error': [],'Notes': [],'delta_bacche_abs':[],'delta_bacche':[],'stridex':[],'stridey':[]}
df = pd.DataFrame(data=d)
df_roll=pd.DataFrame(data=d)

#prepare the parameters.

#queryes to feed the model.
queryes=[["the number of berries", "the raspberries on the ground","the berry", "the berries on the ground","the red berries"]]  #"the berry", "the berries on the ground","the red berries","the number of red berries","the number of raspberries", "the raspberries"

#kernel sizes, this is the size of the square that will be fed to the model (after being reshaped to 224*224).
sqsz=[975]

#tresholds for the clusterfinder, all values outputted from the model under this treshold will be ignored.
tresh=[0.5,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.7]

#max length of the cluster, if the cluster is bigger than this value it will be ignored.
mxlen=[17]

#strides, if 0 no stride will be used, if set to "autostride", it will set the stride automatically to evenly cover the image with a minimum set by the second value in the list.
stride=[[50,50],["autostride",50],[30,30],[40,40],[50,50]]

#NOISE SETTINGS
#number of noise iterations, this is the number of times the image will be fed to the model with a random noise added to it. (0=no noise)
iterations=5
#level of noise, if noise is set to equal this value will be by how much the map will be subtracted
noise_lvl=60
#tipe of noise: "speckle"   "salt and pepper"   "poisson"   "gaussian"
noise_type="None"
#dropoff of the cluster ray
dropoff=5
#radius of the attenuation
rad=50
#coefficient to the density map when added to the noise map
density_map_coeff=0.3

#notes to add to the dataframe.
notes=""

#visualization parameters.
showimage=True
density_datasave=False
rolling_datasave=True
shownoise=False

#loop through the parameters.
for strid in stride:
    
    if strid==0:
        strides=False
    else:
        strides=True

    for tre in tresh:
        for mxl in mxlen:
            for sqsz1 in sqsz:
                for text in queryes:
                    
                    if rolling_datasave:
                        df_roll=pd.DataFrame(data=d)
                    
                    for file in os.listdir(dir_path):
                        
                        filename = os.fsdecode(file)
                        
                        #get the ground truth value.
                        ground_truth=filename[(filename.find("pred")+5):(filename.find("2000")-1)]

                        # Load and process the image.
                        image = Image.open(f"./img/datas/{filename}").convert("RGB")
                        image.load()

                        stridex,stridey,clst,tre,recus,sqz=mainf(
                        
                        model,                                                                                              #model to load.
                        
                        image_file_name = f"./img/datas/{filename}",                                                        #image file name.
                        
                        image=image,                                                                                        #image.

                        text=text,                                                                                          #text prompt.
                        
                        sqsz=sqsz1,                                                                                         #size of image division kernel.
                        
                        dm_save=density_datasave,                                                                           #save density map as npy.
                        
                        showimage=showimage,                                                                                #show also the image.
                        
                        recl=60,                                                                                            #maximum cluster finder recursion; it approximates to the area of cluster.
                        
                        tresh=tre,                                                                                          #image filtering treshold, values under this will not be taken into consideration by the clusterfinder.

                        text_add=f"file no:{iterat} of {dirfiles*len(sqsz)*len(tresh)*len(stride)*len(queryes)}",           #text to add to the print.

                        ground_truth=ground_truth,                                                                          #ground truth value.

                        mxlen=mxl,                                                                                          #maximum cluster length, if the cluster is bigger than this value it will be ignored.

                        no_stride=(not strides),                                                                            #if true no stride will be used.                       

                        stride=strid,                                                                                       #stride value.                                 

                        device=device,                                                                                      #device to use.
                        
                        clusteralg="rec-find",                                                                              #rec-find or ray-find(in development).
                        
                        iterations=iterations,                                                                              #number of noise iterations.
                        
                        noise_lvl=noise_lvl,                                                                                        #noise level

                        noise_type=noise_type,

                        dropoff=dropoff,

                        density_map_coeff=density_map_coeff,

                        shownoise=shownoise,

                        rad=rad
                        
                        )
                        
                        d2 = {'img': [filename], 'exp_val': [ground_truth], 'clus_pred': [len(clst)], 'max rec': [recus], 'treshold': [tre], 'kern_size': [sqz], 'text': [str(text)], 'clus-error': [abs(int(ground_truth)-len(clst))/int(ground_truth)],'Notes': [notes],'delta_bacche_abs':[abs(int(ground_truth)-len(clst))],'delta_bacche':[int(ground_truth)-len(clst)],'stridex':[stridex],'stridey':[stridey]}
                        
                        #append the data to the dataframe.
                        df2=pd.DataFrame(data=d2)
                        df=df.append(df2)
                        df.to_csv('./cvs_data/data.csv')
                        
                        if rolling_datasave:
                            #append the data to the rolling dataframe.
                            df_roll=df_roll.append(df2)

                        iterat+=1
                    
                    #if rolling data save is true save the rolling dataframe.
                    if rolling_datasave:    
                        d3 = {'img': ["Error mean"], 'exp_val': [None], 'clus_pred': [None], 'max rec': [None], 'treshold': [None], 'kern_size': [None], 'text': [str(text)], 'clus-error': [df_roll.iloc[:,9].mean()],'Notes': [None],'delta_bacche_abs':[df_roll.iloc[:,11].mean()],'delta_bacche':[int(ground_truth)-len(clst)],'stridex':[stridex],'stridey':[stridey]}
                        df3=pd.DataFrame(data=d3)
                        with open(f'./rolling_data/out_roll{(iterat-1)/5}.txt', 'w') as f:
                            f.write(df_roll.describe().to_string())
                        df_roll=df_roll.append(df3)
                        df_roll.to_csv(f'./rolling_data/out_roll{(iterat-1)/5}.csv')
