from main import mainf
import os
import pandas as pd

dir_path='./img/datas'
iterat=1

dirfiles=len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))])

d = {'img': [], 'exp_val': [], 'mod_pred': [], 'clus_pred': [], 'max rec': [], 'treshold': [], 'kern_size': [], 'text': [], 'model-error': [], 'clus-error': [],'Notes': [],'delta_bacche_abs':[],'delta_bacche':[]}
df = pd.DataFrame(data=d)
df_roll=pd.DataFrame(data=d)

queryes=["the number of berries"]
sqsz=[975]
recl=[60]
tresh=[0.2,0.3,0.4,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,6,6.5]
mxlen=[17]

notes="very big kernel to make the network see the berries as well as possible, added a max length"
showimage=True

for tre in tresh:
    for mxl in mxlen:
        for recl1 in recl:
            for sqsz1 in sqsz:
                for text in queryes:

                    df_roll=pd.DataFrame(data=d)

                    for file in os.listdir(dir_path):
                        
                        filename = os.fsdecode(file)
                        
                        predc,clst,tre,recus,sqz=mainf(
                        
                        model_file_name = "./chkp/paper-model.pth",         #model file name
                        
                        image_file_name = f"./img/datas/{filename}",        #image file name
                        
                        text = text,                                        #text prompt
                        
                        sqsz=sqsz1,                                         #size of image division kernel
                        
                        dm_save=False,                                      #save density map as npy
                        
                        showimage=showimage,                                #show also the image
                        
                        recl=recl1,                                            #maximum cluster finder recursion; it approximates to the area of cluster
                        
                        tresh=tre,                                          #image filtering treshold, values under this will not be taken into consideration by the clusterfinder

                        text_add=f"file no:{iterat} of {dirfiles*len(queryes)*len(sqsz)*len(recl)*len(tresh)*len(mxlen)}",

                        ground_truth=filename[(filename.find("pred")+5):(filename.find("2000")-1)],

                        mxlen=mxl
                        
                        )
                        
                        ground_truth=filename[(filename.find("pred")+5):(filename.find("2000")-1)]

                        d2 = {'img': [filename], 'exp_val': [ground_truth], 'mod_pred': [predc], 'clus_pred': [len(clst)], 'max rec': [recus], 'treshold': [tre], 'kern_size': [sqz], 'text': [text], 'model-error': [abs(int(ground_truth)-predc)/int(ground_truth)], 'clus-error': [abs(int(ground_truth)-len(clst))/int(ground_truth)],'Notes': [notes],'delta_bacche_abs':[abs(int(ground_truth)-len(clst))],'delta_bacche':[int(ground_truth)-len(clst)]}
                        
                        df2=pd.DataFrame(data=d2)
                        df=df.append(df2)
                        df.to_csv('./cvs_data/out.csv')
                        iterat+=1
                        df_roll=df_roll.append(df2)
                        
                    d3 = {'img': ["Error mean"], 'exp_val': [None], 'mod_pred': [None], 'clus_pred': [None], 'max rec': [None], 'treshold': [None], 'kern_size': [None], 'text': [text], 'model-error': [df_roll.iloc[:,8].mean()], 'clus-error': [df_roll.iloc[:,9].mean()],'Notes': [None],'delta_bacche_abs':[df_roll.iloc[:,11].mean()],'delta_bacche':[int(ground_truth)-len(clst)]}
                    df3=pd.DataFrame(data=d3)
                    with open(f'./rolling_data/out_roll{iterat}.txt', 'w') as f:
                        f.write(df_roll.describe().to_string())
                    df_roll=df_roll.append(df3)
                    df_roll.to_csv(f'./rolling_data/out_roll{iterat}.csv')
                    #create a text file and write inside it the dataframe summary of df_roll
                    

df=df.append(df2)
df.to_csv('./cvs_data/out.csv')

