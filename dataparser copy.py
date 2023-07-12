import pandas as pd
import matplotlib.pyplot as plt

toghether=True
#import the dataframe
df=pd.read_csv("./cvs_data/data.csv")
df = df.sort_values(['kern_size', 'treshold','exp_val'], ascending=[True, True,True])#
#iterate over the dataframe and divide it by the column "expected value"
actual_val=[[[[]]]]#[df['exp_val'][0]
Clus_pred=[[[[]]]]
Clus_error=[[[[]]]]
delta_bacche=[[[[]]]]
delta_bacche_abs=[[[[]]]]
prev_row=1
prev_kern=0
prev_tresh=0
start=False
kernel=0
kernlist=[]
treshold=0
treshlist=[]
for index, row in df.iterrows():
    if start==False:
        actual_val[0][len(actual_val)-1].append(row['exp_val'])
        prev_kern=row['kern_size']
        prev_row=row['exp_val']
        prev_tresh=row['treshold']
        start=True
        kernlist.append(row['kern_size'])
        treshlist.append(row['treshold'])
    if row['kern_size']==prev_kern:
        if row['treshold']==prev_tresh:
            if row['exp_val']==prev_row:
                Clus_pred[kernel][treshold][len(Clus_pred[kernel][treshold])-1].append(row['clus_pred'])
                Clus_error[kernel][treshold][len(Clus_error[kernel][treshold])-1].append(row['clus-error'])
                delta_bacche[kernel][treshold][len(delta_bacche[kernel][treshold])-1].append(row['delta_bacche'])
                delta_bacche_abs[kernel][treshold][len(delta_bacche_abs[kernel][treshold])-1].append(row['delta_bacche_abs'])
            else:
                actual_val[kernel][treshold].append([row['exp_val']])
                Clus_pred[kernel][treshold].append([row['clus_pred']])
                Clus_error[kernel][treshold].append([row['clus-error']])
                delta_bacche[kernel][treshold].append([row['delta_bacche']])
                delta_bacche_abs[kernel][treshold].append([row['delta_bacche_abs']])
                prev_row=row['exp_val']
        else:
            treshold+=1
            actual_val[kernel].append([])
            Clus_pred[kernel].append([])
            Clus_error[kernel].append([])
            delta_bacche[kernel].append([])
            delta_bacche_abs[kernel].append([])
            actual_val[kernel][treshold].append([row['exp_val']])
            Clus_pred[kernel][treshold].append([row['clus_pred']])
            Clus_error[kernel][treshold].append([row['clus-error']])
            delta_bacche[kernel][treshold].append([row['delta_bacche']])
            delta_bacche_abs[kernel][treshold].append([row['delta_bacche_abs']])
            prev_kern=row['kern_size']
            prev_row=row['exp_val']
            kernlist.append(row['kern_size'])
    else:
        kernel+=1
        treshold=0
        actual_val.append([[]])
        Clus_pred.append([[]])
        Clus_error.append([[]])
        delta_bacche.append([[]])
        delta_bacche_abs.append([[]])
        actual_val[kernel][treshold].append([row['exp_val']])
        Clus_pred[kernel][treshold].append([row['clus_pred']])
        Clus_error[kernel][treshold].append([row['clus-error']])
        delta_bacche[kernel][treshold].append([row['delta_bacche']])
        delta_bacche_abs[kernel][treshold].append([row['delta_bacche_abs']])
        prev_kern=row['kern_size']
        prev_row=row['exp_val']
        kernlist.append(row['kern_size'])
        prev_tresh=row['treshold']
        treshlist.append(row['treshold'])

#calculate the mean of the lists.
Clus_pred_mean=[]
Clus_error_mean=[]
delta_bacche_mean=[]
delta_bacche_abs_mean=[]
for y in range(kernel+1):
    Clus_pred_mean.append([])
    Clus_error_mean.append([])
    delta_bacche_mean.append([])
    delta_bacche_abs_mean.append([])
    for x in range(treshold+1):
        Clus_pred_mean[y].append([])
        Clus_error_mean[y].append([])
        delta_bacche_mean[y].append([])
        delta_bacche_abs_mean[y].append([])
        for i in range(len(Clus_pred[y][x])):
            Clus_pred_mean[y][x].append(sum(Clus_pred[y][x][i])/len(Clus_pred[y][x][i]))
            Clus_error_mean[y][x].append(sum(Clus_error[y][x][i])/len(Clus_error[y][x][i]))
            delta_bacche_mean[y][x].append(sum(delta_bacche[y][x][i])/len(delta_bacche[y][x][i]))
            delta_bacche_abs_mean[y][x].append(sum(delta_bacche_abs[y][x][i])/len(delta_bacche_abs[y][x][i]))

if not(toghether):
    for b in range(kernel+1):
        for x in range(treshold+1):
            #plot the data through a multiaxis graph.
            fig, axs = plt.subplots(2, 2)
            axs[0, 0].plot(Clus_pred_mean[b][x])
            axs[0, 0].plot(actual_val[b][x], 'tab:blue')
            axs[0, 0].set_title('Clus_pred_mean')
            axs[0, 1].plot(Clus_error_mean[b][x], 'tab:orange')
            axs[0, 1].set_title('Clus_error_mean')
            axs[1, 0].plot(delta_bacche_mean[b][x], 'tab:green')
            axs[1, 0].set_title('delta_bacche_mean')
            axs[1, 1].plot(delta_bacche_abs_mean[b][x], 'tab:red')
            axs[1, 1].set_title('delta_bacche_abs_mean')
            fig.suptitle('Kernel size: '+str(kernlist[b][x]))
            figure = plt.gcf()  # get current figure
            figure.set_size_inches(12, 8)
            plt.savefig('./cvs_data/plot'+str(kernlist[b][x])+str(treshlist[b])+'.png', dpi=1000)
            plt.show()
else:
    for b in range(kernel+1):
        fig, axs = plt.subplots(2, 2)
        for x in range(treshold+1):
            axs[0, 0].plot(Clus_pred_mean[b][x])
        axs[0, 0].plot(actual_val[b][x], 'tab:blue')
        axs[0, 0].set_title('Clus_pred_mean')
        for x in range(treshold+1):
            axs[0, 1].plot(Clus_error_mean[b][x])
        axs[0, 1].set_title('Clus_error_mean')
        for x in range(treshold+1):
            axs[1, 0].plot(delta_bacche_mean[b][x])
        axs[1, 0].set_title('delta_bacche_mean')
        for x in range(treshold+1):
            axs[1, 1].plot(delta_bacche_abs_mean[b][x])
        axs[1, 1].set_title('delta_bacche_abs_mean')
        fig.suptitle('Kernel size: '+str(kernlist[b])+"treshold: "+"all")
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(12, 8)
        plt.savefig('./cvs_data/plot'+str("all")+str(treshlist[b])+'.png', dpi=1000)
        plt.show()

