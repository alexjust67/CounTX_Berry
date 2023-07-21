import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#this script is used to parse the data from the csv files and plot them.



toghether=False      #used to compare different kernel sizes.

#import the dataframe
print("importing the data")
df=pd.read_csv("./cvs_data/BEST_0.45_50_350_valiset.csv")

#sort the dataframe by kernel size and expected value.
df=df.sort_values(['kern_size', 'exp_val'], ascending=[True, True])

#iterate over the dataframe and divide it by the column "expected value"
actual_val=[[[]]]
Clus_pred=[[[]]]
Clus_error=[[[]]]
delta_bacche=[[[]]]
delta_bacche_abs=[[[]]]
prev_row=1
prev_kern=0
start=False
kernel=0
kernlist=[]
print("dividing the dataframe")
for index, row in df.iterrows():
    if start==False:
        actual_val[0][len(actual_val)-1].append(row['exp_val'])
        prev_kern=row['kern_size']
        prev_row=row['exp_val']
        start=True
        kernlist.append(row['kern_size'])
    if row['kern_size']==prev_kern:
        if row['exp_val']==prev_row:
            Clus_pred[kernel][len(Clus_pred[kernel])-1].append(row['clus_pred'])
            Clus_error[kernel][len(Clus_error[kernel])-1].append(row['clus-error'])
            delta_bacche[kernel][len(delta_bacche[kernel])-1].append(row['delta_bacche'])
            delta_bacche_abs[kernel][len(delta_bacche_abs[kernel])-1].append(row['delta_bacche_abs'])
        else:
            actual_val[kernel].append([row['exp_val']])
            Clus_pred[kernel].append([row['clus_pred']])
            Clus_error[kernel].append([row['clus-error']])
            delta_bacche[kernel].append([row['delta_bacche']])
            delta_bacche_abs[kernel].append([row['delta_bacche_abs']])
            prev_row=row['exp_val']
    else:
        kernel+=1
        actual_val.append([])
        Clus_pred.append([])
        Clus_error.append([])
        delta_bacche.append([])
        delta_bacche_abs.append([])
        actual_val[kernel].append([row['exp_val']])
        Clus_pred[kernel].append([row['clus_pred']])
        Clus_error[kernel].append([row['clus-error']])
        delta_bacche[kernel].append([row['delta_bacche']])
        delta_bacche_abs[kernel].append([row['delta_bacche_abs']])
        prev_kern=row['kern_size']
        prev_row=row['exp_val']
        kernlist.append(row['kern_size'])
print("done dividing the dataframe, averaging...")

#calculate the mean of the lists.
Clus_pred_mean=[]
Clus_error_mean=[]
delta_bacche_mean=[]
delta_bacche_abs_mean=[]
for x in range(kernel+1):
    Clus_pred_mean.append([])
    Clus_error_mean.append([])
    delta_bacche_mean.append([])
    delta_bacche_abs_mean.append([])
    for i in range(len(Clus_pred[x])):
        Clus_pred_mean[x].append(sum(Clus_pred[x][i])/len(Clus_pred[x][i]))
        Clus_error_mean[x].append(sum(Clus_error[x][i])/len(Clus_error[x][i]))
        delta_bacche_mean[x].append(sum(delta_bacche[x][i])/len(delta_bacche[x][i]))
        delta_bacche_abs_mean[x].append(sum(delta_bacche_abs[x][i])/len(delta_bacche_abs[x][i]))

#calculate the variance of the lists.
Clus_pred_var=[]
Clus_error_var=[]
delta_bacche_var=[]
delta_bacche_abs_var=[]
for x in range(kernel+1):
    Clus_pred_var.append([])
    Clus_error_var.append([])
    delta_bacche_var.append([])
    delta_bacche_abs_var.append([])
    for i in range(len(Clus_pred[x])):
        Clus_pred_var[x].append(sum([(y - Clus_pred_mean[x][i]) ** 2 for y in Clus_pred[x][i]]) / (len(Clus_pred[x][i])))#-1
        Clus_error_var[x].append(sum([(y - Clus_error_mean[x][i]) ** 2 for y in Clus_error[x][i]]) / (len(Clus_error[x][i])))#-1
        delta_bacche_var[x].append(sum([(y - delta_bacche_mean[x][i]) ** 2 for y in delta_bacche[x][i]]) / (len(delta_bacche[x][i])))#-1
        delta_bacche_abs_var[x].append(sum([(y - delta_bacche_abs_mean[x][i]) ** 2 for y in delta_bacche_abs[x][i]]) / (len(delta_bacche_abs[x][i])))

print("done averaging, plotting...")
if not(toghether):
    for x in range(kernel+1):
        #plot the data through a multiaxis graph.
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot([y*1 for y in Clus_pred_mean[x]])
        axs[0, 0].plot(actual_val[x], 'tab:blue')
        axs[0, 0].set_ylabel('n of bacche (prediction)')
        axs[0, 0].set_xlabel('n of bacche (ground truth)')
        axs[0, 0].fill_between(range(0,len(actual_val[x])),np.array(Clus_pred_mean[x])-np.sqrt(np.array(Clus_pred_var[x])), np.array(Clus_pred_mean[x])+np.sqrt(np.array(Clus_pred_var[x])),alpha=0.5)
        axs[0, 0].set_ylim([0, 250])
        axs[0, 0].set_title('Clus_pred_mean')
        axs[0, 1].plot(Clus_error_mean[x], 'tab:orange')
        axs[0, 1].set_ylabel('MSE')
        axs[0, 1].set_title('Clus_error_mean')
        
        if True:
            axs[1, 0].set_ylabel('num_datapoins')
            axs[1, 0].set_title('no of datapoints')
            axs[1, 0].plot([len(Clus_pred[x][b]) for b in range(len(Clus_pred[x]))], 'tab:green')
        else:
            axs[1, 0].plot(delta_bacche_mean[x], 'tab:green')
            axs[1, 0].set_ylabel('d_bacche')
            axs[1, 0].set_ylim([-150, 150])
            axs[1, 0].set_title('delta_bacche_mean')
            
        axs[1, 1].plot(delta_bacche_abs_mean[x], 'tab:red')
        axs[1, 1].set_ylim([0, 150])
        axs[1, 1].set_ylabel('abs_d_bacche')
        axs[1, 1].set_title('delta_bacche_abs_mean:'+str(round(np.mean(delta_bacche_abs_mean[x]),2))+"   "+str(round(np.mean(delta_bacche_mean[x]),2)))
        fig.suptitle('Kernel size: 350, Treshold: 0.45, max cluster size: 50, stride: 50, query: the number of berries')
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(12, 8)
        #plt.savefig('D:/Vstudio/Vscode/CounTX_Berry/CounTX_Berry/cvs_data/plot'+str(kernlist[x])+'.png', dpi=1000)
        plt.show()
else:

    fig, axs = plt.subplots(2, 2)
    for x in range(kernel+1):
        axs[0, 0].plot(Clus_pred_mean[x],label=kernlist[x])
    axs[0, 0].plot(actual_val[x], 'tab:blue')
    axs[0, 0].set_ylabel('n of bacche')
    axs[0, 0].set_title('Clus_pred_mean')
    for x in range(kernel+1):
        axs[0, 1].plot(Clus_error_mean[x])
    axs[0, 1].set_ylabel('err percent')
    axs[0, 1].set_yscale('log')
    axs[0, 1].set_title('Clus_error_mean')
    for x in range(kernel+1):
        axs[1, 0].plot(delta_bacche_mean[x])
    axs[1, 0].set_ylabel('d_bacche')
    axs[1, 0].set_title('delta_bacche_mean')
    for x in range(kernel+1):
        axs[1, 1].plot(delta_bacche_abs_mean[x])
    axs[1, 1].set_ylabel('abs_d_bacche')
    axs[1, 1].set_title('delta_bacche_abs_mean')
    fig.legend()
    fig.suptitle('Kernel size: '+"all")
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(12, 8)
    plt.show()

