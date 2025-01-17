import pandas as pd
import matplotlib.pyplot as plt

toghether=True
alltoghether=False
best=True

#import the dataframe
print("importing the dataframe")
df=pd.read_csv("D:/Vstudio/Vscode/CounTX_Berry/CounTX_Berry/cvs_data/data.csv")
#df=df.append(pd.read_csv("./cvs_data/data250-350.csv"))
#df=df.append(pd.read_csv("./cvs_data/data320-330-340.csv"))
#df=df.append(pd.read_csv("./cvs_data/456.csv"))
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
print("dividing the dataframe")
for index, row in df.iterrows():
    if start==False:
        actual_val[0][len(actual_val)-1].append([row['exp_val']])
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
            prev_row=row['exp_val']
            prev_tresh=row['treshold']
            treshlist.append(row['treshold'])
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
        
print("dataframe divided, calculating means")
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
print("means calculated, plotting")
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
elif alltoghether:
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
elif best:
    Clus_pred_mean_best=[]
    first=True
    ind=0
    tot=0
    for l in actual_val[0][0]:
        try:
            tot+=sum(l)
        except:
            continue
    inde=0
    indelist=[]
    for x in Clus_pred_mean:
        Clus_pred_mean_best.append([])
        indelist.append([])
        for y in x:
            if first:
                inde=0
                Clus_pred_mean_best[ind].append(y)
                indelist[ind].append(inde)
                inde+=1
                first=False
            else:
                err=sum(y)-tot
                if err<0:
                    err=-err
                for n in range(len(Clus_pred_mean_best[ind])):
                    err2=sum(Clus_pred_mean_best[ind][n])-tot
                    if err2<0:
                        err2=-err2
                    if err<err2:
                        Clus_pred_mean_best[ind].insert(n,y)
                        indelist[ind].insert(n,inde)
                        break
                    elif n==len(Clus_pred_mean_best[ind])-1:
                        Clus_pred_mean_best[ind].append(y)
                        indelist[ind].append(inde)
                        break
                    inde+=1
        first=True
        ind+=1

    for b in range(kernel+1):
        fig, axs = plt.subplots()
        for x in range(7):
            axs.plot(Clus_pred_mean_best[b][x],label=(treshlist[b]))
        axs.set_ylabel('n of bacche')
        axs.plot(actual_val[b][x], 'tab:blue')
        axs.set_title('Clus_pred_mean')
        fig.legend()
        fig.suptitle('Kernel size: '+str(kernlist[b])+"treshold: "+"all")
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(15, 10)
        #plt.savefig('./cvs_data/plot'+str("all")+str(kernlist[b])+'.png', dpi=1000)
        plt.show()
