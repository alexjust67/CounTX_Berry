import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


density_map=np.load("./img/results/density_map.npy")

def postprocess(density_map,tresh):
    #density_map=np.power(density_map,10)
    #density_map=np.clip(density_map,0.1,10)
    result = (density_map >= tresh) * density_map
    return result

def iscluster(density_map,x,y,rec_level,tresh,lispt,recl):#TODO: add a max recursion depth to check for size of cluster
    

    if rec_level>recl:
        reclim=True

    if density_map[x,y]>tresh:
        density_map[x,y]=0
        lispt.append([x,y])
        for i in range(-1,2):
            for j in range(-1,2):
                if x+i>=0 and x+i<density_map.shape[0] and y+j>=0 and y+j<density_map.shape[1]:
                    density_map,isclsn,reclim,_=iscluster(density_map,x+i,y+j,rec_level+1,tresh,lispt,recl)
        if not reclim:
            return density_map,True,False,lispt
        else:
            return density_map,False,True,[]
    else:
        return density_map,False,False,[]

def clustercount(density_map, tresh=1,tresh2=0,recl=60):
    
    a=postprocess(density_map,tresh2)
    clslis=[]
    clstr=0
    if (True):
        for x in range(0,a.shape[0]):
            for y in range(0,a.shape[1]):
                a,clsn,_,lst=iscluster(a,x,y,0,tresh,[],recl)
                if clsn==True:
                    clstr+=1
                if len(lst)>0:
                    clslis.append(lst)

    #print("Number of clusters: "+str(clstr))

    #img = mpimg.imread("./img/drone1.jpg")
    #plt.subplot(1,2,1)
    #im1=plt.imshow(img,extent=(0,density_map.shape[1],density_map.shape[0],0))
    #im2=plt.imshow(a, cmap='jet', interpolation='nearest',alpha=0.99)
    #plt.subplot(1,2,2)
    #im1=plt.imshow(img,extent=(0,density_map.shape[1],density_map.shape[0],0))
    #im2=plt.imshow(density_map, cmap='jet', interpolation='nearest',alpha=0.75)

    # pred_cnt1 = np.sum(a / 60)      #predicted count
    # print("Predicted Count: " + str(pred_cnt1))
    # pred_cnt2 = np.sum(density_map / 60)       #predicted count
    # print("Predicted Count: " + str(pred_cnt2))
    #plt.show()
    return clstr,clslis