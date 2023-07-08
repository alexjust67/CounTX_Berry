import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


#apply a treshold to the density map.
def postprocess(density_map,tresh):
    result = (density_map >= tresh) * density_map
    return result

#find the clusters in the density map with recursion.
def iscluster(density_map,x,y,rec_level,tresh,lispt,recl):
    
    #if the recursion level is too high, set the reclim to true to ensure that this won't be counted as a cluster.
    if rec_level>recl:  
        reclim=True

    if density_map[x,y]>tresh:      #check if the point is above the treshold.
        #if it is, set the point to 0 and add it to the list of points in the cluster.
        density_map[x,y]=0
        lispt.append([x,y])
        #check the surrounding points and apply the same function to them recursively.
        for i in range(-1,2):
            for j in range(-1,2):
                if x+i>=0 and x+i<density_map.shape[0] and y+j>=0 and y+j<density_map.shape[1]:
                    density_map,isclsn,reclim,_=iscluster(density_map,x+i,y+j,rec_level+1,tresh,lispt,recl)
        #if the recursion level is too high this will ensure that it won't be counted as a cluster.
        if not reclim:
            return density_map,True,False,lispt
        else:
            return density_map,False,True,[]
    else:
        return density_map,False,False,[]

#setup the clusterfinder.
def clustercount(density_map, tresh=1,tresh2=0,recl=60,mxlen=17,algo="rec-find"):
    
    #apply the treshold to the density map.
    a=postprocess(density_map,tresh2)
    clslis=[]
    clstr=0
    #loop through the density map and apply the clusterfinder to each point depending on the algorithm.
    if algo=="rec-find":
        for x in range(0,a.shape[0]):                                               #TODO:maybe add a step to the for loop to make it faster
            for y in range(0,a.shape[1]):
                if a[x,y]>tresh:
                    a,clsn,_,lst=iscluster(a,x,y,tresh,tresh,[],recl)
                    if clsn==True:
                        clstr+=1
                    if len(lst)>0:
                        clslis.append(lst)
    if algo=="ray-find":
        for x in range(0,a.shape[0]):                                               #TODO:maybe add a step to the for loop to make it faster
            for y in range(0,a.shape[1]):
                if a[x,y]>tresh:
                    a,clsn,_,lst=iscluster_dwn(a,x,y,tresh,tresh,[])
                    if clsn==True:
                        clstr+=1
                    if len(lst)>0:
                        clslis.append(lst)
    else:
        raise Exception("Invalid algorithm")
    clusx=[]
    i=0

    #find the minimum and maximum x and y values for each cluster.
    for cl in clslis:
        clusx.append([10000000,10000000,0,0])
        for clus in cl:
            if clus[0]<clusx[i][0]:
                clusx[i][0]=clus[0]
            if clus[1]<clusx[i][1]:
                clusx[i][1]=clus[1]
            if clus[0]>clusx[i][2]:
                clusx[i][2]=clus[0]
            if clus[1]>clusx[i][3]:
                clusx[i][3]=clus[1]
            
        i+=1

    #calculate the width and height of each cluster.
    for it in range(len(clusx)):
        clusx[it][2]=clusx[it][2]-clusx[it][0]
        clusx[it][3]=clusx[it][3]-clusx[it][1]
    #es. clusx=[minx,miny,dx,dy]

    #check if the cluster is longer than mxlen px, if it is remove it from the list
    i=0
    while i<len(clusx):
        if clusx[i][2]>mxlen or clusx[i][3]>mxlen:
            clslis.pop(i)
            clusx.pop(i)
            i-=1
        i+=1
    
    return clstr,clusx

def iscluster_dwn(density_map,x,y,tresh,lispt):        #without reclvl, maybe implement n future
    
    if density_map[x,y]>tresh:
        density_map[x,y]=0
        lispt.append([x,y])
        x1=x
        if density_map[x,y-1]<tresh:
            density_map,isclsn,reclim,_=iscluster_lft(density_map,x+1,y,tresh,lispt)
        else:
            density_map,isclsn,reclim,_=iscluster_dwn(density_map,x,y-1,tresh,lispt)
        for i in range(-1,2):
            while density_map[x1+i,y]>tresh:
                density_map[x,y]=0
                lispt.append([x,y])
                x1-=1
        return density_map,True,False,lispt
    else:
        return density_map,False,False,[]

def iscluster_lft(density_map,x,y,tresh,lispt):
    if density_map[x,y]>tresh:
        density_map[x,y]=0
        lispt.append([x,y])
        y1=y
        if density_map[x+1,y]<tresh:
            density_map,isclsn,reclim,_=iscluster_dwn(density_map,x,y+1,tresh,lispt)
        else:
            density_map,isclsn,reclim,_=iscluster_lft(density_map,x+1,y,tresh,lispt)
        for i in range(-1,2):
            while density_map[x,y1+i]>tresh:
                density_map[x,y]=0
                lispt.append([x,y])
                y1-=1
        return density_map,True,False,lispt
    
    else:
        return density_map,False,False,[]