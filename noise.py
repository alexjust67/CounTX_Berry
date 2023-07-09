import numpy as np
import matplotlib.pyplot as plt
import torch
import copy
def noise_map_creator(imsize, noise_level, noise_type, density_map,cluster_pos,rad,dropoff=30,density_map_coeff=2,shownoise=False):

    #make a deep copy of the list of cluster positions.
    cluster_po = copy.deepcopy(cluster_pos)

    if noise_type == "gaussian":
        noise = np.random.normal(0, noise_level, (imsize[0], imsize[1]))
    elif noise_type == "poisson":
        noise = np.random.poisson(noise_level, (imsize[0], imsize[1]))
    elif noise_type == "salt and pepper":
        noise = np.random.randint(0, noise_level, (imsize[0], imsize[1]))
    elif noise_type == "speckle":
        noise = np.random.normal(0, noise_level, (imsize[0], imsize[1]))
    elif noise_type == "equal":
        noise = np.full((imsize[0], imsize[1]), 0)
    else:
        raise ValueError("Invalid noise type")
    noise_map = noise
    if noise_type != "equal":
        #create a gaussian around each cluster and add the values to the noise map.
        for cl in cluster_po:      #cl[0] is the x coordinate, cl[1] is the y coordinate, cl[2] is the radius of the cluster.
            cent=[cl[0]+0.5*cl[2],cl[1]+0.5*cl[3]]
            cl[0]=int(cent[0])
            cl[1]=int(cent[1])
            cl[2]=rad
            for x in range(cl[0]-cl[2],cl[0]+cl[2]):
                for y in range(cl[1]-cl[2],cl[1]+cl[2]):
                    if x>=0 and x<density_map.shape[0] and y>=0 and y<density_map.shape[1]:
                        noise_map[x,y]+=(np.exp(-((x-cl[0])**2+(y-cl[1])**2)/(2*(cl[2]/dropoff)**2)))*10           #calculate the value of the 2d gaussian at the point and add it to the noise map.
        noise_map = noise_map.numpy()
    else:
        #create a circle around each cluster and add the values to the noise map.
        for cl in cluster_po:
            cent=[cl[0]+0.5*cl[2],cl[1]+0.5*cl[3]]
            cl[0]=int(cent[0])
            cl[1]=int(cent[1])
            cl[2]=rad
            for x in range(cl[0]-cl[2],cl[0]+cl[2]):
                for y in range(cl[1]-cl[2],cl[1]+cl[2]):
                    if x>=0 and x<density_map.shape[0] and y>=0 and y<density_map.shape[1]:
                        dist=np.sqrt((x-cl[0])**2+(y-cl[1])**2)
                        if dist<cl[2]:
                            noise_map[x,y]+=noise_level
    #noralize the noise map
    if shownoise:
        plt.imshow(noise_map)
        plt.show()
    if noise_type != "equal":
        noise_map = (density_map/density_map_coeff) + noise
        noise_map = (noise_map - torch.min(noise_map)) / (torch.max(noise_map) - torch.min(noise_map))
        #invert the noise map
        noise_map = -noise_map
    else:
        noise_map =-noise_map
        noise_map += np.full((imsize[0], imsize[1]), noise_level)
        noise_map = noise_map*(255/noise_level)
        #clip the values >0 to 0
        noise_map = np.clip(noise_map, 0,250, None)

    
    if shownoise:
        plt.imshow(noise_map)
        plt.show()
    
   

    return noise_map

