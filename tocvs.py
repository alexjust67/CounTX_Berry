import pandas as pd
import numpy as np
 
density_map=np.load("./img/results/density_map.npy")


# convert array into dataframe
DF = pd.DataFrame(density_map)
 
# save the dataframe as a csv file
DF.to_csv("data1.csv")