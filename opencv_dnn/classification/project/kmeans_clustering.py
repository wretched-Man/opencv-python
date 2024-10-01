import cv2
import numpy as np
import glob
import os
import shutil

# define KMEANS constants
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 6
flags = cv2.KMEANS_PP_CENTERS

#data
data = np.load("complete/super_feature_map.npy")[5971:]
#image paths
image_paths = glob.glob("complete/images/*", recursive=True)[5971:]

#clustering
ret, label, center = cv2.kmeans(data,K,None,criteria,10,flags)
label = label.ravel()

print(ret)

#writing to output
for means in range(K):
    indices, = np.where(label == means)

    #write only the most compact data
    #if #indices.size < 600:
        #continue

    path = "results/" + str(means+1)
    #check if a folder exists at the location
    #if not create
    if os.path.exists(path):
        continue
    else:
        os.makedirs(path)

    path += '/'

    for count, index in enumerate(indices):
        file_source = image_paths[index]
        shutil.copy2(file_source, path)