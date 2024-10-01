import numpy as np
import cv2


files = ['data_2000.xml', 'data_4000.xml', 'data_7000.xml', 'data_final.xml']
sizes = ['2000', '4000', '7000', '8636']
all_files = np.zeros((1, 2048), np.float32)
for i,file in enumerate(files):
    ifile = cv2.FileStorage(file, cv2.FileStorage_READ)
    node = ifile.getFirstTopLevelNode()
    name = node.name()
    pos_start = int(name.split('_')[-1]) + 50
    
    mfile = np.squeeze(node.mat())
    #iterate through the nodes
    for curr_name in range(pos_start, int(sizes[i]), 50):
        name = '_' + str(curr_name)
        t_node = ifile.getNode(name)
        mfile = np.vstack((mfile, np.squeeze(t_node.mat())))

    all_files = np.vstack((all_files, mfile))

my_files = all_files[1:]

data = np.load('../complete/super_feature_map.npy')
#print('loaded data', data.shape)

final = np.vstack((data, my_files))
#print('final', final.shape)

np.save('super_super_feature_map', final)
