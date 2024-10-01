# The most efficient & expensive way... maximize on batch size
# minimize on K
# We extract features from images,
# we define a number of clusters,
# we randomly select images
# if theey have very different cosine similarities,
# we put them into their own cluster
# we then pass the other images
# checking if they fall into any cluster
# we assign images with different cosine similarity into the same cluster
import cv2
import numpy as np
import glob
import os
import shutil

def forward_output(model, images):
    """
    Forward images in path through model to get output
    """

    #create blob
    input_blobs = cv2.dnn.blobFromImages(
        images=images,
        size=(224, 224),  # img target size
        mean=np.array([103.939, 116.779, 123.68]),
        swapRB=True,  # BGR -> RGB
        crop=False  # center crop
    )

    #reshape blob to NHWC and set as input
    reshaped_blobs = input_blobs.transpose((0, 2, 3, 1))
    model.setInput(reshaped_blobs)

    # we will forward to the pooling layer, 126
    # and get (N, 2048) shape output
    # get the layer name
    layers = model.getLayerNames()
    layer_id = 126
    layer_name = layers[layer_id]

    # forward
    output = model.forward(layer_name)
    output = np.float32(output.squeeze())

    return output


# images to pass as model input at a go
batch_size = 50

#load the onnx model
resnet50 = cv2.dnn.readNetFromONNX('../model/resnet50.onnx')

#image paths
image_paths = glob.glob("dad/dad_images/*.jpg", recursive=True)

#Total images to take
total_images = len(image_paths)

print(total_images)

# get features
for take in range(0, total_images, batch_size):
    images = []
    paths = image_paths[take:take+batch_size]

    # read in the images
    for path in paths:
        img = cv2.imread(path)
        img = cv2.resize(img, (256, 256))
        images.append(img)

    output = forward_output(resnet50, images, paths)

    if take == 0:
        feature_map = output[:]
    else:
        feature_map = np.vstack((feature_map, output))

    print("Done: ", take + batch_size)

#save feature map
np.save("feature_map_" + str(total_images), feature_map)