# # Classification with OpenCV dnn

# Now, we are going to carry out a classification task.
# We will start with the ResNet50 model, pre-trained on the ImageNet dataset.
# We will load it in into OpenCV and use it for inferencing. Let's begin.
import cv2
import matplotlib.pyplot as plt
import numpy as np

#load the onnx model
resnet50 = cv2.dnn.readNetFromONNX('model/resnet50.onnx')

# Now that we have loaded the model, we can do various things to
# understand what this model is, and what it does.

layers = resnet50.getLayerNames()
len_str = 30
for layer in layers:
    clean = layer.split('/')[-1]
    cleaner = clean.split('!')[-1]
    print(cleaner.ljust(30), resnet50.getLayerId(layer))


# We can even select one layer and se its outputs. Let us take one convolution layer.
layer_conv_301 = resnet50.getLayer(44)

print(type(layer_conv_301))

params_conv = layer_conv_301.blobs

for param in params_conv:
    print(param.shape)

# # Inferencing

# In order to do inferencing, we will read in the labels, and an image for inferencing.
with open('model/classification_classes_ILSVRC2012.txt', 'r') as labels_file:
    image_net_names = labels_file.read().split('\n')

class_labels = image_net_names[:-1]
print(len(class_labels))


# Let us now load an image to do inferencing.

img = cv2.imread('images/image1.jpg', cv2.IMREAD_COLOR)
plt.imshow(img[:, :, ::-1]); plt.axis('off')


# Now that we have our image, we can pass it through the network.
# To do this, we convert it into a blob using `blobfromImage`.

input_img = img.astype(np.float32)
 
input_img = cv2.resize(input_img, (256, 256))
 
# define preprocess parameters
mean = np.array([103.939, 116.779, 123.68])

# prepare input blob to fit the model input:
# subtract mean
input_blob = cv2.dnn.blobFromImage(
    image=img,
    size=(224, 224),  # img target size
    mean=mean,
    swapRB=True,  # BGR -> RGB
    crop=True  # center crop
)


# The `mean`, `size` and `swapRB` are gotten from the model parameters,
# we do not come up with our own. Also, since the input blob is in NCHW
# order, we transpose it into NHWC to fit our model and then we can take a look at our image.

reshaped_blob = input_blob.transpose((0, 2, 3, 1))
plt.imshow(reshaped_blob.squeeze())


# We can now run our prediction

resnet50.setInput(reshaped_blob)
preds = resnet50.forward()


# Now that we have our predictions, the largest element in the predictions is the value we want. We can extract it.
classified = np.argmax(preds[0])

plt.imshow(img[:, :, ::-1]); plt.title(f'{class_labels[classified]}, {preds[0][classified] * 100:.2f} %')


# ## Extract Features From Final Layer

# We can get outputs from any layer, not necessarily the top layer.

# ## Feature Extraction
img2 = cv2.imread('images/image4.jpg', cv2.IMREAD_COLOR)
plt.imshow(img2[:, :, ::-1]); plt.axis('off')

img3 = cv2.imread('images/image2.jpg', cv2.IMREAD_COLOR)

input_blobs = cv2.dnn.blobFromImages(
    images=[img, img2, img3],
    size=(224, 224),  # img target size
    mean=mean,
    swapRB=True,  # BGR -> RGB
    crop=False  # center crop
)

layer_id = 123
layer_name = layers[layer_id]


reshaped_blobs = input_blobs.transpose((0, 2, 3, 1))
resnet50.setInput(reshaped_blobs)
reshape_output = resnet50.forward(layer_name)

print(resnet50.getLayer(layer_id).type)

img1_pred = reshape_output[0]
img2_pred = reshape_output[1]
img3_pred = reshape_output[2]