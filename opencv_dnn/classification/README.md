# Classification with OpenCV DNN
In this lesson, I used a preexisting model to perform classification of images. I used the resnet50 model pretrained on Imagenet dataset to classify Images.

## Project
Under the project folder, I used the Resnet model to extract features from images. I then used the `kmeans` function in OpenCV to classify the images based on the simlarity of the features.
The `merge_images.py` and `merge_feature_map.py` are utility functions for combining images sourced from different places together and merging feature maps (since inference was done separately), respectively.

### C++
The inferencing bit for feature extraction took quite a considerable time to run. I imagined I could get a considerable time boost if I rewrote it in C++. And yes, I did! The output of the same was saved in separate xml files which were later merged. The residue of this exists in the `merge_feature_map.py` file. I have also posted my C++ code here. As you can imagine, the feature map and the set of images, I haven't uploaded.
