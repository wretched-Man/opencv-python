# Background Segmentation
In this lesson, I learn about background segmentation, essentially how to separate the foreground from a static background.

I show how to create a foreground mask using one of the many OpenCV's `BackgroundSubtractor` classes, in this case MOG2.

Also, I show the essense of using erosion on the foreground mask to remove noise and prevent the detection of false positives from the image. For this,
I produce two separate result videos, where, in one, the background noise is removed and in another there is noise in the foreground mask. This, so as to show the essense of the extra step 
of removing noise from the foreground mask.

**Note:** Because of the size of the media used in this lesson, I have seen it proper to store the necessary video used in a separate location.

Use-cases of this include intruder detection for surveillance and motion detection in videos.
