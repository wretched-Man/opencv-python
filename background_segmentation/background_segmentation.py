#!/usr/bin/env python
# coding: utf-8

# # Using background segmentation for motion detection in videos

# We are going to be using OpenCV to perform motion detection in a video.
# For this, we are going to use a technique called background subtraction.
# This is a technique in which we will compute a background for a video
# from a series of frames and then using that background, we will subtract
# changes in subsequent frames and make a mask from changes in subsequent
# frames. The non-zero pixels in the resultant mask identify the places
# where there is motion. We will then draw a bounding rectangle on the
# image to show these changes.

# In[ ]:


import cv2
import numpy as np


# We will first create the video capture and video writer objects. Then,
# we will create the foreground masking object. The output video will contain
# the mask and the frame at that point. Let us first preview our video.

# In[ ]:


input_video = 'media/motion_test.mp4'


# In[ ]:


from moviepy.editor import VideoFileClip

# Load the video for playback. 
#clip = VideoFileClip(input_video)
#clip.ipython_display(width = 800)


# In[ ]:


#creating video capture objects
video_cap = cv2.VideoCapture(input_video)


# In[ ]:


#Get total frames
framecount = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
#We create the framesize
#Since our video will be two frames combined, it will
#have a width of 2X the original
frameheight = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
framewidth = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 2)


# ## Without erosion

# In[ ]:


#Creating a video writer
video_writer = cv2.VideoWriter('motion_test_no_erode.mp4',\
                               int(video_cap.get(cv2.CAP_PROP_FOURCC)),\
                               video_cap.get(cv2.CAP_PROP_FPS),\
                              (framewidth, frameheight))


# We will also create a helper function to annotate every frame of
# the output file with the frame number.

# In[ ]:


#Helper annotate function
#It will take a frame and text and write it onto a file
def markFrame(frame, text):
    cv2.putText(frame, text, (0, 15), 2, 0.6, (0, 255, 0), 1)
    return frame


# Now to create the foreground mask object. OpenCv provides multiple
# algorithms for this which can be found at
# (https://docs.opencv.org/4.x/d8/d38/tutorial_bgsegm_bg_subtraction.html).
# We will be using `BackgroundSubtractorMOG2` to create the foreground mask.

# In[ ]:


bgobj = cv2.createBackgroundSubtractorMOG2(history=200)


# In[ ]:


#We now iterate through every frame to create a mask
count = 0 #count no of frames
while True:
    ret, frame = video_cap.read()

    #Nothing to read
    if frame is None:
        break
    count += 1

    #Create a mask
    fgmask = bgobj.apply(frame)

    #We want to create a bounding rectangle on the moving
    #portions of the image
    #We find the coordinates of all non-zero pixels ...
    #these represent where the images have changed
    nonzero = cv2.findNonZero(fgmask)

    #We will then create a rectangle that bounds all the
    #points. We will use these points to draw our bounding
    #rectangle
    x1, y1, x2, y2 = cv2.boundingRect(nonzero)
    
    #We will concantenate the mask and frame to
    #We mark both frames
    #the original frame
    origmarked = markFrame(frame, 'Frame: ' + str(count)\
                             + '/' + str(framecount))
    
    #for the mask, we make it 3 channel so as to mark it
    color_mask = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
    
    #marking the mask
    fg_marked = markFrame(color_mask, 'Frame: ' +\
                    str(count) + '/' + str(framecount))

    #We will now draw a bounding rectangle over the color image
    cv2.rectangle(origmarked, (x1, y1), (x2, y2),\
                  (0, 255, 255), 4)
    #We now concantenate the images horizontally
    finalframe = np.hstack([fg_marked, origmarked])

    #We will draw a dividing line between the two frames
    cv2.line(finalframe, (int(framewidth/2), 0),\
                (int(framewidth/2), frameheight),\
                 (255, 255, 0), 5)

    video_writer.write(finalframe)

video_writer.release()
video_cap.release()


# We are now going to open our file and see the final output video.

# In[ ]:


# Load the video for playback. 
#clip_2 = VideoFileClip('motion_test_no_erode.mp4')
#clip_2.ipython_display(width = 800)


# As we have seen in the video, there are bounding boxes in every frame
# (the bounding box is shown in yellow) even when there is no motion.
# We can solve this by using erosion in the image. Let us try again this
# time using erosion. 

# ## Using erosion

# In[ ]:


video_writer_erode = cv2.VideoWriter('motion_test_erode.mp4',\
                               int(video_cap.get(cv2.CAP_PROP_FOURCC)),\
                               video_cap.get(cv2.CAP_PROP_FPS),\
                              (framewidth, frameheight))


# In[ ]:


#We now iterate through every frame to create a mask
count = 0 #count no of frames
while True:
    ret, frame = video_cap.read()

    #Nothing to read
    if frame is None:
        break
    count += 1

    #Create a mask
    fgmask = bgobj.apply(frame)

    #We want to create a bounding rectangle on the moving
    #portions of the image
    #We find the coordinates of all non-zero pixels ...
    #these represent where the images have changed

    #we now erode, the only change we make
    #create a kernel
    kernel = np.ones((5,5),np.uint8)
    fgmask_erode = cv2.erode(fgmask, kernel)
    nonzero = cv2.findNonZero(fgmask_erode)

    #We will then create a rectangle that bounds all the
    #points. We will use these points to draw our bounding
    #rectangle
    x1, y1, x2, y2 = cv2.boundingRect(nonzero)
    
    #We will concantenate the mask and frame to
    #We mark both frames
    #the original frame
    origmarked = markFrame(frame, 'Frame: ' + str(count)\
                             + '/' + str(framecount))
    
    #for the mask, we make it 3 channel so as to mark it
    color_mask = cv2.cvtColor(fgmask_erode, cv2.COLOR_GRAY2BGR)
    
    #marking the mask
    fg_marked = markFrame(color_mask, 'Frame: ' +\
                    str(count) + '/' + str(framecount))

    #We will now draw a bounding rectangle over the color image
    if nonzero is not None:
        cv2.rectangle(origmarked, (x1, y1), (x2, y2),\
                      (0, 255, 255), 4)
    #We now concantenate the images horizontally
    finalframe = np.hstack([fg_marked, origmarked])

    #We will draw a dividing line between the two frames
    cv2.line(finalframe, (int(framewidth/2), 0),\
                (int(framewidth/2), frameheight),\
                 (255, 255, 0), 5)

    video_writer_erode.write(finalframe)

video_writer_erode.release()
video_cap.release()


# In[ ]:


# Load the video for playback. 
#clip_2 = VideoFileClip('motion_test_erode.mp4')
#clip_2.ipython_display(width = 800)


# In this second try, we see that the number of false positives has
# reduced. This is because we have applied erosion to the image.

# We have now seen how to do motion detection in OpenCV.
