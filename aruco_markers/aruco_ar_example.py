# # Understanding Aruco Markers

# ArUco markers are square black and white images with a unique id that are
# used to mark 3D planes coplanar points. Since they are unique, they can
# easily map 3D coplanar points into 2D images. They are used for
# Augmented Reality, Camera Calibration and pose estimation among others.

# OpenCV provides the ArUco module with which we can draw,
# detect and draw detected markers.
# 
# As stated, each AruCo marker is unique as to the pattern.
# Markers with the same internal size are grouped into a dictionary.
# ArUco dictionaries contain a number of squares that have the same internal size.
# 
# OpenCV provides predefined dictionaries such as DICT_6x6_250,
# where 6x6 is the marker size in bits and 250 is the number of
# markers in the dictionary. Each ArUco marker has a unique ID starting from 0 to N-1.

# ### Loading and displaying ArUco Markers

# Let us load and display a few ArUco markers.
# We will use OpenCV's `getPredefinedDictionary` method to load a dictionary.

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'gray'

print([m for m in dir(cv2.aruco) if 'DICT' in m])

# Some of the existing dictionaries. More information can be found at
# (https://docs.opencv.org/4.x/de/d67/group__objdetect__aruco.html#gga4e13135a118f497c6172311d601ce00da6235dfb8007de53d3e8de11bee1e3854).

#let us load one dictionary
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_250)

# We can now generate a number of markers from these dictionaries and
# display them. We will use the `generateImageMarker` method.

markers = [10, 197, 60]
aruco_markers = []
for marker in markers:
    aruco_markers.append(dictionary.generateImageMarker(marker, 200))

print(aruco_markers[0].shape)

plt.figure(figsize=[10, 10])
plt.subplot(131); plt.imshow(aruco_markers[0]); plt.axis('off')
plt.subplot(132); plt.imshow(aruco_markers[1]); plt.axis('off')
plt.subplot(133); plt.imshow(aruco_markers[2]); plt.axis('off')


# We have displayed 3 images taken from DICT_7X7_250 with a size of 7x7
# boxes. We have scaled the output to 200x200 size.
# 
# Normally, the images are then saved and printed to be placed in real-world
# objects. These markers can then be used to define the ROI of an image.
# This is done by detecting a corner of an image.

# ## AR using ArUco Markers

# We are going to use ArUco Markers to create an AR application.
# 
# The steps are as follows:
# * Extract corner points of ArUco markers from image.
# * Determine ROI from corner points and scale ROI
# * Determine Source Image points
# * Warp Source into ROI shape
# * Create Source Mask
# * Add Source into initial image

frame = cv2.imread('images/office_markers.jpg')

plt.figure(figsize=[10, 10])
plt.imshow(frame[:, :, ::-1]); plt.axis('off')

# We are now going to detect the aruco markers. For that, we need to know
# the dictionary used. For this one, we know that the dictionary is 6x6_250,
# hence we start by loading it in.

dict6 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
corners, ids, rejects = cv2.aruco.detectMarkers(frame, dict6)

# The `detectMarkers` method returns the 4 corners of each ArUco object,
# the ids of each marker and any rejected would-be points. We can visualize
# this output by drawing using the method `drawDetectedMarkers`.

#the algorithm draws directly on the image
frame_drawn = frame.copy()
cv2.aruco.drawDetectedMarkers(frame_drawn, corners, ids)

plt.figure(figsize=[20, 20])
plt.imshow(frame_drawn[:, :, ::-1]); plt.axis('off')


# The size of the image has been enhanced greatly so as to show a few details.
# Namely, the top left corner is marked in a red square. Also the marker is
# highlighted with a green line all round and the ID is written in blue. We
# can also see that there is a white area surrounding the markers that we
# may need to account for.
# 
# Now that we have the corner points, we can establish our ROI which will
# be the corner of every marker.

new_ids = np.squeeze(ids)

ids_corners = list(zip(new_ids, corners))
ids_corners.sort(key = lambda x:x[0])

pts_dst = []
for i, elem in enumerate(ids_corners):
    pts_dst.append(np.squeeze(elem[1])[i])


# Now would be a good time to scale the images.

scale_num = 4
scale_array = np.array([[-scale_num, -scale_num], [scale_num, -scale_num], [scale_num, scale_num], [-scale_num, scale_num]])

pts_dst_m = np.array([np.float32(m+n) for m,n in zip(pts_dst, scale_array)])

# ### Define Source Points

#load source image
source = cv2.imread('images/Apollo-8-Launch.png', cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=[8, 8])
plt.imshow(source)

h, w = source.shape
src_pts = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])

H = cv2.getPerspectiveTransform(src_pts, pts_dst_m)


# Now that we have the homography relating the two points,
# we need to warp the source image into the destination image.

#find size of output image, using marker distance
size_width = int(np.linalg.norm(pts_dst_m[0] - pts_dst_m[1]))
size_height = int(np.linalg.norm(pts_dst_m[0] - pts_dst_m[3]))

dst_apollo = cv2.warpPerspective(source, H, (frame.shape[1], frame.shape[0]))
plt.imshow(dst_apollo)


# The next step is adding the two images. We do this by masking out the
# portion in the destination image that the source image is to be added into.


#get the transformation of the points
#src_pts_m = src_pts.reshape((-1, 1, 2))
#corner_pts = cv2.perspectiveTransform(src_pts_m, H)
frame_dst_final = cv2.fillPoly(frame, [np.int32(pts_dst_m)], (0, 0, 0))
plt.imshow(frame_dst_final)


# We now add the points.
dst_apollo_col = cv2.merge([dst_apollo, dst_apollo, dst_apollo])
final_img = cv2.add(frame_dst_final, dst_apollo_col)
plt.figure(figsize=[20, 20])
plt.imshow(final_img); plt.axis('off')