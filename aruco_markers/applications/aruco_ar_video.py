# Using ArUco Markers to input a video
# into another video

# We read in every frame from both videos
# We morph the frames of the source video
# into the destination video

import cv2
import numpy as np

# define convenience function
# this function takes a frame, finds the aruco
# markers, scales and returns them
def find_markers(frame, dictionary):
    corners, ids, rejects = cv2.aruco.detectMarkers(frame, dictionary)
    ids = np.squeeze(ids)
    ids_corners = list(zip(ids, corners))
    # taking the corner points
    ids_corners.sort(key = lambda x:x[0])

    dst = []
    for i, elem in enumerate(ids_corners):
        dst.append( np.float32(np.squeeze(elem[1])[i]) )

    # scaling
    return np.float32(dst)

# the file that we will put the video into
dst_vid_file = 'media/office_markers.mp4'

# the file that we will morph into the dest file
# Video by olia danilevich:
# https://www.pexels.com/video/father-and-child-having-happy-moments-4625207/

src_vid_file = 'media/race_car_slow_motion.mp4'

# the final file we will put the morphed video
final_vid_file = 'media/final.mp4'

# videoCapture objects
cap_src = cv2.VideoCapture(src_vid_file)
cap_dst = cv2.VideoCapture(dst_vid_file)

# check if file is open
if not cap_src.isOpened() or not cap_dst.isOpened():
    print("Video files could not be opened!")
    exit()

# output video properties
fps = cap_dst.get(cv2.CAP_PROP_FPS)
fourcc = int(cap_dst.get(cv2.CAP_PROP_FOURCC))
width = int(cap_dst.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap_dst.get(cv2.CAP_PROP_FRAME_HEIGHT))
framesize = (width, height)

# output videoWriter object
wrt_fin = cv2.VideoWriter(final_vid_file,\
    fourcc, fps, framesize)


# since we a warping the whole source image,
# the source points are constant, we can define
# them here.
# source frame size
s_width = int(cap_src.get(cv2.CAP_PROP_FRAME_WIDTH))
s_height = int(cap_src.get(cv2.CAP_PROP_FRAME_HEIGHT))
src_pts = np.float32([[0, 0],
                      [s_width-1, 0],
                      [s_width-1, s_height-1],
                      [0, s_height-1]])

# ArUco dictionary
dictionary = cv2.aruco.getPredefinedDictionary(\
                        cv2.aruco.DICT_6X6_250)

while True:
    src_ret, src_frame = cap_src.read()
    dst_ret, dst_frame = cap_dst.read()

    if not src_ret or not dst_ret:
        break

    # find the markers in the dst image
    dst_pts = find_markers(dst_frame, dictionary)

    # find the transform between the two points
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # we warp the source image
    s_warped = cv2.warpPerspective(src_frame, H, framesize)

    # create mask
    masked_dst = cv2.fillPoly(dst_frame, np.int32([dst_pts]), (0, 0, 0))

    # add to make final image
    final_img = cv2.add(masked_dst, s_warped)

    #write frame
    wrt_fin.write(final_img)

#release the video capture
cap_src.release()
cap_dst.release()
#release the video writer
wrt_fin.release()

