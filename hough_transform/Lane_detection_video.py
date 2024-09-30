# # Hough Transform for Video

# We will now use the techniques created in the single-frame example to do lane detection for a video. Our pipeline:
# 1. We redefine the methods for single frame method.
# 2. We create a pipeline for every single frame.
# 3. We load the video and apply the pipeline to every image.

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

plt.rcParams['image.cmap'] = 'gray'

def grad(x1, y1, x2, y2):
    """Finds line gradient from the points"""
    point1 = x2 - x1
    point2 = y2 - y1
    if point1 == 0:
        return 0
    if point2 == 0:
        return 0
    return point2 / point1

def draw_lines(img, lines, color = [255, 0, 0], thickness = 2):
    """Utility for drawing lines."""
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def separate_lines(lines, centre):
    """Segment lines into either left or right."""
    left = []
    right = []
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            line_gradient = grad(x1, y1, x2, y2)
            #threshold lines first
            if np.abs(line_gradient) > .2 and np.abs(line_gradient) < 1:
                if line_gradient < 0 and x2 < centre:
                    left.append(line)
                elif line_gradient > 0 and x2 > centre:
                    right.append(line)
    return left, right

#calculate averages
def average(data):
    if data is not None:
        n = 1
        if len(data) > 0:
            n = len(data)
    return sum(data)/n

# Since the lines break apart, we extrapolate the lines by
# finding the average slope and line and extending it from
# the lower border to upper border
def extrapolate_lanes(lines, upper_border, lower_border):
    slopes = []
    consts = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            line_gradient = grad(x1, y1, x2, y2)
            slopes.append(line_gradient)
            const = y1 - line_gradient * x1
            consts.append(const)

    avg_slopes = average(slopes)
    avg_consts = average(consts)

    # Calculate average intersection at lower_border.
    x_lane_lower_point = int((lower_border - avg_consts) / avg_slopes)
    
    # Calculate average intersection at upper_border.
    x_lane_upper_point = int((upper_border - avg_consts) / avg_slopes)
    return [x_lane_lower_point, lower_border, x_lane_upper_point, upper_border]

def sort_lines_gradient(lines):
    "Sort lines according to gradient."
    if len(lines) < 2:
        return lines

    all_gradients = []
    sorted_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            line_gradient = grad(x1, y1, x2, y2)
            all_gradients.append(line_gradient)

    sorted_gradients = np.argsort(all_gradients)
    for pos in range(len(lines)):
        sorted_lines.append(lines[sorted_gradients[pos]])

    all_gradients.sort()
    return sorted_lines, all_gradients

def gradient_cluster(lines, difference = 0.1):
    """
    Group into clusters.
    difference tells us the max difference in a cluster
    """
    if len(lines) < 2:
        return lines

    final_lines = []
    #sort lines
    sorted_lines, sorted_grads = sort_lines_gradient(lines)
    ungrouped_lines = sorted_lines.copy()
    ungrouped_grads = sorted_grads.copy()

    while len(ungrouped_lines) > 1:
        grouped_lines = []
        grouped_lines.append(ungrouped_lines[0])
        
        for _ in range(1, len(ungrouped_lines)):
            if abs(ungrouped_grads[0] - ungrouped_grads[1]) > difference:
                break
            else:
                grouped_lines.append(ungrouped_lines[1])
                ungrouped_lines.pop(1)
                ungrouped_grads.pop(1)

        ungrouped_lines.pop(0)
        ungrouped_grads.pop(0)
        
        final_lines.append(grouped_lines)
        # sort from smallest to largest group
        final_lines.sort(key = len)
    return final_lines

def appropriate_line(ungrouped, high, low, sensitivity, pos=0):
    """
    Given a list of ungrouped left or right lines,
    group them and extrapolate each group. Find the
    lines that are best fit for the image.
    high, low are y-value extrapolation points
    pos - whether left or right lanes
    """
    grouped_lanes = gradient_cluster(ungrouped, sensitivity)
    
    extrapolated_lines = []
    for group in grouped_lanes:
        extrapolated_lines.append(extrapolate_lanes(group, int(high), int(low)))

    # We have extrapolated every line in the groups
    # we now choose the best fit. For left lines, this
    # is one where the x1 and x2 values are largest
    
    #easier to handle
    lines = np.array(extrapolated_lines, np.int32)
    sorted_lines = lines[np.argsort(lines[:, 0])]
    if pos == 0:
        #left side
        return(list(sorted_lines[0]))
    else:
        return(list(sorted_lines[-1]))       

def draw_con(img, lines):
    """Fill in lane area."""
    points = []
    for x1,y1,x2,y2 in lines[0]:
        points.append([x1,y1])
        points.append([x2,y2])
    for x1,y1,x2,y2 in lines[1]:
        points.append([x2,y2])
        points.append([x1,y1])
    
    points = np.array([points], dtype = 'int32')
    final_img = cv2.fillPoly(img, points, (0,255,0))
    return final_img

def process_image(frame):
    """
    Applies all the methods above on each frame.
    Returns a frame with the lane area drawn.
    """
    if frame.shape[2] != 3:
        return None

    #convert to gray and blur
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_blur = cv2.GaussianBlur(frame_gray, (7, 7), 4)
    #detect edges and lines using HoughLinesP
    frame_edges = cv2.Canny(frame_blur, 50, 100)

    lines = cv2.HoughLinesP(frame_edges, 1, np.pi/180, 10, minLineLength=6, maxLineGap=5)

    #remove lines that do not fit into a ccertain gradient
    # sort lines to left and right
    left_lines, right_lines = separate_lines(lines, frame.shape[1]//2)

    #choosing line to extrapolate
    ideal_left = appropriate_line(left_lines, int(frame.shape[0]*.58), frame.shape[0]-1, 0.05, -1)
    ideal_right = appropriate_line(right_lines, int(frame.shape[0]*.58), frame.shape[0]-1, 0.01)

    #draw on final frame
    final_frame = draw_con(np.zeros_like(frame), [[ideal_left], [ideal_right]])
    return final_frame

# main processing image
cap = cv2.VideoCapture('images/lane1-straight.mp4')

if not cap.isOpened():
    print('Error opening file')

else:
    #get video information
    no_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    #videowriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writecap = cv2.VideoWriter('images/lane1-marked.mp4', fourcc, frame_rate, (width, height))
    if not writecap.isOpened():
        print('Error opening write stream')

    while True:
        retval, frame = cap.read()
        if retval == False:
            print("End of video.")
            break
        else:
            img = process_image(frame)
            final = cv2.addWeighted(frame, 1, img, .4, 0)
            writecap.write(final)

    writecap.release()