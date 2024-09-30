# # Hough Transform for Lane Detection

# We will use Hough Transform to detect lines in an image.

# Photo by 李进: [Link](https://www.pexels.com/photo/scenic-photo-of-wooden-dock-during-dawn-2903939/)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random


# ## Example 1

# We are going to try an example image.

bridge = cv2.imread("images/bridge_lanes.jpg")
bridge_gray = cv2.cvtColor(bridge, cv2.COLOR_BGR2GRAY)
plt.imshow(bridge_gray, cmap='gray')

bridge_blur = cv2.GaussianBlur(bridge_gray, (7, 7), 18)
plt.imshow(bridge_blur, cmap='gray')

bridge_edges = cv2.Canny(bridge_blur, 50, 65)
plt.imshow(bridge_edges, cmap='gray')

bridge_lines = cv2.HoughLinesP(bridge_edges, 1, np.pi/180, 10, minLineLength=15, maxLineGap=5)

#We will draw our lines here
bridge_drawn = np.zeros_like(bridge)

#We now need to segment out the flat regions, whose gradients are small
def grad(x1, y1, x2, y2):
    point1 = x2 - x1
    point2 = y2 - y1
    if point1 == 0:
        return 0
    if point2 == 0:
        return 0
    return point2 / point1

for line in bridge_lines:
    for x1, y1, x2, y2 in line:
        line_gradient = grad(x1, y1, x2, y2)
        if np.abs(line_gradient) > .2 and np.abs(line_gradient) < 1:
            cv2.line(bridge_drawn, (x1, y1), (x2, y2), (255, 255, 0), thickness = 3)


plt.imshow(bridge_drawn, cmap='gray')

plt.imshow(bridge_drawn[720:,:950])

def draw_lines(img, lines, color = [255, 0, 0], thickness = 2):
    """Utility for drawing lines."""
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def separate_lines(lines, centre = 950):
    "Segment into either left or right."
    left = []
    right = []
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            line_gradient = grad(x1, y1, x2, y2)
            if np.abs(line_gradient) > .2 and np.abs(line_gradient) < 1:
                if line_gradient < 0 and x2 < centre:
                    left.append(line)
                elif line_gradient > 0 and x2 > centre:
                    right.append(line)

    return left, right

left_lines, right_lines = separate_lines(bridge_lines)
#well_segmented lines
seg_lines = np.zeros_like(bridge)
draw_lines(seg_lines, left_lines, color = [255, 255, 0], thickness = 3)
draw_lines(seg_lines, right_lines, color = [255, 255, 0], thickness = 3)
plt.imshow(seg_lines)

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

final_lanes = np.zeros_like(bridge)

extrapolate_left = extrapolate_lanes(left_lines, 720, 1200)
extrapolate_right = extrapolate_lanes(right_lines, 720, bridge.shape[0]-1)

draw_lines(final_lanes, [[extrapolate_left]], color = [255, 255, 0], thickness = 10)
draw_lines(final_lanes, [[extrapolate_right]], color = [255, 255, 0], thickness = 10)
plt.imshow(final_lanes)

resulting_img = cv2.addWeighted(bridge, 1, final_lanes, 0.6, 0)
plt.imshow(resulting_img[:, :, ::-1])


# ## Example 2

# We can now list down the actions to come up with a segmentation for the images.
# 1. Take an image.
# 2. Convert the image to grayscale.
# 3. Find the appropriate edges using Canny's
# 4. Use the Hough transform to find lines
# 5. Segment the lines into left or right depending on gradient
# 6. Extrapolate the lines in each of the right and left lanes to find one line that runs the whole length
# 7. Draw the final line on the original image

road = cv2.imread("images/road.jpg")
plt.imshow(road[:, :, ::-1])

road_gray = cv2.cvtColor(road, cv2.COLOR_BGR2GRAY)
road_blur = cv2.GaussianBlur(road_gray, (7, 7), 4)
road_edges = cv2.Canny(road_blur, 50, 100)
plt.imshow(road_edges, cmap='gray')

#We will now find lines
lines = cv2.HoughLinesP(road_edges, 1, np.pi/180, 10, minLineLength=6, maxLineGap=5)

road_lines = np.zeros_like(road)
clean_lines = []

#draw the lines
# Only take the necessary lines
for line in lines:
    for x1, y1, x2, y2 in line:
        line_gradient = grad(x1, y1, x2, y2)
        if np.abs(line_gradient) > .2 and np.abs(line_gradient) < 1:
            clean_lines.append(line)

draw_lines(road_lines, clean_lines, thickness=3)
plt.imshow(road_lines)

#segment left-right
left_lines_road, right_lines_road = separate_lines(clean_lines, 480)
separated_lines_image = np.zeros_like(road)

#draw segmented lines
draw_lines(separated_lines_image, left_lines_road)
draw_lines(separated_lines_image, right_lines_road)
plt.imshow(separated_lines_image)

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

def gradient_cluster(lines, difference = 0.5):
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

grouped_left_lanes = gradient_cluster(left_lines_road, 0.05)
grouped_right_lanes = gradient_cluster(right_lines_road, 0.001)

final_lanes_road = np.zeros_like(road)

extra_left = extrapolate_lanes(grouped_left_lanes[0], 310, road.shape[0])
extra_right = extrapolate_lanes(grouped_right_lanes[-1], 310, road.shape[0])

draw_lines(final_lanes_road, [[extra_left]], color = [255, 255, 0], thickness = 10)
draw_lines(final_lanes_road, [[extra_right]], color = [255, 255, 0], thickness = 10)
plt.imshow(final_lanes_road)

final_road_colored = cv2.addWeighted(road, 1, final_lanes_road, 0.6, 0)
plt.imshow(final_road_colored[:, :, ::-1])