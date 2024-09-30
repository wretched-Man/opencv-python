# Loading the Images
import cv2
import numpy as np
import matplotlib.pyplot as plt

template = cv2.imread('images/form.jpg')
scanned = cv2.imread('images/scanned-form.jpg')

# Display the images. 
plt.figure(figsize = [20, 10])
plt.subplot(121); plt.axis('off'); plt.imshow(template[:, :, ::-1]); plt.title("Original Form")
plt.subplot(122); plt.axis('off'); plt.imshow(scanned[:, :, ::-1]); plt.title("Scanned Form");

# we will use ORB
#creating an ORB object
orb_object = cv2.ORB_create(nfeatures=600)

keypoints1, descriptors1 = orb_object.detectAndCompute(template, None)
keypoints2, descriptors2 = orb_object.detectAndCompute(scanned, None)

#We can drawn the keypoints
template_keypoints = cv2.drawKeypoints(template, keypoints1, None, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
scanned_keypoints = cv2.drawKeypoints(scanned, keypoints2, None, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display the keypoint images. 
plt.figure(figsize = [20, 10])
plt.subplot(121); plt.axis('off'); plt.imshow(template_keypoints[:, :, ::-1]); plt.title("Original Form KP")
plt.subplot(122); plt.axis('off'); plt.imshow(scanned_keypoints[:, :, ::-1]); plt.title("Scanned Form KP");

# creating the object
matcher = cv2.BFMatcher.create(cv2.NORM_HAMMING, True)

#using our matcher object to match
matches_normal = matcher.match(descriptors1, descriptors2)

matches_normal = sorted(matches_normal, key=lambda x:x.distance, reverse=False)
matches_normal = matches_normal[:int(len(matches_normal) * .1)]

drawn_matches = cv2.drawMatches(
            template,
            keypoints1,
            scanned,
            keypoints2,
            matches_normal,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(drawn_matches[:, :, ::-1])

# Extract the location of good matches.
points1 = np.zeros((len(matches_normal), 2), dtype = np.float32)
points2 = np.zeros((len(matches_normal), 2), dtype = np.float32)

for i, match in enumerate(matches_normal):
    points1[i] = keypoints1[match.queryIdx].pt
    points2[i] = keypoints2[match.trainIdx].pt

h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
height, width = template.shape[:2]
scanned_restored = cv2.warpPerspective(scanned, h, (width, height))

plt.imshow(scanned_restored[:, :, ::-1])

