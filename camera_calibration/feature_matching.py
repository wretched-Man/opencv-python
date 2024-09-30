# Loading Images
import cv2
import numpy as np
import matplotlib.pyplot as plt

haystack = cv2.imread('images/haystack.jpg')

plt.figure(figsize=[20, 10])
plt.imshow(haystack[:, :, ::-1]); plt.axis('off'); plt.title('Haystack')

needle = cv2.imread('images/needle.jpg')
plt.imshow(needle[:, :, ::-1]); plt.axis('off'); plt.title('Needle')

#create a SIFT object
sift_detector = cv2.SIFT.create(nfeatures=700)

#detect keypoints and descriptors
key1, des1 = sift_detector.detectAndCompute(haystack, None)
key2, des2 = sift_detector.detectAndCompute(needle, None)

#recommended values for SIFT, SURF etc.
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

# creating flann object
flann = cv2.FlannBasedMatcher(index_params, search_params)

#getting knn matches
matches_knn = flann.knnMatch(des1, des2, k=2)

m, n = matches_knn[0]
print(m.distance, n.distance)
#we see that they have the same query index
print(m.queryIdx, n.queryIdx)
print(m.trainIdx, n.trainIdx)

#applying the ratio test to take good matches
good = []
for m, n in matches_knn:
    if m.distance < 0.7*n.distance:
        good.append(m)

src_pts = np.float32( [key1[m.queryIdx].pt for m in good] ).reshape(-1, 1, 2)
dst_pts = np.float32( [key2[m.trainIdx].pt for m in good] ).reshape(-1, 1, 2)

H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

matchesMask = mask.ravel().tolist()

hn, wn = needle.shape[:2]
pts_needle = np.float32([[0, 0], [0, hn-1], [wn-1, hn-1], [wn-1, 0]]).reshape(-1, 1, 2)

pts_haystack = cv2.perspectiveTransform(pts_needle, np.linalg.inv(H))

new_haystack = haystack.copy()
new_haystack = cv2.polylines(new_haystack, [np.int32(pts_haystack)], True, (255, 255, 255), 3, cv2.LINE_AA)

plt.figure(figsize=[20, 10])
plt.imshow(new_haystack[:, :, ::-1]); plt.axis('off')

drawn_final = cv2.drawMatches(new_haystack, key1, needle, key2, good, None, matchColor=(0, 255, 0),matchesMask=matchesMask)

plt.figure(figsize=[20, 10])
plt.imshow(drawn_final[:, :, ::-1])

