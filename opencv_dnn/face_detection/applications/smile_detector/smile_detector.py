import cv2
import os
import numpy as np

# We will use facial landmarks to detect a smile in a video
# Draw and label the facial landmarks


def detectfaces(net, image, thresh_confidence = .70):
    """
    Detects faces in the image and returns a list of Rect objects that
    define the position of each face in the image given
    """
    # create a blob from the image
    blob = cv2.dnn.blobFromImage(image,
                                scalefactor=1.0,
                                size=(300,300),
                                mean=(104.0, 177.0, 123.0))
    
    # we set it as input to our model
    net.setInput(blob)
    detections = net.forward()

    faces = []

    h, w = image.shape[:2]

    for face in range(detections.shape[2]):
        #get confidence
        confidence = detections[0, 0, face, 2]

        if confidence >= thresh_confidence:
            x1, y1, x2, y2 = (detections[0, 0, face, 3:7] * np.array([w, h, w, h])).astype(int)
            # ensure that the face is within bounds
            if (x1 >= 0) and (x2 < w):
                if  (y1 >= 0) and (y2 < h):
                    width = x2 - x1
                    height = y2 - y1
                    box = np.array((x1, y1, width, height))
                    faces.append(box)
        else:
            continue       
    return np.array(faces)


def finddraw_landmarks(faces, image, facemark):
    """
    Given an image and a list of faces, fit the facemark model
    to the faces and draw the images.

    Returns the image, drawn
    """
    # assume the facemark model
    # facemark = cv2.face.createFacemarkLBN()
    # facemark.loadmodel('')
    retval, face_landmarks = facemark.fit(image, faces)

    if retval == True:
        # IF landmarks were detected
        for landmarks in face_landmarks:
            # Draw the landmarks
            cv2.face.drawFacemarks(image, landmarks, (255, 255, 0))

            for idx in range(landmarks.shape[1]):
                # Annotate the landmarks
                 cv2.putText(image, str(idx), (landmarks[0][idx]).astype(int),\
                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    else:
        print("No facial landmarks were detected.")


def calculate_distance(point1, point2):
    """
    Calculate the Eucledian distance between two points.
    """
    dist = np.linalg.norm(point1 - point2)
    return dist


def smile_detector(image):
    """
    Given an image, detect whether there is a smiling
    face or not.

    Returns True/False and frame if true, None
    """
    faces = detectfaces(net, image, .80)
    number_of_faces = faces.shape[0]

    # hold the face coordinates of the face that is seen to be smiling
    face_coords = []

    if number_of_faces > 0:
        print(number_of_faces)
        # define constants
        zero_const = np.array([0.19591522, 0.4255486])
        one_const = np.array([0.20351438, 0.43752506])
        two_const = np.array([0.20363207, 0.4333129])

        retval, face_landmarks = facemark.fit(image, faces)

        if retval is True:
            for pos, landmark in enumerate(face_landmarks):
                # 3 values for each threshold, at least 2 must
                # be true for the face to be considered smiling
                is_smile = np.zeros((3,), np.uint8)

                # get the ratios
                landmark = np.squeeze(landmark)
                # first calculate (60 - 64), the denominator of the ratio
                denom = calculate_distance(landmark[60], landmark[64])
                # point 61 - 67
                zero = calculate_distance(landmark[61], landmark[67]) / denom
                # point 62 - 66
                one = calculate_distance(landmark[62], landmark[66]) / denom
                # point 63 - 65
                two = calculate_distance(landmark[63], landmark[65]) / denom

                #compare against threshold
                if zero >= zero_const[0] and zero <= zero_const[1]:
                    is_smile[0] = 1

                if one >= one_const[0] and one <= one_const[1]:
                    is_smile[1] = 1

                if two >= two_const[0] and two <= two_const[1]:
                    is_smile[2] = 1

                # we know it passes if it gets at least two
                if np.count_nonzero(is_smile) > 1:
                    face_coords.append(faces[pos])
    return face_coords


# To be able to detect a smile, we will find the distance between four
# sets of points in the mouth, that is: (61, 67) (62, 66) (63, 65) and
# (60, 64). For the first three, we will make them a ratio of the last
# one. We will see these values for a few test images and then find
# a fitting high and low threshold to use.

# load model
# face detection model
configFile = '../model/deploy.prototxt'
modelFile = '../model/res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Landmark detection model
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel('../model/lbfmodel.yaml')

inpath = "../../module_13/project/ignore/complete/images"
outpath = "../../module_13/project/ignore/results/"
# read the images
f = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in \
    os.walk(inpath) for f in filenames]

for read_image in f[:10]:
    image = cv2.imread(read_image)
    faces = smile_detector(image)

    if len(faces) > 0:
        file_name = read_image.split('\\')[-1]
        # copy the file to a different path
        for face in faces:
            cv2.rectangle(image, face, (0, 255, 0), 3)
        cv2.imwrite(outpath + file_name, image)

    cv2.waitKey(0)