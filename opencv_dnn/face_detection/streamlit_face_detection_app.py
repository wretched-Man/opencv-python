# Build a streamlit application that takes a user image,
# Displays the image, edited beside it, with the faces 
# highlighted

# We will need a function that loads the model
# A function that does predictions
# a function that sorts predictions according to a threshold

import cv2
import numpy as np
#from PIL import Image
import streamlit as st

@st.cache_resource()
def load_model():
    """
    Load model and return an OpenCV cv2.dnn.Net object
    """
    configFile = 'model/deploy.prototxt'
    modelFile = 'model/res10_300x300_ssd_iter_140000_fp16.caffemodel'
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return net

def detectfaces(net, frame):
    """
    This function detects all the faces in the given frame and
    returns a list of all faces found.

    Note: it does not filter according to a threshold. It only
    returns the results of the forward method.
    """

    # create a blob from the image
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                scalefactor=1.0,
                                size=(300,300),
                                mean=(104.0, 177.0, 123.0))
    
    # we set it as input to our model
    net.setInput(blob)
    detections = net.forward()
    return detections


def drawdetectedfaces(frame, detections, thresh_confidence=0.5):
    """
    This functions draws the faces detected by `detectfaces`.

    It draws as many as pass the threshold_confidence.
    It returns the frame, drawn
    """

    face_coordinates = []

    h, w = frame.shape[:2]

    for face in range(detections.shape[2]):
        #get confidence
        confidence = detections[0, 0, face, 2]

        if confidence >= thresh_confidence:
            #Take the coordinates
            (Top_x, Top_y) = detections[0, 0, face, 3:5] * np.array([w, h])
            (Bott_x, Bott_y) = detections[0, 0, face, 5:] * np.array([w, h])
        else:
            continue
        
        face_coordinates.append({
            'top': (int(Top_x), int(Top_y)),
            'bottom' : (int(Bott_x), int(Bott_y)),
        })

    # define custom thickness
    custom_thickness = max(1, int(h/200))
    # drawing on the frame
    for accept_face in face_coordinates:
        cv2.rectangle(frame,
                      accept_face['top'],
                      accept_face['bottom'],
                      color = (0, 255, 0),
                      thickness = custom_thickness)
    return frame, len(face_coordinates)

st.title('Face Detection App')

user_file = st.file_uploader(
                'Input an image to detect faces.',
                ['jpg', 'jpeg', 'png']
                 )

#Session state
if 'file_uploaded_name' not in st.session_state:
    st.session_state.file_uploaded_name = None
if 'many_detections' not in st.session_state:
    st.session_state.many_detections = None


if user_file is not None:
    #load model
    net = load_model()
    file_name = user_file.name

    #convert the image to a usable format for OpenCV
    file_bytes = np.asarray(bytearray(user_file.read()),
                            np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    #Define two columns
    image_col, face_detect_col = st.columns(2)

    with image_col:
        st.subheader('Image')

        # Display the image
        st.image(img,
                 caption='Original | ' + user_file.name,
                 channels='BGR'
                )

    #Implement a slider below both columns
    threshold_slider = st.slider(
        'Select a confidence value between 0 - 1',
        0.0, 1.0, 0.5, 0.01
        )
    
    #check if this is a new file
    if file_name != st.session_state.file_uploaded_name:
        st.session_state.file_uploaded_name=file_name
        #face detection
        st.session_state.many_detections = detectfaces(net, img)
        st.write(
            "Image changed. Doing new detections."
        )
    else:
        st.write(
            "Image not changed. No new detections being made."
        )

    faces_image = img.copy()
    faces_image, faces_count = drawdetectedfaces(
                            faces_image,
                            st.session_state.many_detections,
                            threshold_slider)
    
    with face_detect_col:
        st.subheader('Faces detected')
       
        st.image(
            faces_image,
            caption= str(faces_count) +
                    " faces detected.",
            channels='BGR')
        
    #Allow the user to download the image
