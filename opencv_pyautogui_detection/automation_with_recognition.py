# imports
import cv2
import pyautogui as gui
import numpy as np

# for clicking
import win32api
import win32con

def press_key(hexKeyCode):
    win32api.keybd_event(hexKeyCode, 0, 0, 0)  # Press key
    # Release key
    win32api.keybd_event(hexKeyCode, 0,\
                         win32con.KEYEVENTF_KEYUP, 0)


# we now create a method that will tell us whether
# the face is in the bounding box, or if it is not
def face_inbox(bbox, face_coords):
    '''
    This function will check whether the face is
    in the bounding box or not.

    It takes two 4 value tuples:
    (topx, topy, bottx, botty)

    Returns 'left', 'right', 'center', 'down' or 'up'
    depending on the position of the inner box wrt the
    bounding box
    '''

    if(face_coords[0] < bbox[0]):
        return 'left'
    elif(face_coords[1] < bbox[1]):
        return 'up'
    elif(face_coords[2] > bbox[2]):
        return 'right'
    elif(face_coords[3] > bbox[3]):
        return 'down'
    
    return 'center'



def move_key(key:str):
    '''
    Make a keystroke, depending on the key pressed.

    It takes the returns from face_inbox and presses
    a set key.

    You can customize for a single or for all keys.
    '''

    if key == 'left':
        press_key(win32con.VK_LEFT)
    elif key == 'right':
        press_key(win32con.VK_RIGHT)
    elif key == 'up':
        press_key(win32con.VK_UP)
    elif key == 'down':
        press_key(win32con.VK_DOWN)
    else:
        pass


# We load the model
net = cv2.dnn.readNetFromCaffe('model/deploy.prototxt',\
                               'model/res10_300x300_ssd_iter_140000.caffemodel')

def draw_rectangle(top, bottom, frame, color=(0, 255, 0)):
    '''
    Given coordinates and a frame, draw a rectangle.

    Takes two tuples, top and bottom and the frame,
    Optionally takes a color
    Returns a copy of the frame, redrawn.
    '''

    cv2.rectangle(frame, top, bottom, color=color, thickness=3)


#methods
def get_predictions(net, frame):
    '''
    This function takes the frame as input to the model and
    gets the prediciton of whether it has a face or not.

    It returns a dictionary with the coordinates of the face:
    {
      'Top': (TopX, TopY),
      'Bottom': (BottomX, BottomY) }
    '''
    #Will hold the dictionary of coordinates
    face_coordinates = []

    h, w = frame.shape[:2]
    # We will first create a blob from the image
    # A blob is an image/images that have the same depth,
    # shape (width, height) and that have been
    # preprocessed in the same manner
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 scalefactor=1.0,
                                 size=(300,300),
                                 mean=(104.0, 177.0, 123.0))
    
    # The function blobFromImage returns a 4-D tuple like so:
    # (num_images, num_channels, width, height)

    # We then feed our blob into the net
    net.setInput(blob)

    # We perform a feed foward across
    # all layers to get a prediction
    predictions = net.forward()

    # The forward() function also returns a 4-D tuple like so:
    # (1, 1, 200, 7)
    # 1, 1 - number of images working on
    # 200 - number of faces detected
    # 7 - a vector of 7 values like so:
    # [Image number, Binary (0 or 1), confidence score (0 to 1),
    # StartX, StartY, EndX, EndY]

    # With this data, we can filter based on confidence score
    # we iterate through every face
    for face in range(predictions.shape[2]):
        #get confidence
        confidence = predictions[0, 0, face, 2]

        if confidence > 0.5:
            #Take the coordinates
            (Top_x, Top_y) = predictions[0, 0, face, 3:5] * np.array([w, h])
            (Bott_x, Bott_y) = predictions[0, 0, face, 5:] * np.array([w, h])
        else:
            continue
        
        face_coordinates.append({
            'top': (int(Top_x), int(Top_y)),
            'bottom' : (int(Bott_x), int(Bott_y)),
            'confidence' : confidence
        })
    return face_coordinates


# We open the VideoCapture object here to be able
# to create a bounding box
cap = cv2.VideoCapture(0)

frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

#create a bounding box: w=200, h=200
top_x, top_y = int(frame_width//2 - 90), int(frame_height//2 - 85)
bottom_x, bottom_y = int(frame_width//2 + 90), int(frame_height//2 + 100)
bbox = [top_x, top_y, bottom_x, bottom_y]

#assign in tuples
centre_top = (top_x, top_y)
centre_bottom = (bottom_x, bottom_y)

count = -1
init = 0
move = ''
last_move = ''

#the main loop
while True:
    #capture every other frame
    count += 1
    if count % 2 != 0:
        continue

    if not cap.isOpened():
        cap = cv2.VideoCapture()

    ret, frame = cap.read()
    
    if ret is False:
        break

    # Flip frame to match normal movement
    frame = cv2.flip(frame, 2)

    # Get predictions
    faces = get_predictions(net, frame)

    # If we find a face
    if len(faces) > 0:
        # Sort according to highest confidence,
        # in case of many points
        sorted_faces = sorted(faces, key=lambda x: x['confidence'],
                              reverse=True)

        # Take the best sorted image
        face = sorted_faces[0]

        #Get the face coordinates
        face_top_x, face_top_y = face['top']
        face_bott_x, face_bott_y = face['bottom']
        
        # Take the highest element
        draw_rectangle((face_top_x, face_top_y),
                        (face_bott_x, face_bott_y),
                        frame, color=(0, 0, 255))
    
        # check whether the face is in bbox:
        face_coords = [face_top_x, face_top_y, face_bott_x, face_bott_y]

        move = face_inbox(bbox, face_coords)

        cv2.putText(frame, move, (30, 50), 0, 3, (180, 0, 180), 2)

        #initialize game
        if init == 0:
            if move == 'center':
                init = 1
                cv2.putText(frame, 'Initialized', (30, 50), 0, 3, (10, 100, 180), 10)
        else:
            if last_move == 'center':
                move_key(move)

        last_move = move
    
    
    # Draw the central bounding box
    draw_rectangle(centre_top, centre_bottom, frame)

    cv2.imshow('current frame', frame)

    key = cv2.waitKey(5)
    if key == 27:
        break



cap.release()
cv2.destroyAllWindows()
