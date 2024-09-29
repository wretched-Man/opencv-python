import streamlit as st
import pathlib
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import io
import base64
from PIL import Image

# Function to create a download link for output image
def get_image_download_link(img, filename, text):
    """Generates a link to download a particular image file."""
    buffered = io.BytesIO()
    img.save(buffered, format='JPEG')
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href

def get_points(mask):
    """
    Given a threshold image, find the contours and their enclosing regions, centre.
    """
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    
    for contour in contours:
        (x, y), _ = cv2.minEnclosingCircle(contour)
        coords = (int(x), int(y))
        results.append(coords)

    return (True, results)

def sort_output(list_sort):
    result = list_sort.copy()
    if len(list_sort) != 4:
        return list_sort
    
    result.sort(key= lambda x:x[0])
    result[:2].sort(key= lambda x:x[1])
    result[2:].sort(key= lambda x:x[1])
    return result


points1_list, points2_list = [], []

st.sidebar.title('Perspective Transformer')

uploaded_file = st.file_uploader(
                'Upload an image to rectify perspective.',
                ['jpg', 'jpeg', 'png']
                 )

if uploaded_file is not None:

    #convert the image to a usable format for OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()),
                            np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    h, w = img.shape[:2]
    if w > 800:
        h_, w_ = int(h * 800 / w), 800
    else:
        h_, w_ = h, w

    show_mask = st.sidebar.checkbox('Show Mask')
    # Create a canvas component.

    source_canvas = st_canvas(
        fill_color='white',
        stroke_width=5,
        stroke_color= 'black',
        background_image=Image.open(uploaded_file).resize((w_, h_)),
        update_streamlit=True,
        height=h_,
        width=w_,
        drawing_mode='freedraw',
        key="Source",
    )
    source = source_canvas.image_data
    source = source[:, :, 3]

    if source is not None:
        if show_mask:
            st.image(source)

    source_mask = cv2.resize(source, (w, h))
    found, source_list = get_points(source_mask)

    if source_list is not None:
        for point in source_list:
            if point not in points1_list:
                points1_list.append(point)
    
    choice_canvas = st_canvas(
        fill_color='white',
        stroke_width=5,
        stroke_color= 'green',
        background_image=Image.open(uploaded_file).resize((w_, h_)),
        update_streamlit=True,
        height=h_,
        width=w_,
        drawing_mode='freedraw',
        key="Destination",
    )
    dest = choice_canvas.image_data
    dest = dest[:, :, 3]

    if dest is not None:
        if show_mask:
            st.image(dest)

    dest_mask = cv2.resize(dest, (w, h))
    found, dest_points_list = get_points(dest_mask)

    if dest_points_list is not None:
        for point in dest_points_list:
            if point not in points2_list:
                points2_list.append(point)

    #sorting
    sorted_points1 = sort_output(points1_list)
    sorted_points2 = sort_output(points2_list)

    points1 = np.array(sorted_points1)
    points2 = np.array(sorted_points2)

    # computing homography
    H, _ = cv2.findHomography(points1, points2)
    
    result_img = cv2.warpPerspective(img, H, (w, h))
    st.image(result_img)
    st.write(points1, points2)





    

# blob detection
# return centres of blobs


