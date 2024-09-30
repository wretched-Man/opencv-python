import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'gray'

#creating an image
square = np.zeros(apple.shape[:2], np.uint8)
square[240: 440, 70:270] = 255
plt.imshow(square)

sq_points1 = np.float32([[70, 240], [270, 240], [70, 440]])
sq_points2 = np.float32([[112, 240], [312, 240], [70, 440]])
shear_matrix = cv2.getAffineTransform(sq_points1, sq_points2)

sheared_square = cv2.warpAffine(square, shear_matrix, (512, 512))
plt.imshow(sheared_square)

rand_points = np.float32([[70, 290], [312, 240], [70, 390]])
morph_matrix = cv2.getAffineTransform(sq_points1, rand_points)
morphed_square = cv2.warpAffine(square, morph_matrix, (512, 512))
plt.imshow(morphed_square)


def shear_coord(coord, matrix):
    """
    Given a coordinate and a matrix, it returns the sheared coordinate.
    """

    try:
        return np.int32(np.ceil(matrix.dot(coord)))
    except:
        try:
            return np.int32(np.ceil(coord.dot(matrix)))
        except:
            print("Incompatible types!")


def shear_horizontal(img, value):
    """
    Shear an image horizontally.
    The value given will be filled into the horizontal shear matrix.
    """
    if value == 0:
        return img

    result = np.zeros_like(img)

    if len(img.shape) == 3:
        # Handle 3-channel img as 3 1-channel ones
        result[:, :, 0] = shear_horizontal(img[:, :, 0], value)
        result[:, :, 1] = shear_horizontal(img[:, :, 1], value)
        result[:, :, 2] = shear_horizontal(img[:, :, 2], value)
        
        return result
    else:
        shear_matrix = np.array([[1, value],[0 , 1]])
        row_choice = 0
        for y in range(img.shape[0] - 1, -1, -1):
            # count backwards to symbolize that
            # origin is at bottom left corner
            left = np.array([0, y], np.int32)
            left_shear = shear_coord(left, shear_matrix)
    
            #splice into result
            if left_shear[0] > 0:
                take = img.shape[1] - left_shear[0]
                if take > 0:
                    result[row_choice, left_shear[0]:] = img[row_choice, :take]
            else:
                take = img.shape[1] + left_shear[0]
                if take > 0:
                    result[row_choice, :take] = img[row_choice, -take:]
            
            row_choice += 1 
        return result


def shear_vertical(img, value):
    """
    Shear an image, vertically.
    The value given will be filled into the vertical shear matrix.
    """
    if value == 0:
        return img
    
    result = np.zeros_like(img)

    if len(img.shape) == 3:
        result[:, :, 0] = shear_vertical(img[:, :, 0], value)
        result[:, :, 1] = shear_vertical(img[:, :, 1], value)
        result[:, :, 2] = shear_vertical(img[:, :, 2], value)
        
        return result
    else:
        shear_matrix = np.array([[1, 0],[value , 1]])
        col_choice = 0
        for x in range(0, img.shape[0]):
            up = np.array([x, img.shape[0]-1], np.int32) 
            up_shear = shear_coord(up, shear_matrix)
            
            #splice into result
            if up_shear[1] > img.shape[0]:
                remove = up_shear[1] - (img.shape[0] - 1)
                take = img.shape[1] - remove
                if take > 0:
                    result[:take, col_choice] = img[remove:, col_choice]
            else:
                remove = img.shape[0] - up_shear[1]
                take = img.shape[1] - remove
                if take > 0:
                    result[remove:, col_choice] = img[:take, col_choice]
            
            col_choice += 1
        return result

def shear(img, vertical = 0, horizontal = 0):
    """
    Given an image and the vertical and horizontal shear,
    return the sheared image.
    The vertical and horizontal values can be float & -ve.
    """

    if vertical == 0 and horizontal == 0:
        return img

    res = shear_vertical(shear_horizontal(img, horizontal), vertical)
    return res    

apple = cv2.imread('images/apple.png')
sheared_apple = shear(apple, .1, .3)

plt.figure(figsize=[10, 10])
plt.subplot(121); plt.imshow(apple[:, :, ::-1]); plt.title('Original'); plt.axis('off')
plt.subplot(122); plt.imshow(sheared_apple[:, :, ::-1]); plt.title('Sheared'); plt.axis('off')
