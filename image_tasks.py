import numpy as np
import cv2

def resize_image(img, size):
    h, w = img.shape[:2]

    if h == w: 
        return cv2.resize(img, (size, size), cv2.INTER_AREA)

    elif h > w:
        scaling_factor = size / w
        x_shape = size
        y_shape = int(h * scaling_factor)
        img_reshaped = cv2.resize(img, (x_shape, y_shape), cv2.INTER_AREA)
        offset = int((y_shape - size) / 2)
        img_reshaped = img_reshaped[offset:offset+size, :, :]
        
    else :
        scaling_factor = size / h
        x_shape = int(w * scaling_factor) 
        y_shape = size
        img_reshaped = cv2.resize(img, (x_shape, y_shape), cv2.INTER_AREA)
        offset = int((x_shape - size) / 2)
        img_reshaped = img_reshaped[:, offset:offset+size, :]
        
    return img_reshaped

def scale_image(img, scale_ratio):
    """scale_ratio must be >= 1"""
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * scale_ratio), int(h * scale_ratio)))

def vert_trans_image(img, y):
    rows, cols = img.shape[:2]
    translation_matrix = np.float32([[1, 0, 0], [0, 1, y]])
    img_translated = cv2.warpAffine(img, translation_matrix, (cols, rows))
    return img_translated

def rotate_image(img, angle):
    """
    angle: rotation angle (in degrees)
    """
    rows, cols = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D(( (cols - 1)/2 , (rows - 1)/2), angle, 1)
    img_rotated = cv2.warpAffine(img, rotation_matrix, (cols, rows))
    return img_rotated

def occlude_image(img, x, y, size=40):
    """Occlude the given area with a gray square"""
    if x + size > img.shape[1] or y + size > img.shape[0]:
        raise ValueError("Occlusion box is out of bounds")
    img_occluded = img.copy()
    img_occluded[y:y+size, x:x+size, :] = 128
    return img_occluded

def print_image(img, name="Image"):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# def crop_image(img, size=224):
#     """Crop a portion of size `size` from img"""
#     h, w = img.shape[:2]
#     x = np.random.randint(0, w - size)
#     y = np.random.randint(0, h - size)
#     return img[y:y+size, x:x+size, :], x, y


def test():
    img = cv2.imread('images/n01440764_tench.jpeg')
    img = resize_image(img, 256)

    print_image(img, "Original")
    print_image(scale_image(img, 1.5), "Scaled")
    print_image(vert_trans_image(img, -50), "Vertically Translated by -50")
    print_image(rotate_image(img, 200), "Rotated by 200 degrees")
    print_image(occlude_image(img, 200, 120, 50), "Occlusion")

if __name__ == "__main__":
    test()
