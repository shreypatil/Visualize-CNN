import cv2
import numpy as np
import torch

class CenterCrop224(object):

    def __init__(self, output_size=224):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def __call__(self, image):
        img = np.array(image)
        h, w = img.shape[:2]
        sz = 224
        
        if h == w: 
            return cv2.resize(img, (sz, sz), cv2.INTER_AREA)
        
        elif h > w:
            scaling_factor = sz / w
            x_shape = sz
            y_shape = int(h * scaling_factor)
            img_reshaped = cv2.resize(img, (x_shape, y_shape), cv2.INTER_AREA)
            offset = int((y_shape - sz) / 2)
            img_reshaped = img_reshaped[offset:offset + sz, :, :]
            
        else :
            scaling_factor = sz / h
            x_shape = int(w * scaling_factor) 
            y_shape = sz
            img_reshaped = cv2.resize(img, (x_shape, y_shape), cv2.INTER_AREA)
            offset = int((x_shape - sz) / 2)
            img_reshaped = img_reshaped[:, offset:offset+sz, :]
        

        del img 

        return img_reshaped



def resize_input(imgs):
    in_type = type(imgs)
    imgs_np = imgs.cpu().detach().numpy()
    imgs_reshaped = np.zeros((imgs_np.shape[0], 224, 224, 3), dtype=type(imgs_np))
    for i, img in enumerate(imgs_np) :
        img_reshaped = resize_image(img, 224)
        imgs_reshaped[i] = img_reshaped
        
    return torch.from_numpy(imgs_reshaped).type(in_type)
    
