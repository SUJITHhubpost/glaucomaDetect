
#Usage python predict.py '<path of image to be predicted>'

from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(128, 128))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor


if __name__ == "__main__":

    # load model
    model = load_model("model.h5")
    classes = ["Glaucomatous", "Non Glaucomatic"]
    # image path
    img_path = '/home/hp1024/Desktop/keras/dl-medical-imaging/glaucoma/data/normal/ROI - 8451_left.jpeg.png'    # normal
    #img_path = '/home/hp1024/Desktop/keras/dl-medical-imaging/glaucoma/data/Glaucomatous/ROI - image237prime0.jpg.png'      # gl
    img_path = sys.argv[1]
    
    # load a single image
    new_image = load_image(img_path)

    # check prediction
    pred = model.predict(new_image)
    print(pred)
    print(classes[np.argmax(pred, axis=1)[0]])
