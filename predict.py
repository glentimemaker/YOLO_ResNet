import sys
import numpy as np
import cv2
import utils
import matplotlib.pyplot as plt
from model_continue_train import ResNet50

import keras.backend as K
if 'tensorflow' == K.backend():
    import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config2 = tf.ConfigProto()
config2.gpu_options.allow_growth = True
set_session(tf.Session(config=config2))

if len(sys.argv) < 2:
    print("Image name missing; \n usage: python predict.py <image name>")
    sys.exit()
img_name = sys.argv[1]

from io import BytesIO

def convertToJpeg(im):
    with BytesIO() as f:
        im.save(f, format='JPEG')
        return f.getvalue()

img = cv2.imread(img_name)
if len(img.shape) > 3:
    img = convertToJpeg(img)
img_float = cv2.resize(img, (224,224)).astype(np.float32)
img_float -= 128

img_in = np.expand_dims(img_float, axis=0)

model = ResNet50(include_top=False, load_weight=True, weights='models/rerun9_0.01_weights.02-2.05.hdf5',
                input_shape=(224,224,3))
pred = model.predict(img_in)

dims = (img.shape[1], img.shape[0])
bboxes = utils.get_boxes(pred[0], cutoff=0.2, dims=dims)
bboxes = utils.nonmax_suppression(bboxes, iou_cutoff = 0.05)
draw = utils.draw_boxes(img, bboxes, color=(0, 0, 255), thick=3, draw_dot=True, radius=3)
draw = draw.astype(np.uint8)

plt.imshow(draw[...,::-1])
plt.show()
