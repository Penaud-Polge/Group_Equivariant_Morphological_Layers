import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt

import skimage
from skimage.morphology import dilation

from MorphoP4 import *


xinput = layers.Input(shape=(12, 12, 12, 1))
Dilxy=Dilation3Dxy(filters=1,kernel_size=(3,3),kernel_initializer='Zeros')(xinput)
Dilyz=Dilation3Dyz(filters=1,kernel_size=(3,1),kernel_initializer='Zeros')(Dilxy)
model=tf.keras.Model(xinput,Dilyz)
model.summary() 


DATA=np.zeros([1,12,12,12,1])
DATA[0,4:6,4:6,4:6,0]=1

DATA_DILATE=model.predict(DATA)

DATA_DILATE_SKIMAGE = dilation(DATA[0,:,:,:,0], footprint=np.ones([3,3,3]))

print("It should be exact the same result")
print(np.max(np.abs(DATA_DILATE[0,:,:,:,0]-DATA_DILATE_SKIMAGE)))