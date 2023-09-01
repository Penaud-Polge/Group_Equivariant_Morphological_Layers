import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import tensorflow as tf

from MorphoP4 import *

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

image_num = np.random.randint(0, 49000)

Image = x_train[image_num:image_num + 1,:,:]

Image = Image.astype('float32')/255

Image = tf.pad(Image, [[0,0], [5, 5], [5, 5]], "CONSTANT")


#Image = imread('vanilla_image.png')
#Image = Image.astype('float32')/255


Image = tf.expand_dims(Image, axis = -1)

Image_90 = tf.image.rot90(Image, k = 1)

DilLiftingLayer = DilationLiftingP4(1, (5, 5))
DilLiftingLayer.build(Image.shape[1:])

pattern = np.zeros((5,5)).astype('float32')
pattern[2,:2] = 0.2

plt.figure()
plt.imshow(pattern)

pattern = tf.expand_dims(tf.expand_dims(pattern, axis = -1), axis = -1)

DilLiftingLayer.kernel= pattern

SE = DilLiftingLayer.kernel
plt.figure()
plt.imshow(SE[:,:,0,0])


plt.figure()
plt.subplot(1, 2, 1), plt.imshow(Image[0,:,:,0])
plt.subplot(1, 2, 2), plt.imshow(Image_90[0,:,:,0])

DilLifted = DilLiftingLayer(Image)
DilLifted_90 = DilLiftingLayer(Image_90)

print(DilLifted.shape)

plt.figure()
for i in range(4):
    plt.subplot(2, 4, 1+i), plt.imshow(DilLifted[0,i,:,:, 0])
    plt.subplot(2, 4, 5+i), plt.imshow(DilLifted_90[0,i,:,:, 0])


EroLiftingLayer = ErosionLiftingP4(1, (5, 5))
EroLiftingLayer.build(Image.shape[1:])



EroLiftingLayer.kernel = pattern

EroLifted = EroLiftingLayer(Image)
EroLifted_90 = EroLiftingLayer(Image_90)

plt.figure()
for i in range(4):
    plt.subplot(2, 4, 1+i), plt.imshow(EroLifted[0,i,:,:, 0])
    plt.subplot(2, 4, 5+i), plt.imshow(EroLifted_90[0,i,:,:, 0])


ConvLiftingLayer = ConvLiftingP4(1, (5,5))
ConvLiftingLayer.build(Image.shape[1:])

pattern = np.ones((5,5,1,1)).astype('float32')
pattern[:2,:,0,0] = -1.0
pattern[2,:,0,0] = 0.0

ConvLiftingLayer.kernel = pattern

ConvLifted = ConvLiftingLayer(Image)
ConvLifted_90 = ConvLiftingLayer(Image_90)

plt.figure()
for i in range(4):
    plt.subplot(2, 4, 1+i), plt.imshow(ConvLifted[0,i,:,:, 0])
    plt.subplot(2, 4, 5+i), plt.imshow(ConvLifted_90[0,i,:,:, 0])


DilLayerP4 = DilationP4(1, (3, 3))

DilLayerP4.build(Image.shape[1:])

pattern = np.zeros(DilLayerP4.kernel_size).astype('float32')
pattern[:DilLayerP4.kernel_size[0]//2, :DilLayerP4.kernel_size[1]//2] = 0.1
pattern_90 = tf.experimental.numpy.rot90(pattern, k = 1, axes = (0,1))
pattern_180 = tf.experimental.numpy.rot90(pattern, k = 2, axes = (0,1))

patterns = tf.stack([pattern, pattern_90, pattern_180], axis = -1)
patterns = tf.expand_dims(patterns, axis = -1)
DilLayerP4.kernel = tf.expand_dims(patterns, axis = -1)

SE = DilLayerP4.kernel
print(SE.shape)

plt.figure()
plt.title('SE')
plt.imshow(SE[:,:,:,0,0]*10)

Dil = DilLayerP4(EroLifted)
Dil_90 = DilLayerP4(EroLifted_90)


EroLayerP4 = ErosionP4(1, (3, 3))

EroLayerP4.build(Image.shape[1:])

EroLayerP4.kernel = tf.expand_dims(patterns, axis = -1)

SE = EroLayerP4.kernel


Ero = EroLayerP4(DilLifted)
Ero_90 = EroLayerP4(DilLifted_90)

ConvLayerP4 = ConvP4(1, (3,3))

ConvLayerP4.build(Image.shape[1:])

ConvLayerP4.kernel = tf.expand_dims(patterns, axis = -1)

Conv = ConvLayerP4(ConvLifted)
Conv_90 = ConvLayerP4(ConvLifted_90)

plt.figure()
plt.suptitle('Erosion Lifting then Dilation')
for j in range(4):

    plt.subplot(2, 4,1+j), plt.imshow(Dil[0,j, :,:,0])
    plt.subplot(2, 4,5+j), plt.imshow(Dil_90[0,j, :,:,0])


plt.figure()
plt.suptitle('Dilation Lifting then Erosion')
for j in range(4):

    plt.subplot(2, 4,1+j), plt.imshow(Ero[0,j, :,:,0])
    plt.subplot(2, 4,5+j), plt.imshow(Ero_90[0,j, :,:,0])


plt.figure()
plt.suptitle('Conv lifting then Conv')
for j in range(4):

    plt.subplot(2, 4, 1 + j), plt.imshow(Conv[0,j, :,:,0])
    plt.subplot(2, 4, 5+j), plt.imshow(Conv_90[0,j, :,:,0])
plt.show()
