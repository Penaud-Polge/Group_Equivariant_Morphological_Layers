from MorphoLayerP4 import *
import matplotlib.pyplot as plt
from skimage.io import imread

# get image

Image = imread('spiral_196.png')
Image = Image.astype('float32')/255.0
Image = np.expand_dims(Image, 0)

# Lifting Kernel 

# Creating some lifting kernel

kernel1 = np.zeros((3, 3, 3, 1)).astype('float32')
kernel1[:,:,0,0] = np.eye(3)*10.0
kernel1[:,0,1,0] = 10.0
kernel1[0,:,2,0] = 10.0

kernel1 = kernel1 - 10.0

kernel2 = np.zeros((3,3,3,1)).astype('float32')
kernel2[0,0,0,0] = 10.0
kernel2[2,2,1,0] = 10.0
kernel2[0,1,2,0] = 10.0
kernel2 = kernel2 - 10

filter = tf.concat([kernel1, kernel2], axis=-1)

# Lifting Layers on P4

DilationLifting = DilationLiftingP4(num_filters=2, kernel_size = (3,3), padding='valid')
DilationLifting.build(Image.shape[1:])
DilationLifting.kernel = filter
DilatedLifting = DilationLifting(Image)
print("Dilated Lifting Shape : ", DilatedLifting.shape)

ErosionLifting = ErosionLiftingP4(num_filters=2, kernel_size = (3,3), padding='valid')
ErosionLifting.build(Image.shape[1:])
ErosionLifting.kernel = filter
ErodedLifting = ErosionLifting(Image)
print("Eroded Lifting Shape : ", ErodedLifting.shape)


OpeningLifting = OpeningLiftingP4(num_filters=2, kernel_size = (3,3), padding='valid')
OpeningLifting.build(Image.shape[1:])
OpeningLifting.kernel = filter
OpenedLifting = OpeningLifting(Image)
print("Opened Lifting Shape : ", OpenedLifting.shape)

ClosingLifting = ClosingLiftingP4(num_filters=2, kernel_size = (3,3), padding='valid')
ClosingLifting.build(Image.shape[1:])
ClosingLifting.kernel = filter
ClosedLifting = ClosingLifting(Image)
print("Closed Lifting Shape : ", ClosedLifting.shape)


scalarMaxTimesPlusDilationLifting = scalarMaxTimesPlusDilationLiftingP4(num_filters=2, kernel_size = (3,3), padding='valid')
scalarMaxTimesPlusDilationLifting.build(Image.shape[1:])
scalarMaxTimesPlusDilationLifting.kernel = filter
scalarMaxTimesPlusDilatedLifting = scalarMaxTimesPlusDilationLifting(Image)
print("scalarMaxTimesPlusDilated Lifting Shape : ", scalarMaxTimesPlusDilatedLifting.shape)

scalarMaxTimesPlusErosionLifting = scalarMaxTimesPlusErosionLiftingP4(num_filters=2, kernel_size = (3,3), padding='valid')
scalarMaxTimesPlusErosionLifting.build(Image.shape[1:])
scalarMaxTimesPlusErosionLifting.kernel = filter
scalarMaxTimesPlusErodedLifting = scalarMaxTimesPlusErosionLifting(Image)
print("scalarMaxTimesPlusEroded Lifting Shape : ", scalarMaxTimesPlusErodedLifting.shape)

scalarMaxTimesPlusOpeningLifting = scalarMaxTimesPlusOpeningLiftingP4(num_filters=2, kernel_size = (3,3), padding='valid')
scalarMaxTimesPlusOpeningLifting.build(Image.shape[1:])
scalarMaxTimesPlusOpeningLifting.kernel = filter
scalarMaxTimesPlusOpenedLifting = scalarMaxTimesPlusOpeningLifting(Image)
print("scalarMaxTimesPlusOpened Lifting Shape : ", scalarMaxTimesPlusOpenedLifting.shape)

scalarMaxTimesPlusClosingLifting = scalarMaxTimesPlusClosingLiftingP4(num_filters=2, kernel_size = (3,3), padding='valid')
scalarMaxTimesPlusClosingLifting.build(Image.shape[1:])
scalarMaxTimesPlusClosingLifting.kernel = filter
scalarMaxTimesPlusClosedLifting = scalarMaxTimesPlusClosingLifting(Image)
print("scalarMaxTimesPlusClosed Lifting Shape : ", scalarMaxTimesPlusClosedLifting.shape)


MaxTimesPlusDilationLifting = MaxTimesPlusDilationLiftingP4(num_filters=2, kernel_size = (3,3), padding='valid')
MaxTimesPlusDilationLifting.build(Image.shape[1:])
MaxTimesPlusDilationLifting.kernel = filter
print("Times Kernel : ", MaxTimesPlusDilationLifting.timesKernel)
MaxTimesPlusDilatedLifting = MaxTimesPlusDilationLifting(Image)
print("MaxTimesPlusDilated Lifting Shape : ", MaxTimesPlusDilatedLifting.shape)

MaxTimesPlusErosionLifting = MaxTimesPlusErosionLiftingP4(num_filters=2, kernel_size = (3,3), padding='valid')
MaxTimesPlusErosionLifting.build(Image.shape[1:])
MaxTimesPlusErosionLifting.kernel = filter
print("Times Kernel : ", MaxTimesPlusErosionLifting.timesKernel)
MaxTimesPlusErodedLifting = MaxTimesPlusErosionLifting(Image)
print("MaxTimesPlusEroded Lifting Shape : ", MaxTimesPlusErodedLifting.shape)

MaxTimesPlusOpeningLifting = MaxTimesPlusOpeningLiftingP4(num_filters=2, kernel_size = (3,3), padding='valid')
MaxTimesPlusOpeningLifting.build(Image.shape[1:])
MaxTimesPlusOpeningLifting.kernel = filter
print("Times Kernel : ", MaxTimesPlusOpeningLifting.timesKernel)
MaxTimesPlusOpenedLifting = MaxTimesPlusOpeningLifting(Image)
print("MaxTimesPlusOpened Lifting Shape : ", MaxTimesPlusOpenedLifting.shape)

MaxTimesPlusClosingLifting = MaxTimesPlusClosingLiftingP4(num_filters=2, kernel_size = (3,3), padding='valid')
MaxTimesPlusClosingLifting.build(Image.shape[1:])
MaxTimesPlusClosingLifting.kernel = filter
print("Times Kernel : ", MaxTimesPlusClosingLifting.timesKernel)
MaxTimesPlusClosedLifting = MaxTimesPlusClosingLifting(Image)
print("MaxTimesPlusClosed Lifting Shape : ", MaxTimesPlusClosedLifting.shape)


# Show Dilated Images 

plt.figure()
plt.suptitle('Dilation Lifting')
for i in range(2):
    plt.subplot(3, 2, 1 + i), plt.imshow(DilatedLifting[0,0,:,:,i])
    plt.subplot(3, 2, 3 + i), plt.imshow(scalarMaxTimesPlusDilatedLifting[0,0,:,:,i])
    plt.subplot(3, 2, 5 + i), plt.imshow(MaxTimesPlusDilatedLifting[0,0,:,:,i])

# Show Eroded Images

plt.figure()
plt.suptitle('Erosion Lifting')
for i in range(2):
    plt.subplot(3, 2, 1 + i), plt.imshow(ErodedLifting[0,0,:,:,i])
    plt.subplot(3, 2, 3 + i), plt.imshow(scalarMaxTimesPlusErodedLifting[0,0,:,:,i])
    plt.subplot(3, 2, 5 + i), plt.imshow(MaxTimesPlusErodedLifting[0,0,:,:,i])

# Show Opened Images

plt.figure()
plt.suptitle('Opening Lifting')
for i in range(2):
    plt.subplot(3, 2, 1 + i), plt.imshow(OpenedLifting[0,0,:,:,i])
    plt.subplot(3, 2, 3 + i), plt.imshow(scalarMaxTimesPlusOpenedLifting[0,0,:,:,i])
    plt.subplot(3, 2, 5 + i), plt.imshow(MaxTimesPlusOpenedLifting[0,0,:,:,i])

# Show Closed Images

plt.figure()
plt.suptitle('Closing Lifting')
for i in range(2):
    plt.subplot(3, 2, 1 + i), plt.imshow(ClosedLifting[0,0,:,:,i])
    plt.subplot(3, 2, 3 + i), plt.imshow(scalarMaxTimesPlusClosedLifting[0,0,:,:,i])
    plt.subplot(3, 2, 5 + i), plt.imshow(MaxTimesPlusClosedLifting[0,0,:,:,i])

plt.show()

# Layers P4

Dilation = DilationP4(num_filters=4, kernel_size = (3,3))
Dilation.build(Image.shape[1:])

Erosion = ErosionP4(num_filters=4, kernel_size = (3,3))
Erosion.build(Image.shape[1:])

Opening = OpeningP4(num_filters=4, kernel_size = (3,3))
Opening.build(Image.shape[1:])

Closing = ClosingP4(num_filters=4, kernel_size = (3,3))
Closing.build(Image.shape[1:])


scalarMaxTimesPlusDilation = scalarMaxTimesPlusDilationP4(num_filters=4, kernel_size = (3,3))
scalarMaxTimesPlusDilation.build(Image.shape[1:])

scalarMaxTimesPlusErosion = scalarMaxTimesPlusErosionP4(num_filters=4, kernel_size = (3,3))
scalarMaxTimesPlusErosion.build(Image.shape[1:])

scalarMaxTimesPlusOpening = scalarMaxTimesPlusOpeningP4(num_filters=4, kernel_size = (3,3))
scalarMaxTimesPlusOpening.build(Image.shape[1:])

scalarMaxTimesPlusClosing = scalarMaxTimesPlusClosingP4(num_filters=4, kernel_size = (3,3))
scalarMaxTimesPlusClosing.build(Image.shape[1:])


MaxTimesPlusDilation = MaxTimesPlusDilationP4(num_filters=4, kernel_size = (3,3))
MaxTimesPlusDilation.build(Image.shape[1:])

MaxTimesPlusErosion = MaxTimesPlusErosionP4(num_filters=4, kernel_size = (3,3))
MaxTimesPlusErosion.build(Image.shape[1:])

MaxTimesPlusOpening = MaxTimesPlusOpeningP4(num_filters=4, kernel_size = (3,3))
MaxTimesPlusOpening.build(Image.shape[1:])

MaxTimesPlusClosing = MaxTimesPlusClosingP4(num_filters=4, kernel_size = (3,3))
MaxTimesPlusClosing.build(Image.shape[1:])