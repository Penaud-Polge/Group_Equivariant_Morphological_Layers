from MorphoLayerP4 import *
import matplotlib.pyplot as plt
from skimage.io import imread

# get image

(x_train_init, y_train_init), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
Image = np.zeros((1, 48, 48, 1))

Image[0,10:38,10:38,0] = x_train_init[5:6,:,:]
Image = Image.astype('float32')/255

plt.figure()
plt.imshow(Image[0,:,:,0])

"""
Image = imread('spiral_196.png')
Image = Image.astype('float32')/255.0
Image = np.expand_dims(Image, 0)
"""


# Lifting Structural Element

# Creating some lifting SE

"""
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

for i in range(3):
    print(kernel1[...,i,0])

for i in range(3):
    print(kernel2[...,i,0])

filter = tf.concat([kernel1, kernel2], axis=-1)
"""

filter = np.random.rand(3,3,1,2)
filter = (filter > 0.7)*10.0
filter = filter.astype('float32') - 10

for i in range(1):
    print(filter[...,i,0])

for i in range(1):
    print(filter[...,i,1])

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
MaxTimesPlusDilatedLifting = MaxTimesPlusDilationLifting(Image)
print("MaxTimesPlusDilated Lifting Shape : ", MaxTimesPlusDilatedLifting.shape)

MaxTimesPlusErosionLifting = MaxTimesPlusErosionLiftingP4(num_filters=2, kernel_size = (3,3), padding='valid')
MaxTimesPlusErosionLifting.build(Image.shape[1:])
MaxTimesPlusErosionLifting.kernel = filter
MaxTimesPlusErodedLifting = MaxTimesPlusErosionLifting(Image)
print("MaxTimesPlusEroded Lifting Shape : ", MaxTimesPlusErodedLifting.shape)

MaxTimesPlusOpeningLifting = MaxTimesPlusOpeningLiftingP4(num_filters=2, kernel_size = (3,3), padding='valid')
MaxTimesPlusOpeningLifting.build(Image.shape[1:])
MaxTimesPlusOpeningLifting.kernel = filter
MaxTimesPlusOpenedLifting = MaxTimesPlusOpeningLifting(Image)
print("MaxTimesPlusOpened Lifting Shape : ", MaxTimesPlusOpenedLifting.shape)

MaxTimesPlusClosingLifting = MaxTimesPlusClosingLiftingP4(num_filters=2, kernel_size = (3,3), padding='valid')
MaxTimesPlusClosingLifting.build(Image.shape[1:])
MaxTimesPlusClosingLifting.kernel = filter
MaxTimesPlusClosedLifting = MaxTimesPlusClosingLifting(Image)
print("MaxTimesPlusClosed Lifting Shape : ", MaxTimesPlusClosedLifting.shape)


# Show Dilated Images 

plt.figure()
plt.suptitle('Dilation Lifting')
for i in range(2):
    for j in range(4):
        plt.subplot(2, 4, 1 + 4*i + j), plt.imshow(DilatedLifting[0,j,:,:,i])

plt.figure()
plt.suptitle('scalarMaxTimesPlus Dilation Lifting')
for i in range(2):
    for j in range(4):
        plt.subplot(2, 4, 1 + 4*i + j), plt.imshow(scalarMaxTimesPlusDilatedLifting[0,j,:,:,i])
    
plt.figure()
plt.suptitle('MaxTimesPlus Dilation Lifting')
for i in range(2):
    for j in range(4):
        plt.subplot(2, 4, 1 + 4*i + j), plt.imshow(MaxTimesPlusDilatedLifting[0,j,:,:,i])


# Show Eroded Images

plt.figure()
plt.suptitle('Erosion Lifting')
for i in range(2):
    for j in range(4):
        plt.subplot(2, 4, 1 + 4*i + j), plt.imshow(ErodedLifting[0,j,:,:,i])

plt.figure()
plt.suptitle('scalarMaxTimesPlus Erosion Lifting')
for i in range(2):
    for j in range(4):
        plt.subplot(2, 4, 1 + 4*i + j), plt.imshow(scalarMaxTimesPlusErodedLifting[0,j,:,:,i])
    
plt.figure()
plt.suptitle('MaxTimesPlus Erosion Lifting')
for i in range(2):
    for j in range(4):
        plt.subplot(2, 4, 1 + 4*i + j), plt.imshow(MaxTimesPlusErodedLifting[0,j,:,:,i])

# Show Opened Images

plt.figure()
plt.suptitle('Opening Lifting')
for i in range(2):
    for j in range(4):
        plt.subplot(2, 4, 1 + 4*i + j), plt.imshow(OpenedLifting[0,j,:,:,i])

plt.figure()
plt.suptitle('scalarMaxTimesPlus Opening Lifting')
for i in range(2):
    for j in range(4):
        plt.subplot(2, 4, 1 + 4*i + j), plt.imshow(scalarMaxTimesPlusOpenedLifting[0,j,:,:,i])
    
plt.figure()
plt.suptitle('MaxTimesPlus Opening Lifting')
for i in range(2):
    for j in range(4):
        plt.subplot(2, 4, 1 + 4*i + j), plt.imshow(MaxTimesPlusOpenedLifting[0,j,:,:,i])

# Show Closed Images

plt.figure()
plt.suptitle('Closing Lifting')
for i in range(2):
    for j in range(4):
        plt.subplot(2, 4, 1 + 4*i + j), plt.imshow(ClosedLifting[0,j,:,:,i])

plt.figure()
plt.suptitle('scalarMaxTimesPlus Closing Lifting')
for i in range(2):
    for j in range(4):
        plt.subplot(2, 4, 1 + 4*i + j), plt.imshow(scalarMaxTimesPlusClosedLifting[0,j,:,:,i])
    
plt.figure()
plt.suptitle('MaxTimesPlus Closing Lifting')
for i in range(2):
    for j in range(4):
        plt.subplot(2, 4, 1 + 4*i + j), plt.imshow(MaxTimesPlusClosedLifting[0,j,:,:,i])

plt.show()

# P4 Structural Elements

# Creating some P4 Structural Elements

filter = np.random.rand(3,3,3,2,4)
filter = (filter > 0.7)*10.0
filter = filter.astype('float32') - 10



# Layers P4

print(ErodedLifting.shape)

Dilation = DilationP4(num_filters=4, kernel_size = (3,3))
Dilation.build(ErodedLifting.shape[2:])
Dilation.kernel = filter
DilatedP4 = Dilation(ErodedLifting)
print("DilatedP4 shape : ", DilatedP4.shape)

Erosion = ErosionP4(num_filters=4, kernel_size = (3,3))
Erosion.build(DilatedLifting.shape[2:])
Erosion.kernel = filter
ErodedP4 = Erosion(DilatedLifting)
print("ErodedP4 shape : ", ErodedP4.shape)

Opening = OpeningP4(num_filters=4, kernel_size = (3,3))
Opening.build(ClosedLifting.shape[2:])
Opening.kernel = filter
OpenedP4 = Opening(ClosedLifting)
print("OpenedP4 shape : ", OpenedP4.shape)

Closing = ClosingP4(num_filters=4, kernel_size = (3,3))
Closing.build(OpenedLifting.shape[2:])
Closing.kernel = filter
ClosedP4 = Closing(OpenedLifting)
print("ClosedP4 shape : ", ClosedP4.shape)


scalarMaxTimesPlusDilation = scalarMaxTimesPlusDilationP4(num_filters=4, kernel_size = (3,3))
scalarMaxTimesPlusDilation.build(scalarMaxTimesPlusErodedLifting.shape[1:])
scalarMaxTimesPlusDilation.kernel = filter
scalarMaxTimesPlusDilatedP4 = scalarMaxTimesPlusDilation(scalarMaxTimesPlusErodedLifting)
print("scalarMaxTimesPlusDilatedP4 shape : ", scalarMaxTimesPlusDilatedP4.shape)

scalarMaxTimesPlusErosion = scalarMaxTimesPlusErosionP4(num_filters=4, kernel_size = (3,3))
scalarMaxTimesPlusErosion.build(scalarMaxTimesPlusDilatedLifting.shape[1:])
scalarMaxTimesPlusErosion.kernel = filter
scalarMaxTimesPlusErodedP4 = scalarMaxTimesPlusErosion(scalarMaxTimesPlusDilatedLifting)
print("scalarMaxTimesPlusErodedP4 shape : ", scalarMaxTimesPlusErodedP4.shape)

scalarMaxTimesPlusOpening = scalarMaxTimesPlusOpeningP4(num_filters=4, kernel_size = (3,3))
scalarMaxTimesPlusOpening.build(scalarMaxTimesPlusClosedLifting.shape[1:])
scalarMaxTimesPlusOpening.kernel = filter
scalarMaxTimesPlusOpenedP4 = scalarMaxTimesPlusOpening(scalarMaxTimesPlusClosedLifting)
print("scalarMaxTimesPlusOpenedP4 shape : ", scalarMaxTimesPlusOpenedP4.shape)

scalarMaxTimesPlusClosing = scalarMaxTimesPlusClosingP4(num_filters=4, kernel_size = (3,3))
scalarMaxTimesPlusClosing.build(scalarMaxTimesPlusOpenedLifting.shape[1:])
scalarMaxTimesPlusClosing.kernel = filter
scalarMaxTimesPlusClosedP4 = scalarMaxTimesPlusClosing(scalarMaxTimesPlusOpenedLifting)
print("scalarMaxTimesPlusClosedP4 shape : ", scalarMaxTimesPlusClosedP4.shape)

MaxTimesPlusDilation = MaxTimesPlusDilationP4(num_filters=4, kernel_size = (3,3))
MaxTimesPlusDilation.build(MaxTimesPlusErodedLifting.shape[1:])
MaxTimesPlusDilation.kernel = filter
MaxTimesPlusDilatedP4 = MaxTimesPlusDilation(MaxTimesPlusErodedLifting)
print("MaxTimesPlusDilatedP4 shape : ", MaxTimesPlusDilatedP4.shape)

MaxTimesPlusErosion = MaxTimesPlusErosionP4(num_filters=4, kernel_size = (3,3))
MaxTimesPlusErosion.build(MaxTimesPlusDilatedLifting.shape[1:])
MaxTimesPlusErosion.kernel = filter
MaxTimesPlusErodedP4 = MaxTimesPlusErosion(MaxTimesPlusDilatedLifting)
print("MaxTimesPlusErodedP4 shape : ", MaxTimesPlusErodedP4.shape)

MaxTimesPlusOpening = MaxTimesPlusOpeningP4(num_filters=4, kernel_size = (3,3))
MaxTimesPlusOpening.build(MaxTimesPlusClosedLifting.shape[1:])
MaxTimesPlusOpening.kernel = filter
MaxTimesPlusOpenedP4 = MaxTimesPlusOpening(MaxTimesPlusClosedLifting)
print("MaxTimesPlusOpenedP4 shape : ", MaxTimesPlusOpenedP4.shape)

MaxTimesPlusClosing = MaxTimesPlusClosingP4(num_filters=4, kernel_size = (3,3))
MaxTimesPlusClosing.build(MaxTimesPlusOpenedLifting.shape[1:])
MaxTimesPlusClosing.kernel = filter
MaxTimesPlusClosedP4 = MaxTimesPlusClosing(MaxTimesPlusOpenedLifting)
print("MaxTimesPlusClosedP4 shape : ", MaxTimesPlusClosedP4.shape)


plt.figure()
plt.suptitle('Dilation P4')
for i in range(4):
    for j in range(4):
        plt.subplot(4, 4, 1 + 4*i + j), plt.imshow(DilatedP4[0,j,:,:,i])

plt.figure()
plt.suptitle('scalarMaxTimesPlus Dilation P4')
for i in range(4):
    for j in range(4):
        plt.subplot(4, 4, 1 + 4*i + j), plt.imshow(scalarMaxTimesPlusDilatedP4[0,j,:,:,i])
    
plt.figure()
plt.suptitle('MaxTimesPlus Dilation P4')
for i in range(4):
    for j in range(4):
        plt.subplot(4, 4, 1 + 4*i + j), plt.imshow(MaxTimesPlusDilatedP4[0,j,:,:,i])


# Show Eroded Images

plt.figure()
plt.suptitle('Erosion P4')
for i in range(4):
    for j in range(4):
        plt.subplot(4, 4, 1 + 4*i + j), plt.imshow(ErodedP4[0,j,:,:,i])

plt.figure()
plt.suptitle('scalarMaxTimesPlus Erosion P4')
for i in range(4):
    for j in range(4):
        plt.subplot(4, 4, 1 + 4*i + j), plt.imshow(scalarMaxTimesPlusErodedP4[0,j,:,:,i])
    
plt.figure()
plt.suptitle('MaxTimesPlus Erosion P4')
for i in range(4):
    for j in range(4):
        plt.subplot(4, 4, 1 + 4*i + j), plt.imshow(MaxTimesPlusErodedP4[0,j,:,:,i])

# Show Opened Images

plt.figure()
plt.suptitle('Opening P4')
for i in range(4):
    for j in range(4):
        plt.subplot(4, 4, 1 + 4*i + j), plt.imshow(OpenedP4[0,j,:,:,i])

plt.figure()
plt.suptitle('scalarMaxTimesPlus Opening P4')
for i in range(4):
    for j in range(4):
        plt.subplot(4, 4, 1 + 4*i + j), plt.imshow(scalarMaxTimesPlusOpenedP4[0,j,:,:,i])
    
plt.figure()
plt.suptitle('MaxTimesPlus Opening P4')
for i in range(4):
    for j in range(4):
        plt.subplot(4, 4, 1 + 4*i + j), plt.imshow(MaxTimesPlusOpenedP4[0,j,:,:,i])

# Show Closed Images

plt.figure()
plt.suptitle('Closing P4')
for i in range(4):
    for j in range(4):
        plt.subplot(4, 4, 1 + 4*i + j), plt.imshow(ClosedP4[0,j,:,:,i])

plt.figure()
plt.suptitle('scalarMaxTimesPlus Closing P4')
for i in range(4):
    for j in range(4):
        plt.subplot(4, 4, 1 + 4*i + j), plt.imshow(scalarMaxTimesPlusClosedP4[0,j,:,:,i])
    
plt.figure()
plt.suptitle('MaxTimesPlus Closing P4')
for i in range(4):
    for j in range(4):
        plt.subplot(4, 4, 1 + 4*i + j), plt.imshow(MaxTimesPlusClosedP4[0,j,:,:,i])

plt.show()