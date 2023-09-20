import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.ops import nn
from tensorflow.python.keras import activations

@tf.function
def dilation2d(x, st_element, strides, padding,rates=(1, 1)):
    """

    From MORPHOLAYERS

    Basic Dilation Operator
    :param st_element: Nonflat structuring element
    :strides: strides as classical convolutional layers
    :padding: padding as classical convolutional layers
    :rates: rates as classical convolutional layers
    """
    x = tf.nn.dilation2d(x, st_element, (1, ) + strides + (1, ),padding.upper(),"NHWC",(1,)+rates+(1,))            
    return x


@tf.function
def erosion2d(x, st_element, strides, padding,rates=(1, 1)):
    """

    From MORPHOLAYERS

    Basic Erosion Operator
    :param st_element: Nonflat structuring element
    """
    x = tf.nn.erosion2d(x, st_element, (1, ) + strides + (1, ),padding.upper(),"NHWC",(1,)+rates+(1,))
    return x

class ConvLiftingP4(tf.keras.layers.Layer):


    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same', dilation_rate=(1,1), activation=None,use_bias=False,kernel_initializer='Zeros',
                 kernel_constraint=None,kernel_regularization=None,bias_initializer='zeros',bias_regularizer=None,
                 bias_constraint=None,**kwargs):
        
        super(ConvLiftingP4, self).__init__(**kwargs)

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates=dilation_rate

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)

        # for we are assuming channel last
        self.channel_axis = -1

        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.activation = activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        
        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',constraint =self.kernel_constraint,regularizer=self.kernel_regularization)

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.num_filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        # Be sure to call this at the end
        super(ConvLiftingP4, self).build(input_shape)

    def call(self, x):

        output = []

        for i in range(4):

            kernel_rot = tf.experimental.numpy.rot90(self.kernel, k = i, axes = (0,1))

            output.append(tf.nn.conv2d(x, kernel_rot, (1, ) + self.strides + (1, ) , self.padding.upper(), "NHWC", self.rates))

        output = tf.stack(output, axis= 1)
        if self.use_bias:
            output=tf.keras.backend.bias_add(output, self.bias)
        
        if self.activation is not None:
            return self.activation(output)
        
        return output
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.rates,
        })
        return config

class ConvP4(tf.keras.layers.Layer):

    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same', dilation_rate=(1,1), activation=None,use_bias=False,kernel_initializer='Zeros',
                 kernel_constraint=None,kernel_regularization=None,bias_initializer='zeros',bias_regularizer=None,
                 bias_constraint=None,**kwargs):
        
        super(ConvP4, self).__init__(**kwargs)

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates=dilation_rate

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)

        # for we are assuming channel last
        self.channel_axis = -1

        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.activation = activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[self.channel_axis]

        kernel_shape = self.kernel_size + (3, input_dim, self.num_filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',constraint =self.kernel_constraint,regularizer=self.kernel_regularization)

        
        
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.num_filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        # Be sure to call this at the end
        super(ConvP4, self).build(input_shape)
    

    def call(self, x):


        y = tf.concat([x[:,-1:, ...], x, x[:,0:1, ...]], axis = 1)

        res_rota = []

        for j in range(4):

            kernel_rot = tf.experimental.numpy.rot90(self.kernel, k = j, axes = (0,1) )

            res_depth = []

            for k in range(3):

                res_depth.append(tf.nn.conv2d(y[:,j+k,...], kernel_rot[...,k, :,:], (1, ) + self.strides + (1, ) , self.padding.upper(), "NHWC", self.rates))

            res_depth = tf.stack(res_depth, axis=-1)
            res_depth = tf.reduce_sum(res_depth, axis = -1)

            res_rota.append(res_depth)

        output = tf.stack(res_rota, axis= 1)
        if self.use_bias:
            output=tf.keras.backend.bias_add(output, self.bias)

        
        if self.activation is not None:
            return self.activation(output)
        
        return output
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.rates,
        })
        return config


class MaxTimesPlusDilationLiftingP4(tf.keras.layers.Layer):

    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same', dilation_rate=(1,1), activation=None,use_bias=False,kernel_initializer='Zeros',
                 kernel_constraint=None,kernel_regularization=None, timesKernel_initializer = 'Ones', timesKernel_regularizer=None, timesKernel_constraint=tf.keras.constraints.NonNeg(), bias_initializer='zeros',bias_regularizer=None,
                 bias_constraint=None,**kwargs):
        
        super(MaxTimesPlusDilationLiftingP4, self).__init__(**kwargs)

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates=dilation_rate

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)

        self.timesKernel_initializer = tf.keras.initializers.get(timesKernel_initializer)
        self.timesKernel_constraint = tf.keras.constraints.get(timesKernel_constraint)
        self.timesKernel_regularization = tf.keras.regularizers.get(timesKernel_regularizer)

        # for we are assuming channel last
        self.channel_axis = -1

        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.activation = activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        
        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',constraint =self.kernel_constraint,regularizer=self.kernel_regularization)
        self.timesKernel = self.add_weight(shape = kernel_shape,
                                      initializer=self.timesKernel_initializer,
                                      constraint = self.timesKernel_constraint, regularizer=self.timesKernel_regularization)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.num_filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        # Be sure to call this at the end
        super(MaxTimesPlusDilationLiftingP4, self).build(input_shape)

    def call(self, x):

        y = tf.image.extract_patches(x, sizes=(1,) + self.kernel_size + (1,), strides = (1,) + self.strides + (1,), rates=(1,) + self.rates + (1,), padding=self.padding.upper())
            
        y = tf.reshape(y, shape = (-1, x.shape[1], x.shape[2], self.kernel_size[0]*self.kernel_size[1], x.shape[3]))
        y = tf.tile(tf.expand_dims(y, axis = -1), [1, 1, 1, 1, 1, self.num_filters])

        res = []
        for i in range(4):

            kernel_rot = tf.experimental.numpy.rot90(self.kernel, k = i, axes = (0,1))
            timesKernel_rot = tf.experimental.numpy.rot90(self.timesKernel, k = i, axes = (0,1))

            kernel_rot = tf.reshape(kernel_rot, (self.kernel.shape[0]*self.kernel.shape[1], self.kernel.shape[2], self.kernel.shape[3]))

            timesKernel_rot = tf.reshape(timesKernel_rot, (self.timesKernel.shape[0]*self.timesKernel.shape[1], self.timesKernel.shape[2], self.timesKernel.shape[3]))

            res_rot = tf.add(tf.multiply(y, timesKernel_rot), kernel_rot)
            res.append(tf.reduce_sum(tf.reduce_max(res_rot, axis = -3), axis = -2))
        
        output = tf.stack(res, axis = 1)

        if self.use_bias:
            output=tf.keras.backend.bias_add(output, self.bias)
            #output = tf.math.maximum(output, self.bias)
            #output = tf.math.multiply(output, self.bias)
        
        if self.activation is not None:
            return self.activation(output)

        return output
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.rates,
        })
        return config
     
class MaxTimesPlusErosionLiftingP4(tf.keras.layers.Layer):

    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same', dilation_rate=(1,1), activation=None,use_bias=False,kernel_initializer='Zeros',
                 kernel_constraint=None,kernel_regularization=None, timesKernel_initializer = 'Ones', timesKernel_regularizer=None, timesKernel_constraint=tf.keras.constraints.NonNeg(), bias_initializer='zeros',bias_regularizer=None,
                 bias_constraint=None,**kwargs):
        
        super(MaxTimesPlusErosionLiftingP4, self).__init__(**kwargs)

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates=dilation_rate

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)

        self.timesKernel_initializer = tf.keras.initializers.get(timesKernel_initializer)
        self.timesKernel_constraint = tf.keras.constraints.get(timesKernel_constraint)
        self.timesKernel_regularization = tf.keras.regularizers.get(timesKernel_regularizer)

        # for we are assuming channel last
        self.channel_axis = -1

        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.activation = activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        
        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',constraint =self.kernel_constraint,regularizer=self.kernel_regularization)
        self.timesKernel = self.add_weight(shape = kernel_shape,
                                      initializer=self.timesKernel_initializer,
                                      constraint = self.timesKernel_constraint, regularizer=self.timesKernel_regularization)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.num_filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        # Be sure to call this at the end
        super(MaxTimesPlusErosionLiftingP4, self).build(input_shape)

    def call(self, x):

        y = tf.image.extract_patches(x, sizes=(1,) + self.kernel_size + (1,), strides = (1,) + self.strides + (1,), rates=(1,) + self.rates + (1,), padding=self.padding.upper())
            
        y = tf.reshape(y, shape = (-1, x.shape[1], x.shape[2], self.kernel_size[0]*self.kernel_size[1], x.shape[3]))
        y = tf.tile(tf.expand_dims(y, axis = -1), [1, 1, 1, 1, 1, self.num_filters])

        res = []
        for i in range(4):

            kernel_rot = tf.experimental.numpy.rot90(self.kernel, k = i, axes = (0,1))
            timesKernel_rot = tf.experimental.numpy.rot90(self.timesKernel, k = i, axes = (0,1))

            kernel_rot = tf.reshape(kernel_rot, (self.kernel.shape[0]*self.kernel.shape[1], self.kernel.shape[2], self.kernel.shape[3]))

            timesKernel_rot = tf.reshape(timesKernel_rot, (self.timesKernel.shape[0]*self.timesKernel.shape[1], self.timesKernel.shape[2], self.timesKernel.shape[3]))

            res_rot = tf.divide(tf.add(y, -kernel_rot), timesKernel_rot + tf.keras.backend.epsilon())
            res.append(tf.reduce_sum(tf.reduce_min(res_rot, axis = -3), axis = -2))
        
        output = tf.stack(res, axis = 1)

        if self.use_bias:
            output=tf.keras.backend.bias_add(output, self.bias)
            #output = tf.math.maximum(output, self.bias)
            #output = tf.math.multiply(output, self.bias)
        
        if self.activation is not None:
            return self.activation(output)

        return output
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.rates,
        })
        return config
    
class MaxTimesPlusDilationP4(tf.keras.layers.Layer):

    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same', dilation_rate=(1,1), activation=None,use_bias=False,kernel_initializer='Zeros',
                 kernel_constraint=None,kernel_regularization=None, timesKernel_initializer = 'Ones', timesKernel_regularizer=None, timesKernel_constraint=tf.keras.constraints.NonNeg(), bias_initializer='zeros',bias_regularizer=None,
                 bias_constraint=None,**kwargs):
        
        super(MaxTimesPlusDilationP4, self).__init__(**kwargs)

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates=dilation_rate

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)

        self.timesKernel_initializer = tf.keras.initializers.get(timesKernel_initializer)
        self.timesKernel_constraint = tf.keras.constraints.get(timesKernel_constraint)
        self.timesKernel_regularization = tf.keras.regularizers.get(timesKernel_regularizer)

        # for we are assuming channel last
        self.channel_axis = -1

        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.activation = activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        
        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (3, input_dim, self.num_filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',constraint =self.kernel_constraint,regularizer=self.kernel_regularization)
        self.timesKernel = self.add_weight(shape = kernel_shape,
                                      initializer=self.timesKernel_initializer,
                                      constraint = self.timesKernel_constraint, regularizer=self.timesKernel_regularization)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.num_filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        # Be sure to call this at the end
        super(MaxTimesPlusDilationP4, self).build(input_shape)

    def call(self, x):

        y = tf.concat([x[:,-1:, ...], x, x[:,0:1, ...]], axis = 1)
        y = tf.extract_volume_patches(y, ksizes=(1, 1) + self.kernel_size + (1,), strides = (1, 1) + self.strides + (1,),  padding=self.padding.upper())
            
        y = tf.reshape(y, shape = (-1, y.shape[1], y.shape[2], y.shape[3], self.kernel_size[0]*self.kernel_size[1], x.shape[4]))
        y = tf.tile(tf.expand_dims(y, axis = -1), [1, 1, 1, 1, 1, 1, self.num_filters])

        y = tf.transpose(y, perm = [0, 2, 3, 4, 1, 5, 6])

        res = []
        for i in range(4):

            kernel_rot = tf.experimental.numpy.rot90(self.kernel, k = i, axes = (0,1))
            timesKernel_rot = tf.experimental.numpy.rot90(self.timesKernel, k = i, axes = (0,1))

            kernel_rot = tf.reshape(kernel_rot, (self.kernel.shape[0]*self.kernel.shape[1], self.kernel.shape[2], self.kernel.shape[3], self.timesKernel.shape[4]))

            timesKernel_rot = tf.reshape(timesKernel_rot, (self.timesKernel.shape[0]*self.timesKernel.shape[1], self.timesKernel.shape[2], self.timesKernel.shape[3], self.timesKernel.shape[4]))

            res_rot = tf.add(tf.multiply(y[...,i:i+3,:,:], timesKernel_rot), kernel_rot)
            res.append(tf.reduce_sum(tf.reduce_max(res_rot, axis = (-4, -3)), axis = -2))
        
        output = tf.stack(res, axis = 1)

        if self.use_bias:
            output=tf.keras.backend.bias_add(output, self.bias)
            #output = tf.math.maximum(output, self.bias)
            #output = tf.math.multiply(output, self.bias)
        
        if self.activation is not None:
            return self.activation(output)

        return output
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.rates,
        })
        return config

class MaxTimesPlusErosionP4(tf.keras.layers.Layer):

    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same', dilation_rate=(1,1), activation=None,use_bias=False,kernel_initializer='Zeros',
                 kernel_constraint=None,kernel_regularization=None, timesKernel_initializer = 'Ones', timesKernel_regularizer=None, timesKernel_constraint=tf.keras.constraints.NonNeg(), bias_initializer='zeros',bias_regularizer=None,
                 bias_constraint=None,**kwargs):
        
        super(MaxTimesPlusErosionP4, self).__init__(**kwargs)

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates=dilation_rate

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)

        self.timesKernel_initializer = tf.keras.initializers.get(timesKernel_initializer)
        self.timesKernel_constraint = tf.keras.constraints.get(timesKernel_constraint)
        self.timesKernel_regularization = tf.keras.regularizers.get(timesKernel_regularizer)

        # for we are assuming channel last
        self.channel_axis = -1

        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.activation = activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        
        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (3, input_dim, self.num_filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',constraint =self.kernel_constraint,regularizer=self.kernel_regularization)
        self.timesKernel = self.add_weight(shape = kernel_shape,
                                      initializer=self.timesKernel_initializer,
                                      constraint = self.timesKernel_constraint, regularizer=self.timesKernel_regularization)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.num_filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        # Be sure to call this at the end
        super(MaxTimesPlusErosionP4, self).build(input_shape)

    def call(self, x):

        y = tf.concat([x[:,-1:, ...], x, x[:,0:1, ...]], axis = 1)
        y = tf.extract_volume_patches(y, ksizes=(1, 1) + self.kernel_size + (1,), strides = (1, 1) + self.strides + (1,), padding=self.padding.upper())
            
        y = tf.reshape(y, shape = (-1, y.shape[1], y.shape[2], y.shape[3], self.kernel_size[0]*self.kernel_size[1], x.shape[4]))
        y = tf.tile(tf.expand_dims(y, axis = -1), [1, 1, 1, 1, 1, 1, self.num_filters])

        y = tf.transpose(y, perm = [0, 2, 3, 4, 1, 5, 6])

        res = []
        for i in range(4):

            kernel_rot = tf.experimental.numpy.rot90(self.kernel, k = i, axes = (0,1))
            timesKernel_rot = tf.experimental.numpy.rot90(self.timesKernel, k = i, axes = (0,1))

            kernel_rot = tf.reshape(kernel_rot, (self.kernel.shape[0]*self.kernel.shape[1], self.kernel.shape[2], self.kernel.shape[3], self.timesKernel.shape[4]))

            timesKernel_rot = tf.reshape(timesKernel_rot, (self.timesKernel.shape[0]*self.timesKernel.shape[1], self.timesKernel.shape[2], self.timesKernel.shape[3], self.timesKernel.shape[4]))

            res_rot = tf.divide(tf.add(y[...,i:i+3,:,:], -kernel_rot), timesKernel_rot + tf.keras.backend.epsilon())
            res.append(tf.reduce_sum(tf.reduce_min(res_rot, axis = (-4, -3)), axis = -2))
        
        output = tf.stack(res, axis = 1)

        if self.use_bias:
            output=tf.keras.backend.bias_add(output, self.bias)
            #output = tf.math.maximum(output, self.bias)
            #output = tf.math.multiply(output, self.bias)
        
        if self.activation is not None:
            return self.activation(output)

        return output
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.rates,
        })
        return config


class scalarMaxTimesPlusDilationLiftingP4(tf.keras.layers.Layer):

    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same', dilation_rate=(1,1), activation=None,use_bias=False,kernel_initializer='Zeros',
                 kernel_constraint=None,kernel_regularization=None, timesKernel_initializer = 'Ones', timesKernel_regularizer=None, timesKernel_constraint=None, bias_initializer='zeros',bias_regularizer=None,
                 bias_constraint=None,**kwargs):
        
        super(scalarMaxTimesPlusDilationLiftingP4, self).__init__(**kwargs)

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates=dilation_rate

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)

        self.timesKernel_initializer = tf.keras.initializers.get(timesKernel_initializer)
        self.timesKernel_constraint = tf.keras.constraints.get(timesKernel_constraint)
        self.timesKernel_regularization = tf.keras.regularizers.get(timesKernel_regularizer)

        # for we are assuming channel last
        self.channel_axis = -1

        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.activation = activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        
        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',constraint =self.kernel_constraint,regularizer=self.kernel_regularization)
        self.timesKernel = self.add_weight(shape = (input_dim, self.num_filters),
                                      initializer=self.timesKernel_initializer,
                                      constraint = self.timesKernel_constraint, regularizer=self.timesKernel_regularization)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.num_filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        # Be sure to call this at the end
        super(scalarMaxTimesPlusDilationLiftingP4, self).build(input_shape)

    def call(self, x):

        output = []

        for i in range(4):

            kernel_rot = tf.experimental.numpy.rot90(self.kernel, k = i, axes = (0,1))
            res_filters = []
            
            for j in range(self.num_filters):

                res_filters.append(tf.reduce_sum(dilation2d(tf.math.multiply(x,self.timesKernel[:,j]), kernel_rot[...,j], self.strides, self.padding),axis=-1))

            output.append(tf.stack(res_filters, axis = -1))

        output = tf.stack(output, axis= 1)
        if self.use_bias:
            output=tf.keras.backend.bias_add(output, self.bias)
            #output = tf.math.maximum(output, self.bias)
            #output = tf.math.multiply(output, self.bias)
        
        if self.activation is not None:
            return self.activation(output)
        
        return output
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.rates,
        })
        return config
    
class scalarMaxTimesPlusErosionLiftingP4(tf.keras.layers.Layer):

    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same', dilation_rate=(1,1), activation=None,use_bias=False,kernel_initializer='Zeros',
                 kernel_constraint=None,kernel_regularization=None, timesKernel_initializer = 'Ones', timesKernel_regularizer=None, timesKernel_constraint=tf.keras.constraints.NonNeg(), bias_initializer='zeros',bias_regularizer=None,
                 bias_constraint=None,**kwargs):
        
        super(scalarMaxTimesPlusErosionLiftingP4, self).__init__(**kwargs)

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates=dilation_rate

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)

        self.timesKernel_initializer = tf.keras.initializers.get(timesKernel_initializer)
        self.timesKernel_constraint = tf.keras.constraints.get(timesKernel_constraint)
        self.timesKernel_regularization = tf.keras.regularizers.get(timesKernel_regularizer)

        # for we are assuming channel last
        self.channel_axis = -1

        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.activation = activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        
        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',constraint =self.kernel_constraint,regularizer=self.kernel_regularization)
        self.timesKernel = self.add_weight(shape = (input_dim, self.num_filters),
                                      initializer=self.timesKernel_initializer,
                                      constraint = self.timesKernel_constraint, regularizer=self.timesKernel_regularization)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.num_filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        # Be sure to call this at the end
        super(scalarMaxTimesPlusErosionLiftingP4, self).build(input_shape)

    def call(self, x):

        output = []

        for i in range(4):

            kernel_rot = tf.experimental.numpy.rot90(self.kernel, k = i, axes = (0,1))
            res_filters = []
            
            for j in range(self.num_filters):

                res_filters.append(tf.reduce_sum(erosion2d(tf.math.divide(x, self.timesKernel[:,j] + tf.keras.backend.epsilon()), kernel_rot[...,j], self.strides, self.padding),axis=-1))

            output.append(tf.stack(res_filters, axis = -1))

        output = tf.stack(output, axis= 1)
        if self.use_bias:
            output=tf.keras.backend.bias_add(output, self.bias)
            #output = tf.math.maximum(output, self.bias)
            #output = tf.math.multiply(output, self.bias)
        
        if self.activation is not None:
            return self.activation(output)
        
        return output
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.rates,
        })
        return config
    
class scalarMaxTimesPlusOpeningLiftingP4(tf.keras.layers.Layer):

    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same', dilation_rate=(1,1), activation=None,use_bias=False,kernel_initializer='Zeros',
                 kernel_constraint=None,kernel_regularization=None, timesKernel_initializer = 'Ones', timesKernel_regularizer=None, timesKernel_constraint=None, bias_initializer='zeros',bias_regularizer=None,
                 bias_constraint=None,**kwargs):
        
        super(scalarMaxTimesPlusOpeningLiftingP4, self).__init__(**kwargs)

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates=dilation_rate

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)

        self.timesKernel_initializer = tf.keras.initializers.get(timesKernel_initializer)
        self.timesKernel_constraint = tf.keras.constraints.get(timesKernel_constraint)
        self.timesKernel_regularization = tf.keras.regularizers.get(timesKernel_regularizer)

        # for we are assuming channel last
        self.channel_axis = -1

        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.activation = activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        
        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',constraint =self.kernel_constraint,regularizer=self.kernel_regularization)
        self.timesKernel = self.add_weight(shape = (input_dim, self.num_filters),
                                      initializer=self.timesKernel_initializer,
                                      constraint = self.timesKernel_constraint, regularizer=self.timesKernel_regularization)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.num_filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        # Be sure to call this at the end
        super(scalarMaxTimesPlusOpeningLiftingP4, self).build(input_shape)

    def call(self, x):

        output = []

        for i in range(4):

            kernel_rot = tf.experimental.numpy.rot90(self.kernel, k = i, axes = (0,1))
            res_filters = []
            
            for j in range(self.num_filters):

                ero = erosion2d(tf.math.divide(x,self.timesKernel[:,j] + tf.keras.backend.epsilon()), kernel_rot[...,j], strides = (1,1), padding = self.padding)


                res_filters.append(tf.reduce_sum(dilation2d(tf.math.multiply(ero,self.timesKernel[:,j]), kernel_rot[...,j], self.strides, self.padding),axis=-1))

            output.append(tf.stack(res_filters, axis = -1))

        output = tf.stack(output, axis= 1)
        if self.use_bias:
            output=tf.keras.backend.bias_add(output, self.bias)
            #output = tf.math.maximum(output, self.bias)
            #output = tf.math.multiply(output, self.bias)
        
        if self.activation is not None:
            return self.activation(output)
        
        return output
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.rates,
        })
        return config

class scalarMaxTimesPlusClosingLiftingP4(tf.keras.layers.Layer):

    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same', dilation_rate=(1,1), activation=None,use_bias=False,kernel_initializer='Zeros',
                 kernel_constraint=None,kernel_regularization=None, timesKernel_initializer = 'Ones', timesKernel_regularizer=None, timesKernel_constraint=None, bias_initializer='zeros',bias_regularizer=None,
                 bias_constraint=None,**kwargs):
        
        super(scalarMaxTimesPlusClosingLiftingP4, self).__init__(**kwargs)

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates=dilation_rate

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)

        self.timesKernel_initializer = tf.keras.initializers.get(timesKernel_initializer)
        self.timesKernel_constraint = tf.keras.constraints.get(timesKernel_constraint)
        self.timesKernel_regularization = tf.keras.regularizers.get(timesKernel_regularizer)

        # for we are assuming channel last
        self.channel_axis = -1

        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.activation = activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        
        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',constraint =self.kernel_constraint,regularizer=self.kernel_regularization)
        self.timesKernel = self.add_weight(shape = (input_dim, self.num_filters),
                                      initializer=self.timesKernel_initializer,
                                      constraint = self.timesKernel_constraint, regularizer=self.timesKernel_regularization)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.num_filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        # Be sure to call this at the end
        super(scalarMaxTimesPlusClosingLiftingP4, self).build(input_shape)

    def call(self, x):

        output = []

        for i in range(4):

            kernel_rot = tf.experimental.numpy.rot90(self.kernel, k = i, axes = (0,1))
            res_filters = []
            
            for j in range(self.num_filters):

                dil = dilation2d(tf.math.multiply(x,self.timesKernel[:,j]), kernel_rot[...,j], strides = (1,1), padding = self.padding)


                res_filters.append(tf.reduce_sum(erosion2d(tf.math.divide(dil,self.timesKernel[:,j] + tf.keras.backend.epsilon()), kernel_rot[...,j], self.strides, self.padding),axis=-1))

            output.append(tf.stack(res_filters, axis = -1))

        output = tf.stack(output, axis= 1)
        if self.use_bias:
            output=tf.keras.backend.bias_add(output, self.bias)
            #output = tf.math.maximum(output, self.bias)
            #output = tf.math.multiply(output, self.bias)
        
        if self.activation is not None:
            return self.activation(output)
        
        return output
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.rates,
        })
        return config
    
class scalarMaxTimesPlusDilationP4(tf.keras.layers.Layer):

    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same', dilation_rate=(1,1), activation=None,use_bias=False,kernel_initializer='Zeros',
                 kernel_constraint=None,kernel_regularization=None, timesKernel_initializer = 'Ones', timesKernel_regularizer=None, timesKernel_constraint=None, bias_initializer='zeros',bias_regularizer=None,
                 bias_constraint=None,**kwargs):
        
        super(scalarMaxTimesPlusDilationP4, self).__init__(**kwargs)

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates=dilation_rate

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)

        self.timesKernel_initializer = tf.keras.initializers.get(timesKernel_initializer)
        self.timesKernel_constraint = tf.keras.constraints.get(timesKernel_constraint)
        self.timesKernel_regularization = tf.keras.regularizers.get(timesKernel_regularizer)

        # for we are assuming channel last
        self.channel_axis = -1

        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
    
    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[self.channel_axis]

        kernel_shape = self.kernel_size + (3, input_dim, self.num_filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',constraint =self.kernel_constraint,regularizer=self.kernel_regularization)

        self.timesKernel = self.add_weight(shape = (3, input_dim, self.num_filters),
                                      initializer=self.timesKernel_initializer,
                                      constraint = self.timesKernel_constraint, regularizer=self.timesKernel_regularization)
        
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.num_filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        # Be sure to call this at the end
        super(scalarMaxTimesPlusDilationP4, self).build(input_shape)

    def call(self, x):


        y = tf.concat([x[:,-1:, ...], x, x[:,0:1, ...]], axis = 1)

        res_rota = []

        for j in range(4):

            kernel_rot = tf.experimental.numpy.rot90(self.kernel, k = j, axes = (0,1) )

            res_SE_depth = []

            for k in range(3):

                res_filters = []

                for i in range(self.num_filters):

                    res_filters.append(tf.reduce_sum(dilation2d(tf.math.multiply(y[:,j + k,...], self.timesKernel[k,:,i]), kernel_rot[..., k, :,i],self.strides, self.padding), axis = -1))

                res_filters = tf.stack(res_filters, axis=-1)

                res_SE_depth.append(res_filters)

            res_SE_depth = tf.stack(res_SE_depth, axis=-1)
            res_SE_depth = tf.reduce_max(res_SE_depth, axis = -1)
            res_rota.append(res_SE_depth)

        output = tf.stack(res_rota, axis= 1)
        if self.use_bias:
            output=tf.keras.backend.bias_add(output, self.bias)
            #output = tf.math.maximum(output, self.bias)
            #output = tf.math.multiply(output, self.bias)
        
        if self.activation is not None:
            return self.activation(output)
        
        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.rates,
        })
        return config
    
class scalarMaxTimesPlusErosionP4(tf.keras.layers.Layer):

    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same', dilation_rate=(1,1), activation=None,use_bias=False,kernel_initializer='Zeros',
                 kernel_constraint=None,kernel_regularization=None, timesKernel_initializer = 'Ones', timesKernel_regularizer=None, timesKernel_constraint=tf.keras.constraints.NonNeg(), bias_initializer='zeros',bias_regularizer=None,
                 bias_constraint=None,**kwargs):
        
        super(scalarMaxTimesPlusErosionP4, self).__init__(**kwargs)

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates=dilation_rate

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)

        self.timesKernel_initializer = tf.keras.initializers.get(timesKernel_initializer)
        self.timesKernel_constraint = tf.keras.constraints.get(timesKernel_constraint)
        self.timesKernel_regularization = tf.keras.regularizers.get(timesKernel_regularizer)

        # for we are assuming channel last
        self.channel_axis = -1

        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
    
    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[self.channel_axis]

        kernel_shape = self.kernel_size + (3, input_dim, self.num_filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',constraint =self.kernel_constraint,regularizer=self.kernel_regularization)

        self.timesKernel = self.add_weight(shape = (3, input_dim, self.num_filters),
                                      initializer=self.timesKernel_initializer,
                                      constraint = self.timesKernel_constraint, regularizer=self.timesKernel_regularization)
        
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.num_filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        # Be sure to call this at the end
        super(scalarMaxTimesPlusErosionP4, self).build(input_shape)

    def call(self, x):


        y = tf.concat([x[:,-1:, ...], x, x[:,0:1, ...]], axis = 1)

        res_rota = []

        for j in range(4):

            kernel_rot = tf.experimental.numpy.rot90(self.kernel, k = j, axes = (0,1) )

            res_SE_depth = []

            for k in range(3):

                res_filters = []

                for i in range(self.num_filters):

                    res_filters.append(tf.reduce_sum(erosion2d(tf.math.divide(y[:,j + k,...], self.timesKernel[k,:,i] + tf.keras.backend.epsilon()), kernel_rot[..., k, :,i],self.strides, self.padding), axis = -1))

                res_filters = tf.stack(res_filters, axis=-1)

                res_SE_depth.append(res_filters)

            res_SE_depth = tf.stack(res_SE_depth, axis=-1)
            res_SE_depth = tf.reduce_max(res_SE_depth, axis = -1)
            res_rota.append(res_SE_depth)

        output = tf.stack(res_rota, axis= 1)
        if self.use_bias:
            output=tf.keras.backend.bias_add(output, self.bias)
            #output = tf.math.maximum(output, self.bias)
            #output = tf.math.multiply(output, self.bias)
        
        if self.activation is not None:
            return self.activation(output)
        
        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.rates,
        })
        return config
    
class scalarMaxTimesPlusOpeningP4(tf.keras.layers.Layer):

    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same', dilation_rate=(1,1), activation=None,use_bias=False,kernel_initializer='Zeros',
                 kernel_constraint=None,kernel_regularization=None, timesKernel_initializer = 'Ones', timesKernel_regularizer=None, timesKernel_constraint=None, bias_initializer='zeros',bias_regularizer=None,
                 bias_constraint=None,**kwargs):
        
        super(scalarMaxTimesPlusOpeningP4, self).__init__(**kwargs)

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates=dilation_rate

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)

        self.timesKernel_initializer = tf.keras.initializers.get(timesKernel_initializer)
        self.timesKernel_constraint = tf.keras.constraints.get(timesKernel_constraint)
        self.timesKernel_regularization = tf.keras.regularizers.get(timesKernel_regularizer)

        # for we are assuming channel last
        self.channel_axis = -1

        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
    
    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[self.channel_axis]

        kernel_shape = self.kernel_size + (3, input_dim, self.num_filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',constraint =self.kernel_constraint,regularizer=self.kernel_regularization)

        self.timesKernel = self.add_weight(shape = (3, input_dim, self.num_filters),
                                      initializer=self.timesKernel_initializer,
                                      constraint = self.timesKernel_constraint, regularizer=self.timesKernel_regularization)
        
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.num_filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        # Be sure to call this at the end
        super(scalarMaxTimesPlusOpeningP4, self).build(input_shape)

    def call(self, x):


        y = tf.concat([x[:,-1:, ...], x, x[:,0:1, ...]], axis = 1)

        res_rota = []

        for j in range(4):

            kernel_rot = tf.experimental.numpy.rot90(self.kernel, k = j, axes = (0,1) )

            res_SE_depth = []

            for k in range(3):

                res_filters = []

                for i in range(self.num_filters):

                    ero = erosion2d(tf.math.divide(y[:,j + k,...], self.timesKernel[k,:,i] + tf.keras.backend.epsilon()), kernel_rot[..., k, :,i],strides = (1, 1), padding = self.padding)

                    res_filters.append(tf.reduce_sum(dilation2d(tf.math.multiply(ero, self.timesKernel[k,:,i]), kernel_rot[..., k, :,i],self.strides, self.padding), axis = -1))

                res_filters = tf.stack(res_filters, axis=-1)

                res_SE_depth.append(res_filters)

            res_SE_depth = tf.stack(res_SE_depth, axis=-1)
            res_SE_depth = tf.reduce_max(res_SE_depth, axis = -1)
            res_rota.append(res_SE_depth)

        output = tf.stack(res_rota, axis= 1)
        if self.use_bias:
            output=tf.keras.backend.bias_add(output, self.bias)
            #output = tf.math.maximum(output, self.bias)
            #output = tf.math.multiply(output, self.bias)
        
        if self.activation is not None:
            return self.activation(output)
        
        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.rates,
        })
        return config

class scalarMaxTimesPlusClosingP4(tf.keras.layers.Layer):

    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same', dilation_rate=(1,1), activation=None,use_bias=False,kernel_initializer='Zeros',
                 kernel_constraint=None,kernel_regularization=None, timesKernel_initializer = 'Ones', timesKernel_regularizer=None, timesKernel_constraint=None, bias_initializer='zeros',bias_regularizer=None,
                 bias_constraint=None,**kwargs):
        
        super(scalarMaxTimesPlusClosingP4, self).__init__(**kwargs)

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates=dilation_rate

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)

        self.timesKernel_initializer = tf.keras.initializers.get(timesKernel_initializer)
        self.timesKernel_constraint = tf.keras.constraints.get(timesKernel_constraint)
        self.timesKernel_regularization = tf.keras.regularizers.get(timesKernel_regularizer)

        # for we are assuming channel last
        self.channel_axis = -1

        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
    
    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[self.channel_axis]

        kernel_shape = self.kernel_size + (3, input_dim, self.num_filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',constraint =self.kernel_constraint,regularizer=self.kernel_regularization)

        self.timesKernel = self.add_weight(shape = (3, input_dim, self.num_filters),
                                      initializer=self.timesKernel_initializer,
                                      constraint = self.timesKernel_constraint, regularizer=self.timesKernel_regularization)
        
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.num_filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        # Be sure to call this at the end
        super(scalarMaxTimesPlusClosingP4, self).build(input_shape)

    def call(self, x):


        y = tf.concat([x[:,-1:, ...], x, x[:,0:1, ...]], axis = 1)

        res_rota = []

        for j in range(4):

            kernel_rot = tf.experimental.numpy.rot90(self.kernel, k = j, axes = (0,1) )

            res_SE_depth = []

            for k in range(3):

                res_filters = []

                for i in range(self.num_filters):

                    dil = dilation2d(tf.math.multiply(y[:,j + k,...], self.timesKernel[k,:,i]), kernel_rot[..., k, :,i],strides = (1, 1), padding = self.padding)

                    res_filters.append(tf.reduce_sum(erosion2d(tf.math.divide(dil, self.timesKernel[k,:,i]+ tf.keras.backend.epsilon()), kernel_rot[..., k, :,i],self.strides, self.padding), axis = -1))

                res_filters = tf.stack(res_filters, axis=-1)

                res_SE_depth.append(res_filters)

            res_SE_depth = tf.stack(res_SE_depth, axis=-1)
            res_SE_depth = tf.reduce_max(res_SE_depth, axis = -1)
            res_rota.append(res_SE_depth)

        output = tf.stack(res_rota, axis= 1)
        if self.use_bias:
            output=tf.keras.backend.bias_add(output, self.bias)
            #output = tf.math.maximum(output, self.bias)
            #output = tf.math.multiply(output, self.bias)
        
        if self.activation is not None:
            return self.activation(output)
        
        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.rates,
        })
        return config


class AveragePoolingSpatial3D(tf.keras.layers.Layer):

    def __init__(self, **kwargs):

        super(AveragePoolingSpatial3D, self).__init__()

    def build(self, input_shape):

        self.inputShape = input_shape

    def call(self, inputs):

        results = tf.reduce_mean(inputs, axis=[2, 3])
        return results