# Code by Valentin Penaud--Polge
# Code coming from Morpholayers is specified
# Morpholayers : 
# https://people.cmm.minesparis.psl.eu/users/velasco/morpholayers/intro.html

# Please cite the paper if you use this code:
"""
Valentin Penaud--Polge, Santiago Velasco-Forero and Jesus Angulo
Group Equivariant Networks Using Morphological Operators
In: DGMM Proceedings. 2024
"""

import numpy as np

import tensorflow as tf
from tensorflow.python.ops import nn
from tensorflow.python.keras import activations

### Basic Functions from MORPHOLAYERS

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
    x = tf.nn.dilation2d(x, st_element, (1, strides[0], strides[1], 1 ),padding.upper(),"NHWC",(1,rates[0], rates[1],1))            
    return x

@tf.function
def erosion2d(x, st_element, strides, padding,rates=(1, 1)):
    """

    From MORPHOLAYERS

    Basic Erosion Operator
    :param st_element: Nonflat structuring element
    """
    x = tf.nn.erosion2d(x, st_element, (1, strides[0], strides[1], 1),padding.upper(),"NHWC",(1,rates[0],rates[1],1))
    return x

### Convolutional counterpart on P4

# P4 Lifting Layer

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
        #kernel_shape = self.kernel_size + (input_dim, self.num_filters)
        kernel_shape = (self.kernel_size[0], self.kernel_size[1], input_dim, self.num_filters)

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

            output.append(tf.nn.conv2d(x, kernel_rot, (1, self.strides[0], self.strides[1], 1) , self.padding.upper(), "NHWC", self.rates))

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
            'activation': self.activation,
            'use_bias': self.use_bias,

        })
        return config

# P4 Layer

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

        #kernel_shape = self.kernel_size + (3, input_dim, self.num_filters)
        kernel_shape = (self.kernel_size[0], self.kernel_size[1], 3, input_dim, self.num_filters)

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

                res_depth.append(tf.nn.conv2d(y[:,j+k,...], kernel_rot[...,k, :,:], (1, self.strides[0], self.strides[1], 1), self.padding.upper(), "NHWC", self.rates))

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
            'activation': self.activation,
            'use_bias': self.use_bias,
        })
        return config


### MAX-PLUS MorphoLayer on P4 (Dilation, Erosion, Opening and Closing)

# P4 Lifting Layers

class DilationLiftingP4(tf.keras.layers.Layer):

    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same', dilation_rate=(1,1), activation=None,use_bias=False,kernel_initializer='Zeros',
                 kernel_constraint=None,kernel_regularization=None,bias_initializer='zeros',bias_regularizer=None,
                 bias_constraint=None,**kwargs):
        
        super(DilationLiftingP4, self).__init__(**kwargs)

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
        #kernel_shape = self.kernel_size + (input_dim, self.num_filters)
        kernel_shape = (self.kernel_size[0], self.kernel_size[1], input_dim, self.num_filters)
        
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
        super(DilationLiftingP4, self).build(input_shape)

    def call(self, x):

        output = []

        for i in range(4):

            kernel_rot = tf.experimental.numpy.rot90(self.kernel, k = i, axes = (0,1))
            res_filters = []
            
            for j in range(self.num_filters):

                res_filters.append(tf.reduce_sum(dilation2d(x, kernel_rot[...,j], self.strides, self.padding),axis=-1))

            output.append(tf.stack(res_filters, axis = -1))

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
            'activation': self.activation,
            'use_bias': self.use_bias,
        })
        return config
    
class ErosionLiftingP4(tf.keras.layers.Layer):

    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same', dilation_rate=(1,1), activation=None,use_bias=False,kernel_initializer='Zeros',
                 kernel_constraint=None,kernel_regularization=None,bias_initializer='zeros',bias_regularizer=None,
                 bias_constraint=None,**kwargs):
        
        super(ErosionLiftingP4, self).__init__(**kwargs)

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
        #kernel_shape = self.kernel_size + (input_dim, self.num_filters)
        kernel_shape = (self.kernel_size[0], self.kernel_size[1], input_dim, self.num_filters)
        
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
        super(ErosionLiftingP4, self).build(input_shape)

    def call(self, x):

        output = []

        for i in range(4):

            kernel_rot = tf.experimental.numpy.rot90(self.kernel, k = i, axes = (0,1))
            res_filters = []
            for j in range(self.num_filters):
                res_filters.append(tf.reduce_sum(erosion2d(x, kernel_rot[...,j], self.strides, self.padding),axis=-1))

            output.append(tf.stack(res_filters, axis = -1))

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
            'activation': self.activation,
            'use_bias': self.use_bias,
        })
        return config
    
class OpeningLiftingP4(tf.keras.layers.Layer):

    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same', dilation_rate=(1,1), activation=None,use_bias=False,kernel_initializer='Zeros',
                 kernel_constraint=None,kernel_regularization=None,bias_initializer='zeros',bias_regularizer=None,
                 bias_constraint=None,**kwargs):
        
        super(OpeningLiftingP4, self).__init__(**kwargs)

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
        #kernel_shape = self.kernel_size + (input_dim, self.num_filters)
        kernel_shape = (self.kernel_size[0], self.kernel_size[1], input_dim, self.num_filters)
        
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
        super(OpeningLiftingP4, self).build(input_shape)

    def call(self, x):

        output = []

        for i in range(4):

            kernel_rot = tf.experimental.numpy.rot90(self.kernel, k = i, axes = (0,1))
            res_filters = []
            for j in range(self.num_filters):

                ero = erosion2d(x, kernel_rot[...,j], strides = (1,1), padding = self.padding)

                res_filters.append(tf.reduce_sum(dilation2d(ero, kernel_rot[...,j], self.strides, self.padding),axis=-1))

            output.append(tf.stack(res_filters, axis = -1))

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
            'activation': self.activation,
            'use_bias': self.use_bias,
        })
        return config
    
class ClosingLiftingP4(tf.keras.layers.Layer):

    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same', dilation_rate=(1,1), activation=None,use_bias=False,kernel_initializer='Zeros',
                 kernel_constraint=None,kernel_regularization=None,bias_initializer='zeros',bias_regularizer=None,
                 bias_constraint=None,**kwargs):
        
        super(ClosingLiftingP4, self).__init__(**kwargs)

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
        #kernel_shape = self.kernel_size + (input_dim, self.num_filters)
        kernel_shape = (self.kernel_size[0], self.kernel_size[1], input_dim, self.num_filters)
        

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
        super(ClosingLiftingP4, self).build(input_shape)

    def call(self, x):

        output = []

        for i in range(4):

            kernel_rot = tf.experimental.numpy.rot90(self.kernel, k = i, axes = (0,1))
            res_filters = []
            for j in range(self.num_filters):

                dil = dilation2d(x, kernel_rot[...,j], strides = (1,1), padding = self.padding)

                res_filters.append(tf.reduce_sum(erosion2d(dil, kernel_rot[...,j], self.strides, self.padding),axis=-1))

            output.append(tf.stack(res_filters, axis = -1))

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
            'activation': self.activation,
            'use_bias': self.use_bias,
        })
        return config
    
# P4 Layers

class DilationP4(tf.keras.layers.Layer):

    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same', dilation_rate=(1,1), activation=None,use_bias=False,kernel_initializer='Zeros',
                 kernel_constraint=None,kernel_regularization=None,bias_initializer='zeros',bias_regularizer=None,
                 bias_constraint=None,**kwargs):
        
        super(DilationP4, self).__init__(**kwargs)

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

        #kernel_shape = self.kernel_size + (3, input_dim, self.num_filters)
        kernel_shape = (self.kernel_size[0], self.kernel_size[1], 3, input_dim, self.num_filters)
        

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
        super(DilationP4, self).build(input_shape)

    def call(self, x):


        y = tf.concat([x[:,-1:, ...], x, x[:,0:1, ...]], axis = 1)

        res_rota = []

        for j in range(4):

            kernel_rot = tf.experimental.numpy.rot90(self.kernel, k = j, axes = (0,1) )

            res_SE_depth = []

            for k in range(3):

                res_filters = []

                for i in range(self.num_filters):

                    res_filters.append(dilation2d(y[:,j + k,...], kernel_rot[..., k, :,i],self.strides, self.padding))

                res_filters = tf.stack(res_filters, axis=-1)

                res_SE_depth.append(res_filters)

            res_SE_depth = tf.stack(res_SE_depth, axis=-1)
            res_SE_depth = tf.reduce_max(res_SE_depth, axis = -1)
            res_rota.append(res_SE_depth)

        output = tf.stack(res_rota, axis= 1)

        output = tf.reduce_sum(output, axis = -2)

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
            'activation': self.activation,
            'use_bias': self.use_bias,
        })
        return config

class ErosionP4(tf.keras.layers.Layer):

    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same', dilation_rate=(1,1), activation=None,use_bias=False,kernel_initializer='Zeros',
                 kernel_constraint=None,kernel_regularization=None,bias_initializer='zeros',bias_regularizer=None,
                 bias_constraint=None,**kwargs):
        
        super(ErosionP4, self).__init__(**kwargs)

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

        #kernel_shape = self.kernel_size + (3, input_dim, self.num_filters)
        kernel_shape = (self.kernel_size[0], self.kernel_size[1], 3, input_dim, self.num_filters)
        
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
        super(ErosionP4, self).build(input_shape)

    def call(self, x):

        #kernel_ero = tf.reverse(self.kernel, axis=[2])
        kernel_ero = []

        kernel_ero.append(tf.experimental.numpy.rot90(self.kernel[:,:,2,...], k = 3, axes = (0,1)))
        kernel_ero.append(self.kernel[:,:,1,...]  )     
        kernel_ero.append(tf.experimental.numpy.rot90(self.kernel[:,:,0,...], k = 1, axes = (0,1)))
        kernel_ero = tf.stack(kernel_ero, axis = 2)
        

        y = tf.concat([x[:,-1:, ...], x, x[:,0:1, ...]], axis = 1)

        res_rota = []

        for j in range(4):

            kernel_rot = tf.experimental.numpy.rot90(kernel_ero, k = j, axes = (0,1) )

            res_SE_depth = []

            for k in range(3):

                res_filters = []

                for i in range(self.num_filters):

                    res_filters.append(erosion2d(y[:,j + k,...], kernel_rot[..., k, :,i],self.strides, self.padding))

                res_filters = tf.stack(res_filters, axis=-1)

                res_SE_depth.append(res_filters)

            res_SE_depth = tf.stack(res_SE_depth, axis=-1)
            res_SE_depth = tf.reduce_min(res_SE_depth, axis = -1)
            res_rota.append(res_SE_depth)

        output = tf.stack(res_rota, axis= 1)

        output = tf.reduce_sum(output, axis = -2)

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
            'activation': self.activation,
            'use_bias': self.use_bias,
        })
        return config
    
class OpeningP4(tf.keras.layers.Layer):

    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same', dilation_rate=(1,1), activation=None,use_bias=False,kernel_initializer='Zeros',
                 kernel_constraint=None,kernel_regularization=None,bias_initializer='zeros',bias_regularizer=None,
                 bias_constraint=None,**kwargs):
        
        super(OpeningP4, self).__init__(**kwargs)

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

        #kernel_shape = self.kernel_size + (3, input_dim, self.num_filters)
        kernel_shape = (self.kernel_size[0], self.kernel_size[1], 3, input_dim, self.num_filters)
        
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
        super(OpeningP4, self).build(input_shape)

    def call(self, x):

        y = tf.concat([x[:,-1:, ...], x, x[:,0:1, ...]], axis = 1)

        res_rota = []

        kernel_ero = []

        kernel_ero.append(tf.experimental.numpy.rot90(self.kernel[:,:,2,...], k = 3, axes = (0,1)))
        kernel_ero.append(self.kernel[:,:,1,...]  )     
        kernel_ero.append(tf.experimental.numpy.rot90(self.kernel[:,:,0,...], k = 1, axes = (0,1)))
        kernel_ero = tf.stack(kernel_ero, axis = 2)

        for j in range(4):

            kernel_rot = tf.experimental.numpy.rot90(kernel_ero, k = j, axes = (0,1) )

            res_SE_depth = []

            for k in range(3):

                res_filters = []

                for i in range(self.num_filters):

                    res_filters.append(erosion2d(y[:,j + k,...], kernel_rot[..., k, :,i],self.strides, self.padding))

                res_filters = tf.stack(res_filters, axis=-1)

                res_SE_depth.append(res_filters)

            res_SE_depth = tf.stack(res_SE_depth, axis=-1)
            res_SE_depth = tf.reduce_min(res_SE_depth, axis = -1)
            res_rota.append(res_SE_depth)
        

        ero = tf.stack(res_rota, axis= 1)

        ero = tf.concat([ero[:,-1:, ...], ero, ero[:,0:1, ...]], axis = 1)

        res_rota = []

        for j in range(4):

            kernel_rot = tf.experimental.numpy.rot90(self.kernel, k = j, axes = (0,1) )

            res_SE_depth = []

            for k in range(3):

                res_filters = []

                for i in range(self.num_filters):

                    res_filters.append(dilation2d(ero[:,j + k,..., i], kernel_rot[..., k, :,i],self.strides, self.padding))

                res_filters = tf.stack(res_filters, axis=-1)

                res_SE_depth.append(res_filters)

            res_SE_depth = tf.stack(res_SE_depth, axis=-1)
            res_SE_depth = tf.reduce_max(res_SE_depth, axis = -1)
            res_rota.append(res_SE_depth)

        open_res = tf.stack(res_rota, axis= 1)

        output = tf.reduce_sum(open_res, axis = -2)

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
            'activation': self.activation,
            'use_bias': self.use_bias,
        })
        return config
    
class ClosingP4(tf.keras.layers.Layer):

    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same', dilation_rate=(1,1), activation=None,use_bias=False,kernel_initializer='Zeros',
                 kernel_constraint=None,kernel_regularization=None,bias_initializer='zeros',bias_regularizer=None,
                 bias_constraint=None,**kwargs):
        
        super(ClosingP4, self).__init__(**kwargs)

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

        #kernel_shape = self.kernel_size + (3, input_dim, self.num_filters)
        kernel_shape = (self.kernel_size[0], self.kernel_size[1], 3, input_dim, self.num_filters)
        
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
        super(ClosingP4, self).build(input_shape)

    def call(self, x):

        y = tf.concat([x[:,-1:, ...], x, x[:,0:1, ...]], axis = 1)

        res_rota = []

        for j in range(4):

            kernel_rot = tf.experimental.numpy.rot90(self.kernel, k = j, axes = (0,1) )

            res_SE_depth = []

            for k in range(3):

                res_filters = []

                for i in range(self.num_filters):

                    res_filters.append(dilation2d(y[:,j + k,...], kernel_rot[..., k, :,i],self.strides, self.padding))

                res_filters = tf.stack(res_filters, axis=-1)

                res_SE_depth.append(res_filters)

            res_SE_depth = tf.stack(res_SE_depth, axis=-1)
            res_SE_depth = tf.reduce_max(res_SE_depth, axis = -1)
            res_rota.append(res_SE_depth)

        dil = tf.stack(res_rota, axis= 1)

        dil = tf.concat([dil[:,-1:, ...], dil, dil[:,0:1, ...]], axis = 1)

        res_rota = []

        kernel_ero = []

        kernel_ero.append(tf.experimental.numpy.rot90(self.kernel[:,:,2,...], k = 3, axes = (0,1)))
        kernel_ero.append(self.kernel[:,:,1,...]  )     
        kernel_ero.append(tf.experimental.numpy.rot90(self.kernel[:,:,0,...], k = 1, axes = (0,1)))
        kernel_ero = tf.stack(kernel_ero, axis = 2)
        
        for j in range(4):

            kernel_rot = tf.experimental.numpy.rot90(kernel_ero, k = j, axes = (0,1) )

            res_SE_depth = []

            for k in range(3):

                res_filters = []

                for i in range(self.num_filters):

                    res_filters.append(erosion2d(dil[:,j + k,..., i], kernel_rot[..., k, :,i],self.strides, self.padding))

                res_filters = tf.stack(res_filters, axis=-1)

                res_SE_depth.append(res_filters)

            res_SE_depth = tf.stack(res_SE_depth, axis=-1)
            res_SE_depth = tf.reduce_min(res_SE_depth, axis = -1)
            res_rota.append(res_SE_depth)

        close_res = tf.stack(res_rota, axis= 1)

        output = tf.reduce_sum(close_res, axis = -2)
        
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
            'activation': self.activation,
            'use_bias': self.use_bias,
        })
        return config
    
###  MAX-TIMES-PLUS MorphoLayer on P4 (Dilation, Erosion, Opening and Closing)

# P4 Lifting Layers

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
        #kernel_shape = self.kernel_size + (input_dim, self.num_filters)
        kernel_shape = (self.kernel_size[0], self.kernel_size[1], input_dim, self.num_filters)

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

        y = tf.image.extract_patches(x, sizes=(1, self.kernel_size[0], self.kernel_size[1], 1), strides = (1,self.strides[0],self.strides[1],1), rates=(1,self.rates[0], self.rates[1],1 ), padding=self.padding.upper())
            
        y = tf.reshape(y, shape = (-1, y.shape[1], y.shape[2], self.kernel_size[0]*self.kernel_size[1], x.shape[3]))
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
            'activation': self.activation,
            'use_bias': self.use_bias,
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
        #kernel_shape = self.kernel_size + (input_dim, self.num_filters)
        kernel_shape = (self.kernel_size[0], self.kernel_size[1], input_dim, self.num_filters)

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

        # For duality, we take symmetic with respect to center of Structural Element.
        kernel_ero = tf.experimental.numpy.rot90(self.kernel, k = 2, axes = (0,1))
        timesKernel_ero = tf.experimental.numpy.rot90(self.timesKernel, k = 2, axes = (0,1))

        y = tf.image.extract_patches(x, sizes=(1, self.kernel_size[0], self.kernel_size[1], 1), strides = (1,self.strides[0],self.strides[1],1), rates=(1,self.rates[0], self.rates[1],1 ), padding=self.padding.upper())
            
        y = tf.reshape(y, shape = (-1, y.shape[1], y.shape[2], self.kernel_size[0]*self.kernel_size[1], x.shape[3]))
        y = tf.tile(tf.expand_dims(y, axis = -1), [1, 1, 1, 1, 1, self.num_filters])

        res = []
        for i in range(4):

            kernel_rot = tf.experimental.numpy.rot90(kernel_ero, k = i, axes = (0,1))
            timesKernel_rot = tf.experimental.numpy.rot90(timesKernel_ero, k = i, axes = (0,1))

            kernel_rot = tf.reshape(kernel_rot, (self.kernel.shape[0]*self.kernel.shape[1], self.kernel.shape[2], self.kernel.shape[3]))

            timesKernel_rot = tf.reshape(timesKernel_rot, (self.timesKernel.shape[0]*self.timesKernel.shape[1], self.timesKernel.shape[2], self.timesKernel.shape[3]))

            res_rot = tf.divide(tf.add(y, -kernel_rot), timesKernel_rot + tf.keras.backend.epsilon())
            res.append(tf.reduce_sum(tf.reduce_min(res_rot, axis = -3), axis = -2))
        
        output = tf.stack(res, axis = 1)

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
            'activation': self.activation,
            'use_bias': self.use_bias,
        })
        return config

class MaxTimesPlusOpeningLiftingP4(tf.keras.layers.Layer):

    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same', dilation_rate=(1,1), activation=None,use_bias=False,kernel_initializer='Zeros',
                 kernel_constraint=None,kernel_regularization=None, timesKernel_initializer = 'Ones', timesKernel_regularizer=None, timesKernel_constraint=tf.keras.constraints.NonNeg(), bias_initializer='zeros',bias_regularizer=None,
                 bias_constraint=None,**kwargs):
        
        super(MaxTimesPlusOpeningLiftingP4, self).__init__(**kwargs)

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
        #kernel_shape = self.kernel_size + (input_dim, self.num_filters)
        kernel_shape = (self.kernel_size[0], self.kernel_size[1], input_dim, self.num_filters)

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
        super(MaxTimesPlusOpeningLiftingP4, self).build(input_shape)

    def call(self, x):

        kernel_ero = tf.reverse(self.kernel, axis=[0,1])
        timesKernel_ero = tf.reverse(self.timesKernel, axis=[0,1])

        y = tf.image.extract_patches(x, sizes=(1,self.kernel_size[0], self.kernel_size[1],1), strides = (1, 1, 1, 1), rates=(1,self.rates[0],self.rates[1],1), padding=self.padding.upper())
            
        y = tf.reshape(y, shape = (-1, y.shape[1], y.shape[2], self.kernel_size[0]*self.kernel_size[1], x.shape[3]))
        y = tf.tile(tf.expand_dims(y, axis = -1), [1, 1, 1, 1, 1, self.num_filters])

        res = []
        for i in range(4):

            kernel_rot = tf.experimental.numpy.rot90(self.kernel, k = i, axes = (0,1))
            timesKernel_rot = tf.experimental.numpy.rot90(self.timesKernel, k = i, axes = (0,1))

            kernel_ero_rot = tf.experimental.numpy.rot90(kernel_ero, k = i, axes = (0,1))
            timesKernel_ero_rot = tf.experimental.numpy.rot90(timesKernel_ero, k = i, axes = (0,1))
            

            kernel_rot = tf.reshape(kernel_rot, (self.kernel.shape[0]*self.kernel.shape[1], self.kernel.shape[2], self.kernel.shape[3]))

            timesKernel_rot = tf.reshape(timesKernel_rot, (self.timesKernel.shape[0]*self.timesKernel.shape[1], self.timesKernel.shape[2], self.timesKernel.shape[3]))

            kernel_ero_rot = tf.reshape(kernel_ero_rot, (self.kernel.shape[0]*self.kernel.shape[1], self.kernel.shape[2], self.kernel.shape[3]))

            timesKernel_ero_rot = tf.reshape(timesKernel_ero_rot, (self.timesKernel.shape[0]*self.timesKernel.shape[1], self.timesKernel.shape[2], self.timesKernel.shape[3]))


            ero_rot = tf.reduce_min(tf.divide(tf.add(y, -kernel_ero_rot), timesKernel_ero_rot + tf.keras.backend.epsilon()), axis = -3)

            ero_patched = []

            for k in range(self.num_filters):

                ero_patch = tf.image.extract_patches(ero_rot[..., k], sizes=(1,self.kernel_size[0], self.kernel_size[1],1), strides = (1,self.strides[0],self.strides[1],1), rates=(1,self.rates[0], self.rates[1],1 ), padding=self.padding.upper())
                ero_patch = tf.reshape(ero_patch, shape= (-1, ero_patch.shape[1], ero_patch.shape[2], self.kernel_size[0]*self.kernel_size[1], ero_rot.shape[3]))
                ero_patched.append(ero_patch)

            ero_patched = tf.stack(ero_patched, axis = -1)
            res_rot = tf.add(tf.multiply(ero_patched, timesKernel_rot), kernel_rot)
            res.append(tf.reduce_sum(tf.reduce_max(res_rot, axis = -3), axis = -2))
        
        output = tf.stack(res, axis = 1)

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
            'activation': self.activation,
            'use_bias': self.use_bias,
        })
        return config

class MaxTimesPlusClosingLiftingP4(tf.keras.layers.Layer):

    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same', dilation_rate=(1,1), activation=None,use_bias=False,kernel_initializer='Zeros',
                 kernel_constraint=None,kernel_regularization=None, timesKernel_initializer = 'Ones', timesKernel_regularizer=None, timesKernel_constraint=tf.keras.constraints.NonNeg(), bias_initializer='zeros',bias_regularizer=None,
                 bias_constraint=None,**kwargs):
        
        super(MaxTimesPlusClosingLiftingP4, self).__init__(**kwargs)

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
        #kernel_shape = self.kernel_size + (input_dim, self.num_filters)
        kernel_shape = (self.kernel_size[0], self.kernel_size[1], input_dim, self.num_filters)

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
        super(MaxTimesPlusClosingLiftingP4, self).build(input_shape)

    def call(self, x):

        kernel_ero = tf.reverse(self.kernel, axis=[0,1])
        timesKernel_ero = tf.reverse(self.timesKernel, axis=[0,1])
        
        y = tf.image.extract_patches(x, sizes=(1,self.kernel_size[0], self.kernel_size[1],1), strides = (1, 1, 1, 1), rates=(1,self.rates[0],self.rates[1],1), padding=self.padding.upper())
      
        y = tf.reshape(y, shape = (-1, y.shape[1], y.shape[2], self.kernel_size[0]*self.kernel_size[1], x.shape[3]))
        y = tf.tile(tf.expand_dims(y, axis = -1), [1, 1, 1, 1, 1, self.num_filters])

        res = []
        for i in range(4):

            kernel_rot = tf.experimental.numpy.rot90(self.kernel, k = i, axes = (0,1))
            timesKernel_rot = tf.experimental.numpy.rot90(self.timesKernel, k = i, axes = (0,1))

            kernel_ero_rot = tf.experimental.numpy.rot90(kernel_ero, k = i, axes = (0,1))
            timesKernel_ero_rot = tf.experimental.numpy.rot90(timesKernel_ero, k = i, axes = (0,1))

            kernel_rot = tf.reshape(kernel_rot, (self.kernel.shape[0]*self.kernel.shape[1], self.kernel.shape[2], self.kernel.shape[3]))

            timesKernel_rot = tf.reshape(timesKernel_rot, (self.timesKernel.shape[0]*self.timesKernel.shape[1], self.timesKernel.shape[2], self.timesKernel.shape[3]))

            kernel_ero_rot = tf.reshape(kernel_ero_rot, (self.kernel.shape[0]*self.kernel.shape[1], self.kernel.shape[2], self.kernel.shape[3]))

            timesKernel_ero_rot = tf.reshape(timesKernel_ero_rot, (self.timesKernel.shape[0]*self.timesKernel.shape[1], self.timesKernel.shape[2], self.timesKernel.shape[3]))

            dil_rot = tf.reduce_max(tf.add(tf.multiply(y, timesKernel_rot), kernel_rot), axis = -3)

            dil_patched = []

            for k in range(self.num_filters):

                dil_patch = tf.image.extract_patches(dil_rot[..., k], sizes=(1,self.kernel_size[0], self.kernel_size[1],1), strides = (1,self.strides[0],self.strides[1],1), rates=(1,self.rates[0], self.rates[1],1 ), padding=self.padding.upper())
                dil_patch = tf.reshape(dil_patch, shape= (-1, dil_patch.shape[1], dil_patch.shape[2], self.kernel_size[0]*self.kernel_size[1], dil_rot.shape[3]))
                dil_patched.append(dil_patch)

            dil_patched = tf.stack(dil_patched, axis = -1)

            res_rot = tf.divide(tf.add(dil_patched, -kernel_ero_rot), timesKernel_ero_rot + tf.keras.backend.epsilon())
            res.append(tf.reduce_sum(tf.reduce_min(res_rot, axis = -3), axis = -2))
        
        output = tf.stack(res, axis = 1)

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
            'activation': self.activation,
            'use_bias': self.use_bias,
        })
        return config

# P4 Layers

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
        kernel_shape = (self.kernel_size[0], self.kernel_size[1],3, input_dim, self.num_filters)

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
        y = tf.extract_volume_patches(y, ksizes=(1, 1, self.kernel_size[0], self.kernel_size[1], 1), strides = (1, 1, self.strides[0], self.strides[1], 1),  padding=self.padding.upper())
            
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
            'activation': self.activation,
            'use_bias': self.use_bias,
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
        #kernel_shape = self.kernel_size + (3, input_dim, self.num_filters)
        kernel_shape = (self.kernel_size[0], self.kernel_size[1], 3, input_dim, self.num_filters)

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
        y = tf.extract_volume_patches(y, ksizes=(1, 1, self.kernel_size[0], self.kernel_size[1], 1), strides = (1, 1, self.strides[0], self.strides[1], 1), padding=self.padding.upper())
            
        y = tf.reshape(y, shape = (-1, y.shape[1], y.shape[2], y.shape[3], self.kernel_size[0]*self.kernel_size[1], x.shape[4]))
        y = tf.tile(tf.expand_dims(y, axis = -1), [1, 1, 1, 1, 1, 1, self.num_filters])

        y = tf.transpose(y, perm = [0, 2, 3, 4, 1, 5, 6])

        kernel_ero = []

        kernel_ero.append(tf.experimental.numpy.rot90(self.kernel[:,:,2,...], k = 3, axes = (0,1)))
        kernel_ero.append(self.kernel[:,:,1,...]  )     
        kernel_ero.append(tf.experimental.numpy.rot90(self.kernel[:,:,0,...], k = 1, axes = (0,1)))
        kernel_ero = tf.stack(kernel_ero, axis = 2)

        timesKernel_ero = []

        timesKernel_ero.append(tf.experimental.numpy.rot90(self.timesKernel[:,:,2,...], k = 3, axes = (0,1)))
        timesKernel_ero.append(self.timesKernel[:,:,1,...]  )     
        timesKernel_ero.append(tf.experimental.numpy.rot90(self.timesKernel[:,:,0,...], k = 1, axes = (0,1)))
        timesKernel_ero = tf.stack(timesKernel_ero, axis = 2)


        kernel_ero = tf.reverse(kernel_ero, axis=[0,1])
        timesKernel_ero = tf.reverse(timesKernel_ero, axis=[0,1])


        res = []
        for i in range(4):

            kernel_rot = tf.experimental.numpy.rot90(kernel_ero, k = i, axes = (0,1))
            timesKernel_rot = tf.experimental.numpy.rot90(timesKernel_ero, k = i, axes = (0,1))

            kernel_rot = tf.reshape(kernel_rot, (self.kernel.shape[0]*self.kernel.shape[1], self.kernel.shape[2], self.kernel.shape[3], self.timesKernel.shape[4]))

            timesKernel_rot = tf.reshape(timesKernel_rot, (self.timesKernel.shape[0]*self.timesKernel.shape[1], self.timesKernel.shape[2], self.timesKernel.shape[3], self.timesKernel.shape[4]))

            res_rot = tf.divide(tf.add(y[...,i:i+3,:,:], -kernel_rot), timesKernel_rot + tf.keras.backend.epsilon())
            res.append(tf.reduce_sum(tf.reduce_min(res_rot, axis = (-4, -3)), axis = -2))
        
        output = tf.stack(res, axis = 1)

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
            'activation': self.activation,
            'use_bias': self.use_bias,
        })
        return config

class MaxTimesPlusOpeningP4(tf.keras.layers.Layer):

    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same', dilation_rate=(1,1), activation=None,use_bias=False,kernel_initializer='Zeros',
                 kernel_constraint=None,kernel_regularization=None, timesKernel_initializer = 'Ones', timesKernel_regularizer=None, timesKernel_constraint=tf.keras.constraints.NonNeg(), bias_initializer='zeros',bias_regularizer=None,
                 bias_constraint=None,**kwargs):
        
        super(MaxTimesPlusOpeningP4, self).__init__(**kwargs)

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
        #kernel_shape = self.kernel_size + (3, input_dim, self.num_filters)
        kernel_shape = (self.kernel_size[0], self.kernel_size[1], 3, input_dim, self.num_filters)

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
        super(MaxTimesPlusOpeningP4, self).build(input_shape)

    def call(self, x):

        y = tf.concat([x[:,-1:, ...], x, x[:,0:1, ...]], axis = 1)
        y = tf.extract_volume_patches(y, ksizes=(1, 1, self.kernel_size[0], self.kernel_size[1], 1), strides = (1, 1, 1,1, 1), padding=self.padding.upper())
            
        y = tf.reshape(y, shape = (-1, y.shape[1], y.shape[2], y.shape[3], self.kernel_size[0]*self.kernel_size[1], x.shape[4]))
        y = tf.tile(tf.expand_dims(y, axis = -1), [1, 1, 1, 1, 1, 1, self.num_filters])

        y = tf.transpose(y, perm = [0, 2, 3, 4, 1, 5, 6])

        kernel_ero = []

        kernel_ero.append(tf.experimental.numpy.rot90(self.kernel[:,:,2,...], k = 3, axes = (0,1)))
        kernel_ero.append(self.kernel[:,:,1,...]  )     
        kernel_ero.append(tf.experimental.numpy.rot90(self.kernel[:,:,0,...], k = 1, axes = (0,1)))
        kernel_ero = tf.stack(kernel_ero, axis = 2)

        timesKernel_ero = []

        timesKernel_ero.append(tf.experimental.numpy.rot90(self.timesKernel[:,:,2,...], k = 3, axes = (0,1)))
        timesKernel_ero.append(self.timesKernel[:,:,1,...]  )     
        timesKernel_ero.append(tf.experimental.numpy.rot90(self.timesKernel[:,:,0,...], k = 1, axes = (0,1)))
        timesKernel_ero = tf.stack(timesKernel_ero, axis = 2)

        kernel_ero = tf.reverse(kernel_ero, axis=[0,1])
        timesKernel_ero = tf.reverse(timesKernel_ero, axis=[0,1])

        res = []
        for i in range(4):

            kernel_ero_rot = tf.experimental.numpy.rot90(kernel_ero, k = i, axes = (0,1))
            timesKernel_ero_rot = tf.experimental.numpy.rot90(timesKernel_ero, k = i, axes = (0,1))
            
            kernel_ero_rot = tf.reshape(kernel_ero_rot, (self.kernel.shape[0]*self.kernel.shape[1], self.kernel.shape[2], self.kernel.shape[3], self.kernel.shape[4]))

            timesKernel_ero_rot = tf.reshape(timesKernel_ero_rot, (self.timesKernel.shape[0]*self.timesKernel.shape[1], self.timesKernel.shape[2], self.timesKernel.shape[3], self.timesKernel.shape[4]))

            res_rot = tf.divide(tf.add(y[...,i:i+3,:,:], -kernel_ero_rot), timesKernel_ero_rot + tf.keras.backend.epsilon())
            res.append(tf.reduce_min(res_rot, axis = (-4, -3)))
        
        ero = tf.stack(res, axis = 1)

        ero = tf.concat([ero[:,-1:, ...], ero, ero[:,0:1, ...]], axis = 1)
        ero_patches = []
        for j in range(self.num_filters):
            
            ero_patch = tf.extract_volume_patches(ero[...,j], ksizes=(1, 1, self.kernel_size[0], self.kernel_size[1], 1), strides = (1, 1, self.strides[0], self.strides[1], 1), padding=self.padding.upper())
            
            ero_patch = tf.reshape(ero_patch, shape = (-1, ero_patch.shape[1], ero_patch.shape[2], ero_patch.shape[3], self.kernel_size[0]*self.kernel_size[1], x.shape[4]))
            
            ero_patches.append(ero_patch)

        ero_patches = tf.stack(ero_patches, axis = -1)

        ero_patches = tf.transpose(ero_patches, perm = [0, 2, 3, 4, 1, 5, 6])

        res = []
        for i in range(4):

            kernel_rot = tf.experimental.numpy.rot90(self.kernel, k = i, axes = (0,1))
            timesKernel_rot = tf.experimental.numpy.rot90(self.timesKernel, k = i, axes = (0,1))

            kernel_rot = tf.reshape(kernel_rot, (self.kernel.shape[0]*self.kernel.shape[1], self.kernel.shape[2], self.kernel.shape[3], self.timesKernel.shape[4]))

            timesKernel_rot = tf.reshape(timesKernel_rot, (self.timesKernel.shape[0]*self.timesKernel.shape[1], self.timesKernel.shape[2], self.timesKernel.shape[3], self.timesKernel.shape[4]))

            res_rot = tf.add(tf.multiply(ero_patches[...,i:i+3,:,:], timesKernel_rot), kernel_rot)
            res.append(tf.reduce_sum(tf.reduce_max(res_rot, axis = (-4, -3)), axis = -2))
        
        output = tf.stack(res, axis = 1)

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
            'activation': self.activation,
            'use_bias': self.use_bias,
        })
        return config

class MaxTimesPlusClosingP4(tf.keras.layers.Layer):

    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same', dilation_rate=(1,1), activation=None,use_bias=False,kernel_initializer='Zeros',
                 kernel_constraint=None,kernel_regularization=None, timesKernel_initializer = 'Ones', timesKernel_regularizer=None, timesKernel_constraint=tf.keras.constraints.NonNeg(), bias_initializer='zeros',bias_regularizer=None,
                 bias_constraint=None,**kwargs):
        
        super(MaxTimesPlusClosingP4, self).__init__(**kwargs)

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
        #kernel_shape = self.kernel_size + (3, input_dim, self.num_filters)
        kernel_shape = (self.kernel_size[0], self.kernel_size[1], 3, input_dim, self.num_filters)

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
        super(MaxTimesPlusClosingP4, self).build(input_shape)

    def call(self, x):

        y = tf.concat([x[:,-1:, ...], x, x[:,0:1, ...]], axis = 1)
        y = tf.extract_volume_patches(y, ksizes=(1, 1, self.kernel_size[0], self.kernel_size[1], 1), strides = (1, 1, 1,1, 1), padding=self.padding.upper())
            
        y = tf.reshape(y, shape = (-1, y.shape[1], y.shape[2], y.shape[3], self.kernel_size[0]*self.kernel_size[1], x.shape[4]))
        y = tf.tile(tf.expand_dims(y, axis = -1), [1, 1, 1, 1, 1, 1, self.num_filters])

        y = tf.transpose(y, perm = [0, 2, 3, 4, 1, 5, 6])

        kernel_ero = []

        kernel_ero.append(tf.experimental.numpy.rot90(self.kernel[:,:,2,...], k = 3, axes = (0,1)))
        kernel_ero.append(self.kernel[:,:,1,...]  )     
        kernel_ero.append(tf.experimental.numpy.rot90(self.kernel[:,:,0,...], k = 1, axes = (0,1)))
        kernel_ero = tf.stack(kernel_ero, axis = 2)

        timesKernel_ero = []

        timesKernel_ero.append(tf.experimental.numpy.rot90(self.timesKernel[:,:,2,...], k = 3, axes = (0,1)))
        timesKernel_ero.append(self.timesKernel[:,:,1,...]  )     
        timesKernel_ero.append(tf.experimental.numpy.rot90(self.timesKernel[:,:,0,...], k = 1, axes = (0,1)))
        timesKernel_ero = tf.stack(timesKernel_ero, axis = 2)

        kernel_ero = tf.reverse(kernel_ero, axis=[0,1])
        timesKernel_ero = tf.reverse(timesKernel_ero, axis=[0,1])

        res = []
        for i in range(4):

            kernel_rot = tf.experimental.numpy.rot90(self.kernel, k = i, axes = (0,1))
            timesKernel_rot = tf.experimental.numpy.rot90(self.timesKernel, k = i, axes = (0,1))

            kernel_rot = tf.reshape(kernel_rot, (self.kernel.shape[0]*self.kernel.shape[1], self.kernel.shape[2], self.kernel.shape[3], self.timesKernel.shape[4]))

            timesKernel_rot = tf.reshape(timesKernel_rot, (self.timesKernel.shape[0]*self.timesKernel.shape[1], self.timesKernel.shape[2], self.timesKernel.shape[3], self.timesKernel.shape[4]))

            res_rot = tf.add(tf.multiply(y[...,i:i+3,:,:], timesKernel_rot), kernel_rot)
            
            res.append(tf.reduce_max(res_rot, axis = (-4, -3)))
        
        dil = tf.stack(res, axis = 1)

        dil = tf.concat([dil[:,-1:, ...], dil, dil[:,0:1, ...]], axis = 1)
        dil_patches = []
        for j in range(self.num_filters):
            
            dil_patch = tf.extract_volume_patches(dil[...,j], ksizes=(1, 1, self.kernel_size[0], self.kernel_size[1], 1), strides = (1, 1, self.strides[0], self.strides[1], 1), padding=self.padding.upper())
            
            dil_patch = tf.reshape(dil_patch, shape = (-1, dil_patch.shape[1], dil_patch.shape[2], dil_patch.shape[3], self.kernel_size[0]*self.kernel_size[1], x.shape[4]))
            
            dil_patches.append(dil_patch)

        dil_patches = tf.stack(dil_patches, axis = -1)

        dil_patches = tf.transpose(dil_patches, perm = [0, 2, 3, 4, 1, 5, 6])

        res = []
        for i in range(4):

            kernel_rot = tf.experimental.numpy.rot90(kernel_ero, k = i, axes = (0,1))
            timesKernel_rot = tf.experimental.numpy.rot90(timesKernel_ero, k = i, axes = (0,1))

            kernel_rot = tf.reshape(kernel_rot, (self.kernel.shape[0]*self.kernel.shape[1], self.kernel.shape[2], self.kernel.shape[3], self.timesKernel.shape[4]))

            timesKernel_rot = tf.reshape(timesKernel_rot, (self.timesKernel.shape[0]*self.timesKernel.shape[1], self.timesKernel.shape[2], self.timesKernel.shape[3], self.timesKernel.shape[4]))

            res_rot = tf.divide(tf.add(dil_patches[...,i:i+3,:,:], -kernel_rot), timesKernel_rot + tf.keras.backend.epsilon())

            res.append(tf.reduce_sum(tf.reduce_min(res_rot, axis = (-4, -3)), axis = -2))
        
        output = tf.stack(res, axis = 1)

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
            'activation': self.activation,
            'use_bias': self.use_bias,
        })
        return config



class SpatialAveragePoolingP4(tf.keras.layers.Layer):

    def __init__(self, **kwargs):

        super(SpatialAveragePoolingP4, self).__init__()

    def build(self, input_shape):

        self.inputShape = input_shape

    def call(self, inputs):

        results = tf.reduce_mean(inputs, axis=[2, 3])
        return results