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

                res_filters.append(tf.reduce_sum(erosion2d(tf.math.divide(self.timesKernel[:,j], x + tf.keras.backend.epsilon()), kernel_rot[...,j], self.strides, self.padding),axis=-1))

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

class AveragePoolingSpatial3D(tf.keras.layers.Layer):

    def __init__(self, **kwargs):

        super(AveragePoolingSpatial3D, self).__init__()

    def build(self, input_shape):

        self.inputShape = input_shape

    def call(self, inputs):

        results = tf.reduce_mean(inputs, axis=[2, 3])
        return results