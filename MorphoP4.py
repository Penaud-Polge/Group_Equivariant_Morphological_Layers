# All code stamped with "From MORPHOLAYERS" come from https://github.com/Jacobiano/morpholayers/


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

                    res_filters.append(tf.reduce_sum(dilation2d(y[:,j + k,...], kernel_rot[..., k, :,i],self.strides, self.padding), axis = -1))

                res_filters = tf.stack(res_filters, axis=-1)

                res_SE_depth.append(res_filters)

            res_SE_depth = tf.stack(res_SE_depth, axis=-1)
            res_SE_depth = tf.reduce_max(res_SE_depth, axis = -1)
            res_rota.append(res_SE_depth)

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
        super(ErosionP4, self).build(input_shape)

    def call(self, x):


        y = tf.concat([x[:,-1:, ...], x, x[:,0:1, ...]], axis = 1)

        res_rota = []

        for j in range(4):

            kernel_rot = tf.experimental.numpy.rot90(self.kernel, k = j, axes = (0,1) )

            res_SE_depth = []

            for k in range(3):

                res_filters = []

                for i in range(self.num_filters):

                    res_filters.append(tf.reduce_sum(erosion2d(y[:,j + k,...], kernel_rot[..., k, :,i],self.strides, self.padding), axis = -1))

                res_filters = tf.stack(res_filters, axis=-1)

                res_SE_depth.append(res_filters)

            res_SE_depth = tf.stack(res_SE_depth, axis=-1)
            res_SE_depth = tf.reduce_min(res_SE_depth, axis = -1)
            res_rota.append(res_SE_depth)

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
    


class Dilation3Dxy(tf.keras.layers.Layer):
    '''
     Dilation 3D Layer: Dilation for now assuming channel last
    '''
    def __init__(self, filters, kernel_size,strides=(1, 1),padding='same', dilation_rate=(1,1), kernel_initializer='RandomUniform',
    kernel_constraint=None,kernel_regularization=None, **kwargs):
        super(Dilation3Dxy, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.num_filters= filters
        self.strides = strides
        self.padding = padding
        self.rates=dilation_rate
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)
        # for we are assuming channel last
        self.channel_axis = -1

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim,self.num_filters)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',constraint =self.kernel_constraint,regularizer=self.kernel_regularization)
        super(Dilation3Dxy, self).build(input_shape)


    def call(self,x):
        #print('x.shape',x.shape)
        #print('self.kernel.shape',self.kernel.shape)
        tile=tf.reshape(x,[tf.shape(x)[0],tf.shape(x)[1],tf.shape(x)[2]*tf.shape(x)[3],tf.shape(x)[4]])
        #print('tile.shape',tile.shape)
        res=[]
        for i in range(self.num_filters):
            res.append(dilation2d(tile, self.kernel[..., i],self.strides, self.padding))
        res=tf.stack(res,axis=-1)
        #print('res.shape',res.shape)
        res=tf.reshape(res,[tf.shape(res)[0],tf.shape(res)[1],tf.shape(x)[2],tf.shape(x)[3],tf.shape(res)[3]*tf.shape(res)[4]])
        return res
    
class Dilation3Dyz(tf.keras.layers.Layer):
    '''
     Dilation 3D Layer: Dilation for now assuming channel last
    '''
    def __init__(self, filters, kernel_size,strides=(1, 1),padding='same', dilation_rate=(1,1), kernel_initializer='RandomUniform',
    kernel_constraint=None,kernel_regularization=None, **kwargs):
        super(Dilation3Dyz, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.num_filters= filters
        self.strides = strides
        self.padding = padding
        self.rates=dilation_rate
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)
        # for we are assuming channel last
        self.channel_axis = -1

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim,self.num_filters)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',constraint =self.kernel_constraint,regularizer=self.kernel_regularization)
        super(Dilation3Dyz, self).build(input_shape)


    def call(self,x):
        #print('x.shape',x.shape)
        #print('self.kernel.shape',self.kernel.shape)
        tile=tf.reshape(x,[tf.shape(x)[0],tf.shape(x)[1]*tf.shape(x)[2],tf.shape(x)[3],tf.shape(x)[4]])
        #print('tile.shape',tile.shape)
        res=[]
        for i in range(self.num_filters):
            res.append(dilation2d(tile, self.kernel[..., i],self.strides, self.padding))
        res=tf.stack(res,axis=-1)
        #print('res.shape',res.shape)
        res=tf.reshape(res,[tf.shape(res)[0],tf.shape(x)[1],tf.shape(x)[2],tf.shape(res)[2],tf.shape(res)[3]*tf.shape(res)[4]])
        return res