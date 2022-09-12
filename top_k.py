import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers


class TopKConv(layers.DepthwiseConv2D):
    def __init__(self, kernel_size, filter_shape, k=10, **kwargs):
        super().__init__(kernel_size, **kwargs)
        self.k = k
        self.filter_shape = filter_shape
        
    def call(self, inputs):
        outputs = keras.backend.depthwise_conv2d(
            inputs,
            self.depthwise_kernel,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format)
    
        outputs = tf.reshape(outputs, (-1, self.filter_shape, self.filter_shape, 
                                       tf.shape(inputs)[-1], self.depth_multiplier))
        
        outputs = self.max_sum(outputs)
        
        outputs = keras.backend.bias_add(
            outputs,
            self.bias,
            data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        
    def build(self, input_shape):
        
        self.use_bias = False        
        super(TopKConv, self).build(input_shape)
        
        self.bias = self.add_weight(shape=(self.depth_multiplier,),
                            initializer=self.bias_initializer,
                            name='bias',
                            regularizer=self.bias_regularizer,
                            constraint=self.bias_constraint)
    
    def max_sum(self, inpt):
        # get sum of each channel
        channel_sums = tf.reduce_sum(inpt, axis=(1, 2))
        # channel_sums = tf.reduce_max(inpt, axis=(1, 2))

        # get indices of top-k channels by sum
        max_idx = tf.math.top_k(tf.transpose(channel_sums, [0, 2, 1]), self.k).indices

        # extract top-k channels
        top_channels = tf.gather(tf.transpose(inpt, [0, 4, 3, 1, 2]), max_idx, axis=2, batch_dims=2)

        # reduce-sum top-k input channels for each output channel
        return tf.reduce_sum(tf.transpose(top_channels, [0, 3, 4, 2, 1]), axis=3)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.depth_multiplier)
    
    def get_config(self):
        config = super().get_config()
        config.update({"k": self.k})
        return config

