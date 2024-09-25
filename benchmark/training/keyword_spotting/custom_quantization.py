import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
from  tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import activations

LastValueQuantizer = tfmot.quantization.keras.quantizers.LastValueQuantizer
MovingAverageQuantizer = tfmot.quantization.keras.quantizers.MovingAverageQuantizer
quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
quantize_annotate_model = tfmot.quantization.keras.quantize_annotate_model
quantize_scope = tfmot.quantization.keras.quantize_scope
quantizers = tfmot.quantization.keras.quantizers


class WeightsQuantizer(quantizers.LastValueQuantizer):
  def __init__(self, num_bits, per_axis, symmetric, narrow_range):
    super(WeightsQuantizer, self).__init__(num_bits=num_bits, per_axis=per_axis, 
                                           symmetric=symmetric, narrow_range=narrow_range)

  ## build() function needs a specific description for per-axis quantization
  def build(self, tensor_shape, name, layer):
    if self.per_axis:
      min_weight = layer.add_weight(
        name + '_min',
        shape=(tensor_shape[-1],),
        initializer=tf.keras.initializers.Constant(-6.0),
        trainable=False)
      
      max_weight = layer.add_weight(
        name + '_max',
        shape=(tensor_shape[-1],),
        initializer=tf.keras.initializers.Constant(6.0),
        trainable=False)
          
      return {'min_var': min_weight, 'max_var': max_weight}
    else:
      return super(WeightsQuantizer, self).build(tensor_shape, name, layer)


class DenseQuantizeConfig(tfmot.quantization.keras.QuantizeConfig, ):
  global ACTIV_NBITS
  global WEIGHT_NBITS
  global WEIGHT_SYMMETRIC
  global WEIGHT_PER_AXIS

  def get_weights_and_quantizers(self, layer):
    ## Configuration for Dense layers weights is done here
    return [(layer.kernel, WeightsQuantizer(num_bits=WEIGHT_NBITS,
                                            symmetric=WEIGHT_SYMMETRIC,
                                            narrow_range=False,
                                            per_axis=WEIGHT_PER_AXIS))]

  def get_activations_and_quantizers(self, layer):
    ## Configuration for Dense layers activations is done here
    return [(layer.activation, MovingAverageQuantizer(num_bits=ACTIV_NBITS,
                                                      symmetric=False,
                                                      narrow_range=False,
                                                      per_axis=False))]

  def set_quantize_weights(self, layer, quantize_weights):
    layer.kernel = quantize_weights[0]

  def set_quantize_activations(self, layer, quantize_activations):
    layer.activation = quantize_activations[0]

  def get_output_quantizers(self, layer):
    return []

  def get_config(self):
    return {}


class ConvQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
  global ACTIV_NBITS
  global WEIGHT_NBITS
  global WEIGHT_SYMMETRIC
  global WEIGHT_PER_AXIS

  def get_weights_and_quantizers(self, layer):
    ## Configuration for Conv layers weights is done here
    if hasattr(layer, 'kernel'):
      kernel = layer.kernel
    else:
      kernel = layer.depthwise_kernel

    return [(kernel, WeightsQuantizer(num_bits=WEIGHT_NBITS,
                                      symmetric=WEIGHT_SYMMETRIC,
                                      narrow_range=False,
                                      per_axis=WEIGHT_PER_AXIS))]

  def get_activations_and_quantizers(self, layer):
    ## Configuration for Conv layers activations is done here
    return [(layer.activation, MovingAverageQuantizer(num_bits=ACTIV_NBITS,
                                                      symmetric=False,
                                                      narrow_range=False,
                                                      per_axis=False))]

  def set_quantize_weights(self, layer, quantize_weights):
    layer.kernel = quantize_weights[0]

  def set_quantize_activations(self, layer, quantize_activations):
    layer.activation = quantize_activations[0]

  def get_output_quantizers(self, layer):
    return []

  def get_config(self):
    return {}


class OutputQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
  global ACTIV_NBITS
  global WEIGHT_NBITS
  global WEIGHT_SYMMETRIC
  global WEIGHT_PER_AXIS
  
  def get_weights_and_quantizers(self, layer):
    return []
  
  def get_activations_and_quantizers(self, layer):
    return []

  def set_quantize_weights(self, layer, quantize_weights):
    pass

  def set_quantize_activations(self, layer, quantize_activations):
    pass

  def get_output_quantizers(self, layer):
    # Configuration for layers outputs quantization is done here
    return [MovingAverageQuantizer(num_bits=ACTIV_NBITS, symmetric=False, narrow_range=False, per_axis=False)]

  def get_config(self):
    return {}


class NoOpQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
  def get_weights_and_quantizers(self, layer):
    return []

  def get_activations_and_quantizers(self, layer):
    return []

  def set_quantize_weights(self, layer, quantize_weights):
    pass

  def set_quantize_activations(self, layer, quantize_activations):
    pass

  def get_output_quantizers(self, layer):
    return []

  def get_config(self):
    return {}


def layer_quantization(layer):
  ## Not annotated layers will not be quantized for training
  if isinstance(layer, tf.keras.layers.Dense):
    return tfmot.quantization.keras.quantize_annotate_layer(layer, DenseQuantizeConfig())
  elif isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.DepthwiseConv2D):
    return tfmot.quantization.keras.quantize_annotate_layer(layer, ConvQuantizeConfig())
  #elif isinstance(layer, tf.keras.layers.BatchNormalization):
  #  return tfmot.quantization.keras.quantize_annotate_layer(layer, NoOpQuantizeConfig())
  elif isinstance(layer, tf.keras.layers.Activation):
    return tfmot.quantization.keras.quantize_annotate_layer(layer, OutputQuantizeConfig())
  #elif isinstance(layer, tf.keras.layers.Flatten):
   # return tfmot.quantization.keras.quantize_annotate_layer(layer, OutputQuantizeConfig())
  #else:
  #  return tfmot.quantization.keras.quantize_annotate_layer(layer, NoOpQuantizeConfig())
  return layer


def check_transformable(model):
  idx = 0
  transformable = True
  
  for l in model.layers:
    if "Lambda" in l.__class__.__name__:
      transformable = False
      
  return transformable


def check_model(model):
  idx = 0
  while idx < len(model.layers):
    ## Check structure with batchNorm
    if (model.layers[idx].__class__.__name__ == "BatchNormalization" and
        model.layers[idx+1].__class__.__name__ != "Activation"):
      print("ERROR: A BatchNormalization layer shall be followed by an Activation layer (layer %d)" % idx)
      print("       Please add a Linear or RelU activation after this BatchNormalization.")
      exit()
      
    idx += 1


def transform_model(model):
  x = model.input
  addedNbr = 1
  
  idx = 0
  while idx < len(model.layers):
    if model.layers[idx].__class__.__name__ == "InputLayer":
      idx += 1
      continue
    
    x = model.layers[idx](x)
    
    ## Add a linear activation (= no activation) after batch normalization layers if no activation present.
    ## This is needed for quantization statistics after BN
    if (model.layers[idx].__class__.__name__ == "BatchNormalization" and
        model.layers[idx+1].__class__.__name__ != "Activation"):
      x = Activation('linear', name="linear_activation_%d" % addedNbr)(x)
      addedNbr += 1
      
    idx += 1
    
  transformedModel = Model(inputs=model.inputs, outputs=x)

  ## Replace activation in last layer as a separate layer (needed for quantization)
  """
  if (transformedModel.layers[-1].__class__.__name__ != 'Activation' and
      'activation' in transformedModel.layers[-1].get_config()):
    activationName = transformedModel.layers[-1].get_config()['activation']
    if activationName != 'linear':
      transformedModel.layers[-1].activation = activations.linear
      lastActivationLayer = Activation(activationName, name="last_activation")(transformedModel.output)
      transformedModel = Model(inputs=model.inputs, outputs=lastActivationLayer)
  """
  return transformedModel


  
# Correct structure for batch normalization: CONV-BN-ACTIVATION or DENSE-BN-ACTIVATION
def quantize(model, activ_nbits=8, weight_nbits=8, weight_symmetric=True, weight_per_axis=False):
  global ACTIV_NBITS
  global WEIGHT_NBITS
  global WEIGHT_SYMMETRIC
  global WEIGHT_PER_AXIS

  ACTIV_NBITS      = activ_nbits
  WEIGHT_NBITS     = weight_nbits
  WEIGHT_SYMMETRIC = weight_symmetric
  WEIGHT_PER_AXIS  = weight_per_axis

  print("Quantization parameters")
  print("-----------------------")
  print("Activations nbits:", ACTIV_NBITS)
  print("Weights nbits:", WEIGHT_NBITS)
  print("Symmetrical weights:", WEIGHT_SYMMETRIC)
  print("Per-axis weights:", WEIGHT_PER_AXIS)
  print("-----------------------")


  if check_transformable(model):
    print("Model will be transformed if needed.")
    transformedModel = tf.keras.models.clone_model(model)
    transformedModel.set_weights(model.get_weights())
    transformedModel = transform_model(transformedModel)
  else:
    transformedModel = model
    check_model(transformedModel)
    
  annotated_model = tf.keras.models.clone_model(
            transformedModel,
            clone_function=layer_quantization
            )
  

  with quantize_scope(
    {'DenseQuantizeConfig': DenseQuantizeConfig,
     'ConvQuantizeConfig':ConvQuantizeConfig,
     'OutputQuantizeConfig':OutputQuantizeConfig,
     'NoOpQuantizeConfig':NoOpQuantizeConfig}):
    quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)
    return quant_aware_model