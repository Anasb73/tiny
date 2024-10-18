#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
************************************************************************************
**
** Copyright (C) 2021 ASYGN SAS. All rights reserved.
**
** This file is part of the demonstration applications of the IPCEI Gemini Project.
**
** This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
** WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
**
************************************************************************************
Created on Mon Feb  7 11:28:41 2022

@author: jpotot
"""

##### https://github.com/tensorflow/model-optimization/blob/c8cce59d4c6354d7668be7b2c508054217ec60a5/tensorflow_model_optimization/python/core/quantization/keras/default_8bit/default_8bit_quantize_registry.py

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.quantization.keras.quantizers import (
    LastValueQuantizer,
    MovingAverageQuantizer,
)


class WeightsQuantizer(LastValueQuantizer):
    """Quantizer for handling weights in all layers."""

    def __init__(self, num_bits=8, per_axis=False, symmetric=True, narrow_range=False):
        """Construct LastValueQuantizer with params specific for TFLite Convs."""
        super(WeightsQuantizer, self).__init__(
            num_bits=num_bits, per_axis=per_axis, symmetric=symmetric, narrow_range=narrow_range
        )

    def build(self, tensor_shape, name, layer):
        if self.per_axis:
            min_weight = layer.add_weight(
                name + "_min",
                shape=(tensor_shape[-1],),
                initializer=tf.keras.initializers.Constant(-6.0),
                trainable=False,
            )
            max_weight = layer.add_weight(
                name + "_max",
                shape=(tensor_shape[-1],),
                initializer=tf.keras.initializers.Constant(6.0),
                trainable=False,
            )

            return {"min_var": min_weight, "max_var": max_weight}
        else:
            return super(WeightsQuantizer, self).build(tensor_shape, name, layer)


class AthenaConvQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    def __init__(
        self,
    ):
        self.weight_attrs = ["kernel"]
        self.activation_attrs = ["activation"]
        self.quantize_output = False

        self.weight_quantizer = WeightsQuantizer(
            num_bits=8, per_axis=True, symmetric=True, narrow_range=True
        )
        self.activation_quantizer = MovingAverageQuantizer(
            num_bits=8, symmetric=False, narrow_range=False, per_axis=False
        )

    def get_weights_and_quantizers(self, layer):
        output = []
        for weight_attr in self.weight_attrs:
            if weight_attr == "kernel" and hasattr(layer, "depthwise_kernel"):
                output.append((layer.depthwise_kernel, self.weight_quantizer))
            else:
                output.append((getattr(layer, weight_attr), self.weight_quantizer))
        return output

    def get_activations_and_quantizers(self, layer):
        return [
            (getattr(layer, activation_attr), self.activation_quantizer)
            for activation_attr in self.activation_attrs
        ]

    def set_quantize_weights(self, layer, quantize_weights):
        if len(self.weight_attrs) != len(quantize_weights):
            raise ValueError(
                "`set_quantize_weights` called on layer {} with {} "
                "weight parameters, but layer expects {} values.".format(
                    layer.name, len(quantize_weights), len(self.weight_attrs)
                )
            )

        for weight_attr, weight in zip(self.weight_attrs, quantize_weights):
            if hasattr(layer, "depthwise_kernel") and weight_attr == "kernel":
                layer.kernel = weight
            else:
                current_weight = getattr(layer, weight_attr)
                if current_weight.shape != weight.shape:
                    raise ValueError(
                        "Existing layer weight shape {} is incompatible with"
                        "provided weight shape {}".format(current_weight.shape, weight.shape)
                    )
                setattr(layer, weight_attr, weight)

    def set_quantize_activations(self, layer, quantize_activations):
        if len(self.activation_attrs) != len(quantize_activations):
            raise ValueError(
                "`set_quantize_activations` called on layer {} with {} "
                "activation parameters, but layer expects {} values.".format(
                    layer.name, len(quantize_activations), len(self.activation_attrs)
                )
            )

        for activation_attr, activation in zip(self.activation_attrs, quantize_activations):
            setattr(layer, activation_attr, activation)

    # Configure how to quantize outputs (may be equivalent to activations).
    def get_output_quantizers(self, layer):
        if self.quantize_output:
            return [self.activation_quantizer]
        return []

    def get_config(self):
        return {}


class AthenaDenseQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    def __init__(
        self,
    ):
        self.weight_attrs = ["kernel"]
        self.activation_attrs = ["activation"]
        self.quantize_output = False

        self.weight_quantizer = LastValueQuantizer(
            num_bits=8, symmetric=True, narrow_range=True, per_axis=False
        )
        self.activation_quantizer = MovingAverageQuantizer(
            num_bits=8, symmetric=False, narrow_range=False, per_axis=False
        )

    def get_weights_and_quantizers(self, layer):
        return [
            (getattr(layer, weight_attr), self.weight_quantizer)
            for weight_attr in self.weight_attrs
        ]

    def get_activations_and_quantizers(self, layer):
        return [
            (getattr(layer, activation_attr), self.activation_quantizer)
            for activation_attr in self.activation_attrs
        ]

    def set_quantize_weights(self, layer, quantize_weights):
        if len(self.weight_attrs) != len(quantize_weights):
            raise ValueError(
                "`set_quantize_weights` called on layer {} with {} "
                "weight parameters, but layer expects {} values.".format(
                    layer.name, len(quantize_weights), len(self.weight_attrs)
                )
            )

        for weight_attr, weight in zip(self.weight_attrs, quantize_weights):
            current_weight = getattr(layer, weight_attr)
            if current_weight.shape != weight.shape:
                raise ValueError(
                    "Existing layer weight shape {} is incompatible with"
                    "provided weight shape {}".format(current_weight.shape, weight.shape)
                )

            setattr(layer, weight_attr, weight)

    def set_quantize_activations(self, layer, quantize_activations):
        if len(self.activation_attrs) != len(quantize_activations):
            raise ValueError(
                "`set_quantize_activations` called on layer {} with {} "
                "activation parameters, but layer expects {} values.".format(
                    layer.name, len(quantize_activations), len(self.activation_attrs)
                )
            )

        for activation_attr, activation in zip(self.activation_attrs, quantize_activations):
            setattr(layer, activation_attr, activation)

    # Configure how to quantize outputs (may be equivalent to activations).
    def get_output_quantizers(self, layer):
        if self.quantize_output:
            return [self.activation_quantizer]
        return []

    def get_config(self):
        return {}


class AthenaActivationQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    def _assert_activation_layer(self, layer):
        if not isinstance(layer, tf.keras.layers.Activation):
            raise RuntimeError(
                "Default8BitActivationQuantizeConfig can only be used with "
                "`keras.layers.Activation`."
            )

    def get_weights_and_quantizers(self, layer):
        self._assert_activation_layer(layer)
        return []

    def get_activations_and_quantizers(self, layer):
        self._assert_activation_layer(layer)
        return []

    def set_quantize_weights(self, layer, quantize_weights):
        self._assert_activation_layer(layer)

    def set_quantize_activations(self, layer, quantize_activations):
        self._assert_activation_layer(layer)

    def get_output_quantizers(self, layer):
        self._assert_activation_layer(layer)
        if not hasattr(layer.activation, "__name__"):
            raise ValueError(
                "Activation {} not supported by AthenaActivationQuantizeConfig.".format(
                    layer.activation
                )
            )

        if layer.activation.__name__ in ["relu", "swish", "gelu", "linear"]:
            # 'relu' should generally get fused into the previous layer.
            return [
                MovingAverageQuantizer(
                    num_bits=8, per_axis=False, symmetric=False, narrow_range=False
                )
            ]
        elif layer.activation.__name__ in ["linear", "softmax", "sigmoid", "tanh"]:
            return []

        raise ValueError(
            "Activation {} not supported by AthenaActivationQuantizeConfig.".format(
                layer.activation
            )
        )
        return [
            MovingAverageQuantizer(num_bits=8, symmetric=False, narrow_range=False, per_axis=False)
        ]

    def get_config(self):
        return {}


class AthenaNoOpQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    """QuantizeConfig which does not quantize any part of the layer."""

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


class AthenaOutputQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    def get_weights_and_quantizers(self, layer):
        return []

    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_weights(self, layer, quantize_weights):
        pass

    def set_quantize_activations(self, layer, quantize_activations):
        pass

    def get_output_quantizers(self, layer):
        return [
            MovingAverageQuantizer(num_bits=8, per_axis=False, symmetric=False, narrow_range=False)
        ]

    def get_config(self):
        return {}


##### CUSTOM ATHENA QUANTIZATION
def apply_athena_quantization(layer):
    if isinstance(layer, tf.keras.layers.Dense):
        print(f"Add quant annotation on layer {layer.name}")
        return tfmot.quantization.keras.quantize_annotate_layer(
            layer, quantize_config=AthenaDenseQuantizeConfig()
        )
    elif isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.DepthwiseConv2D):
        print(f"Add quant annotation on layer {layer.name}")
        return tfmot.quantization.keras.quantize_annotate_layer(
            layer, quantize_config=AthenaConvQuantizeConfig()
        )
    elif isinstance(layer, tf.keras.layers.Activation):
        if layer.activation.__name__ in ["relu", "swish", "gelu"]:
            print(f"Add quant annotation on layer {layer.name}")
            return tfmot.quantization.keras.quantize_annotate_layer(
                layer, quantize_config=AthenaActivationQuantizeConfig()
            )
    elif isinstance(layer, tf.keras.layers.ReLU):
        print(f"Add quant annotation on layer {layer.name}")
        return tfmot.quantization.keras.quantize_annotate_layer(
            layer, quantize_config=AthenaActivationQuantizeConfig()
        )
    print(f"{layer.name} will not be quantized")
    return layer


##### DEFAULT ATHENA QUANTIZATION
def apply_default_athena_quantization(layer):
    if isinstance(layer, tf.keras.layers.Dense):
        print(f"Add quant annotation on layer {layer.name}")
        return tfmot.quantization.keras.quantize_annotate_layer(
            layer,
        )
    elif isinstance(layer, tf.keras.layers.Conv2D):
        print(f"Add quant annotation on layer {layer.name}")
        return tfmot.quantization.keras.quantize_annotate_layer(
            layer,
        )
    elif isinstance(layer, tf.keras.layers.Activation):
        if layer.activation.__name__ in ["relu", "swish", "gelu"]:
            print(f"Add quant annotation on layer {layer.name}")
            return tfmot.quantization.keras.quantize_annotate_layer(
                layer,
            )
    elif isinstance(layer, tf.keras.layers.ReLU):
        print(f"Add quant annotation on layer {layer.name}")
        return tfmot.quantization.keras.quantize_annotate_layer(
            layer,
        )
    print(f"{layer.name} will not be quantized")
    return layer


def quantize_athena_model(model):
    annotate_model = tf.keras.models.clone_model(model, clone_function=apply_athena_quantization)
    with tfmot.quantization.keras.quantize_scope(
        {
            "AthenaConvQuantizeConfig": AthenaConvQuantizeConfig,
            "AthenaDenseQuantizeConfig": AthenaDenseQuantizeConfig,
            "AthenaActivationQuantizeConfig": AthenaActivationQuantizeConfig,
            "AthenaNoOpQuantizeConfig": AthenaNoOpQuantizeConfig,
            "AthenaOutputQuantizeConfig": AthenaOutputQuantizeConfig,
        }
    ):
        quant_aware_model = tfmot.quantization.keras.quantize_apply(annotate_model)
    return quant_aware_model


def quantize_default_athena(model):
    annotate_model = tf.keras.models.clone_model(
        model, input_tensors=None, clone_function=apply_default_athena_quantization
    )
    with tfmot.quantization.keras.quantize_scope():
        quant_aware_model = tfmot.quantization.keras.quantize_apply(annotate_model)
    return quant_aware_model


def apply_quantization_aware_training(model, type_config="athena"):
    print(f"Apply {type_config} quantization aware training model")
    if type_config == "default":
        quant_aware_model = tfmot.quantization.keras.quantize_model(model)
    elif type_config == "athena":
        quant_aware_model = quantize_athena_model(model)
    elif type_config == "default_athena":
        quant_aware_model = quantize_default_athena(model)
    else:
        raise ValueError(f"Unsupported type config '{type_config}'")
    return quant_aware_model


def load_quantize_model(path, params_load={}, type_config="default", **kwargs):
    print(f"Loading quantize model from {path} with config {type_config}")
    if type_config == "default":
        with tfmot.quantization.keras.quantize_scope():
            quant_aware_model = tf.keras.models.load_model(path, **params_load)
    elif type_config == "athena":
        with tfmot.quantization.keras.quantize_scope(
            {
                "AthenaConvQuantizeConfig": AthenaConvQuantizeConfig,
                "AthenaDenseQuantizeConfig": AthenaDenseQuantizeConfig,
                "AthenaActivationQuantizeConfig": AthenaActivationQuantizeConfig,
                "AthenaNoOpQuantizeConfig": AthenaNoOpQuantizeConfig,
                "AthenaOutputQuantizeConfig": AthenaOutputQuantizeConfig,
            }
        ):
            quant_aware_model = tf.keras.models.load_model(path, **params_load)
    elif type_config == "default_athena":
        with tfmot.quantization.keras.quantize_scope():
            quant_aware_model = tf.keras.models.load_model(path, **params_load)
    else:
        raise ValueError(f"Unsupported type config '{type_config}'")
    return quant_aware_model
