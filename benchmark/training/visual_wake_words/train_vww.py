# Lint as: python3
"""Training for the visual wakewords person detection model.

The visual wakewords person detection model is a core model for the TinyMLPerf
benchmark suite. This script provides source for how the reference model was
created and trained, and can be used as a starting point for open submissions
using re-training.
"""

import os, sys, json

from absl import app
from vww_model import mobilenet_v1

import tensorflow as tf
assert tf.__version__.startswith('2')
from tensorflow.keras.callbacks import ModelCheckpoint

import nn_train_utils
from nn_train_utils import DataGenWrapper
import custom_quantization


IMAGE_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 2


with open('config.json', 'r') as f:
  config = json.load(f)

def train_epochs(model, train_generator, val_generator, epoch_count,
                 learning_rate, callbacks):
  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
      loss='categorical_crossentropy',
      metrics=['accuracy'])
  history_fine = model.fit(
      train_generator,
      steps_per_epoch=len(train_generator),
      epochs=epoch_count,
      validation_data=val_generator,
      validation_steps=len(val_generator),
      batch_size=BATCH_SIZE,
      callbacks=callbacks,
      verbose=1)
  return model


def main(argv):
  if len(argv) >= 2:
    model = tf.keras.models.load_model(argv[1])
  else:
    model = mobilenet_v1()

  model.summary()


  train_x_paths, train_y, valid_x_paths, valid_y =  \
    nn_train_utils.read_train_dataset(config['NB_OUTPUT'],
                                      config['DATASET_PATH'],
                                      config['VALIDATION_SPLIT'],
                                      config['MAX_DSET_INPUTS'])

  train_generator = DataGenWrapper(train_x_paths,
                                   train_y,
                                   x_type=config['INPUT_TYPE'],
                                   input_shape=config['INPUT_SHAPE'],
                                   input_scaling=config['INPUT_SCALING'],
                                   batch_size=config['BATCH_SIZE'],
                                   standardize=config['STD_INPUTS'],
                                   augmentation=config['AUGMENTATION'],
                                   shuffle=True)

  valid_generator = DataGenWrapper(valid_x_paths,
                                   valid_y,
                                   x_type=config['INPUT_TYPE'],
                                   input_shape=config['INPUT_SHAPE'],
                                   input_scaling=config['INPUT_SCALING'],
                                   batch_size=config['BATCH_SIZE'],
                                   standardize=config['STD_INPUTS'],
                                   augmentation=config['AUGMENTATION'],
                                   shuffle=False)

  

  ## Train with checkpoint (keep best model)
  model_name = "/work1/gitlab-runner-docker-data/models/vww/"  + config['OUTPUT_MODEL_NAME']
  checkpoint = ModelCheckpoint(model_name, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
  callbacks_list = [checkpoint]
          
  model = train_epochs(model, train_generator, valid_generator, 20, 0.001, callbacks_list)
  model = train_epochs(model, train_generator, valid_generator, 10, 0.0005, callbacks_list)
  model = train_epochs(model, train_generator, valid_generator, 20, 0.00025, callbacks_list)

  validation_split = 0.1



  ## Quantize and QAT training
  qat_model = custom_quantization.quantize(model)
  qat_model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
      loss='categorical_crossentropy',
      metrics=['accuracy'])
  

  print("Running Quantization-aware training")
  print("***********************************")

  model_name = "/work1/gitlab-runner-docker-data/models/vww/"  + config['QAT_MODEL_NAME']
  checkpoint = ModelCheckpoint(model_name, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
  callbacks_list = [checkpoint]

  history = qat_model.fit(x=train_generator,
                          validation_data=valid_generator,
                          epochs=config['N_RETRAIN_EPOCHS'],
                          verbose=1,
                          callbacks=callbacks_list)

  qat_model = train_epochs(qat_model, train_generator, valid_generator, 20, 0.001, callbacks_list)
  qat_model = train_epochs(qat_model, train_generator, valid_generator, 10, 0.0005, callbacks_list)
  qat_model= train_epochs(qat_model, train_generator, valid_generator, 20, 0.00025, callbacks_list)

  qat_model.summary()


if __name__ == '__main__':
  app.run(main)
