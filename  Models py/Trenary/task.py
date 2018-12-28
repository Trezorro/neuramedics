# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This code implements a Feed forward neural network using Keras API.
Ternary model.
"""

from comet_ml import Experiment
import argparse
import glob
import json
import os
import numpy as np

import keras
from keras.models import load_model
import model
from tensorflow.python.lib.io import file_io
from keras.preprocessing.image import ImageDataGenerator


# Add the following code anywhere in your machine learning file experiment =
Experiment(api_key="Qs9hrAPUWIusY5UKNebGp1MAN", project_name="data-challange-1", workspace="blazejmanczak")

CLASS_SIZE = 3

# CHUNK_SIZE specifies the number of lines
# to read in case the file is very large
FILE_PATH = 'checkpoint.{epoch:02d}.hdf5'
RETINOPATHY_MODEL = 'retinopathy.hdf5'


class ContinuousEval(keras.callbacks.Callback):
    """Continuous eval callback to evaluate the checkpoint once
       every so many epochs.
    """

    def __init__(self,
                 eval_frequency,
                 job_dir):
        self.eval_frequency = eval_frequency
        self.job_dir = job_dir
        [self.X_test, self.Y_test] = model.read_test_data()
        self.loss = 0
        self.acc = 0

    def on_epoch_begin(self, epoch, logs={}):
        if epoch > 0 and epoch % self.eval_frequency == 0:
            # Unhappy hack to work around h5py not being able to write to GCS.
            # Force snapshots and saves to local filesystem, then copy them over to GCS.
            model_path_glob = 'checkpoint.*'
            model_path_glob = os.path.join(self.job_dir, model_path_glob)
            checkpoints = glob.glob(model_path_glob)
            if len(checkpoints) > 0:
                checkpoints.sort()
                retinopathy_model = load_model(checkpoints[-1])
                retinopathy_model = model.compile_model(retinopathy_model)
                loss, acc = retinopathy_model.evaluate(
                    self.X_test, self.Y_test)
                self.loss, self.acc = retinopathy_model.evaluate(
                    self.X_test, self.Y_test)
                print('\nEvaluation epoch[{}] metrics[{:.2f}, {:.2f}] {}'.format(
                    epoch, loss, acc, retinopathy_model.metrics_names))
            else:
                print('\nEvaluation epoch[{}] (no checkpoints found)'.format(epoch))

def save_model(model, acc, name):

    if acc >= .01:
        #print('Saving model')
        model.save('/mnt/server-home/dc_group08/models/' + name + '_accuracy_' + str(acc) + '.hdf5')
    else:
        #print('The model was not saved, the accuracy is:' + str(acc))
        pass

def dispatch(train_files,
             eval_files,
             job_dir,
             train_steps,
             eval_steps,
             train_batch_size,
             eval_batch_size,
             learning_rate,
             eval_frequency,
             first_layer_size,
             num_layers,
             scale_factor,
             eval_num_epochs,
             num_epochs,
             checkpoint_epochs):
    retinopathy_model = model.model_fn(CLASS_SIZE)

    try:
        os.makedirs(job_dir)
    except:
        pass

    # Unhappy hack to work around h5py not being able to write to GCS.
    # Force snapshots and saves to local filesystem, then copy them over to GCS.
    checkpoint_path = FILE_PATH
    checkpoint_path = os.path.join(job_dir, checkpoint_path)

    # Model checkpoint callback
    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        verbose=2,
        period=checkpoint_epochs,
        mode='max')

    # Continuous eval callback
    evaluation = ContinuousEval(eval_frequency,
                                job_dir)

    # Tensorboard logs callback
    tblog = keras.callbacks.TensorBoard(
        log_dir=os.path.join(job_dir, 'logs'),
        histogram_freq=0,
        write_graph=True,
        embeddings_freq=0)

    callbacks = [checkpoint, evaluation, tblog]
    #callbacks = [evaluation, tblog]

    [X_train, Y_train] = model.read_train_data()

    datagen = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)

    retinopathy_model.fit_generator(
        datagen.flow(X_train, Y_train, batch_size=100),
        steps_per_epoch=200,
        epochs=50,
        callbacks=callbacks,
        verbose=2,
        validation_data=(evaluation.X_test, evaluation.Y_test))

    retinopathy_model.save(os.path.join(job_dir, RETINOPATHY_MODEL)) # previosuly commented out

    save_model(retinopathy_model, evaluation.acc, 'Jaap' )

if __name__ == "__main__":



    parser = argparse.ArgumentParser()
    parser.add_argument('--train-files',
                        required=False,
                        type=str,
                        help='Training files local or GCS', nargs='+')
    parser.add_argument('--eval-files',
                        required=False,
                        type=str,
                        help='Evaluation files local or GCS', nargs='+')
    parser.add_argument('--job-dir',
                        required=True,
                        type=str,
                        help='GCS or local dir to write checkpoints and export model')
    parser.add_argument('--train-steps',
                        type=int,
                        default=100,
                        help="""\
                       Maximum number of training steps to perform
                       Training steps are in the units of training-batch-size.
                       So if train-steps is 500 and train-batch-size if 100 then
                       at most 500 * 100 training instances will be used to train.
                      """)
    parser.add_argument('--eval-steps',
                        help='Number of steps to run evalution for at each checkpoint',
                        default=100,
                        type=int)
    parser.add_argument('--train-batch-size',
                        type=int,
                        default=40,
                        help='Batch size for training steps')
    parser.add_argument('--eval-batch-size',
                        type=int,
                        default=40,
                        help='Batch size for evaluation steps')
    parser.add_argument('--learning-rate',
                        type=float,
                        default=0.003,
                        help='Learning rate for SGD')
    parser.add_argument('--eval-frequency',
                        default=10,
                        help='Perform one evaluation per n epochs')
    parser.add_argument('--first-layer-size',
                        type=int,
                        default=256,
                        help='Number of nodes in the first layer of DNN')
    parser.add_argument('--num-layers',
                        type=int,
                        default=2,
                        help='Number of layers in DNN')
    parser.add_argument('--scale-factor',
                        type=float,
                        default=0.25,
                        help="""\
                      Rate of decay size of layer for Deep Neural Net.
                      max(2, int(first_layer_size * scale_factor**i)) \
                      """)
    parser.add_argument('--eval-num-epochs',
                        type=int,
                        default=1,
                        help='Number of epochs during evaluation')
    parser.add_argument('--num-epochs',
                        type=int,
                        default=20,
                        help='Maximum number of epochs on which to train')
    parser.add_argument('--checkpoint-epochs',
                        type=int,
                        default=10,
                        help='Checkpoint per n training epochs')
    parse_args, unknown = parser.parse_known_args()


    dispatch(**parse_args.__dict__)
