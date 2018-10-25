"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@contribution : Fares Abawi (6abawi@informatik.uni-hamburg.de)

"""

import argparse
import os
import random
import string
from pprint import pprint

import numpy as np
import keras
import keras.backend as K
import keras.preprocessing.image
from keras.utils.vis_utils import plot_model

import keras_retinanet.losses
import keras_retinanet.models as models
from keras_retinanet.models.resnet import ResNet50RetinaNet, ResNet101RetinaNet, ResNet152RetinaNet
from keras_fusionnet.preprocessing.fusion_generator import NicoGraspingGeneratorV0_0,\
    NicoGraspingGeneratorV0_1, NicoGraspingGeneratorV1_0, NicoGraspingGeneratorV1_1
from keras_retinanet.utils.keras_version import check_keras_version
from keras_fusionnet.models.fusionnet import FusionNet

import tensorflow as tf


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.val_losses = []
        self.train_losses = []
        self.epochs = 0
        self.best_epoch = 0
        self.best_val_loss = float("inf")
        self.best_train_loss = float("inf")

    def on_epoch_end(self, batch, logs={}):
        self.train_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.epochs += 1
        if self.val_losses[-1] < self.best_val_loss:
            self.best_val_loss = self.val_losses[-1]
            self.best_train_loss = self.train_losses[-1]
            self.best_epoch = self.epochs

def store_configuration(args, params, model_summary):
    file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        '..', 'logs', str(args.identifier) + '_details.config')
    with open(file, 'wt') as out:
        pprint(vars(args), stream=out)
        pprint(params, stream=out)
        # pprint(model_summary, stream=out)

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_model(weights='imagenet', model_type=None, resnet_type=None,
                 retinanet_model_path=None, retinanet_freeze_weights=False,
                 fusionnet_layers=None):
    fusionnet = FusionNet()
    retinanet_image_input = keras.layers.Input((60, 80, 3), name='retinanet_image_input')
    fusionnet_image_input = keras.layers.Input((60, 80, 3), name='fusionnet_image_input')
    fusionnet_category_input = keras.layers.Input((20,), name='fusionnet_object_description_input')


    if retinanet_model_path is not None:
        retinanet = models.load_model(retinanet_model_path)
        retinanet.trainable = retinanet_freeze_weights
    else:
        # TODO (fabawi) : num_class should be dynamic since adding new label types would break the network
        if resnet_type == 'resnet_50_retinanet':
            resnet = ResNet50RetinaNet
        elif resnet_type == 'resnet_101_retinanet':
            resnet = ResNet101RetinaNet
        elif resnet_type == 'resnet_152_retinanet':
            resnet = ResNet152RetinaNet
        else:
            resnet = ResNet50RetinaNet

        retinanet = resnet(retinanet_image_input, num_classes=19, weights=weights)

    if model_type == 'fusionnetV0_0':
        return fusionnet.fusionnet_v0_0(fusionnet_image_input, retinanet, fusionnet_layers)
    elif model_type == 'fusionnetV0_1':
        return fusionnet.fusionnet_v0_1(fusionnet_image_input, fusionnet_layers)
    elif model_type == 'fusionnetV1_0':
        return fusionnet.fusionnet_v1_0(fusionnet_image_input, fusionnet_category_input, retinanet, fusionnet_layers)
    elif model_type == 'fusionnetV1_1':
        return fusionnet.fusionnet_v1_1(fusionnet_image_input, fusionnet_category_input, fusionnet_layers)
    else:
        return fusionnet.fusionnet_v0_0(fusionnet_image_input, retinanet, fusionnet_layers)


def train(args, params=None):
    if params is not None:
        opt_lr_val = params['opt_lr_val']
        opt_reducelr_plateau = params['opt_reducelr_plateau']
        opt_resnet_type_val = params['opt_resnet_type_val']
        opt_fusionnet_layers_val = params['opt_fusionnet_layers_val']
    else:
        params = dict()
        opt_lr_val = params['opt_lr_val'] = 0.01
        opt_reducelr_plateau  = params['opt_reducelr_plateau'] = False
        opt_resnet_type_val = params['opt_resnet_type_val'] = ''
        opt_fusionnet_layers_val = params['opt_fusionnet_layers_val'] = None

    # generate random identifier
    if args.identifier is not None:
        identifier = args.identifier
    else:
        identifier = args.identifier = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # create the model
    print('Creating model, this may take a second...')
    if args.pretrained_retinanet:
        model = create_model(weights=args.weights,
                             model_type=args.fusionnet_model_type, resnet_type=opt_resnet_type_val,
                             retinanet_model_path=args.pretrained_retinanet_model_path,
                             retinanet_freeze_weights=args.freeze_retinanet_weights,
                             fusionnet_layers=opt_fusionnet_layers_val)
    else:
        model = create_model(weights=args.weights,
                             model_type=args.fusionnet_model_type, resnet_type=opt_resnet_type_val,
                             fusionnet_layers=opt_fusionnet_layers_val)

    # compile model (note: set loss to None since loss is added inside layer)
    model.compile(
        loss={
            'fusionnet_output': keras.losses.mean_squared_error  # keras_retinanet.losses.focal()
        },
        # optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
        optimizer=keras.optimizers.SGD(lr=opt_lr_val, momentum=0.9, nesterov=True)  # decay=1e-6, lr= 0.01
    )

    # create image data generator objects
    train_image_data_generator = keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=False,
    )
    val_image_data_generator = keras.preprocessing.image.ImageDataGenerator()

    # check for the model type
    if args.fusionnet_model_type == 'fusionnetV0_0':
        grasp_generator_class = NicoGraspingGeneratorV0_0
    elif args.fusionnet_model_type == 'fusionnetV0_1':
        grasp_generator_class = NicoGraspingGeneratorV0_1
    elif args.fusionnet_model_type == 'fusionnetV1_0':
        grasp_generator_class = NicoGraspingGeneratorV1_0
    elif args.fusionnet_model_type == 'fusionnetV1_1':
        grasp_generator_class = NicoGraspingGeneratorV1_1
    else:
        grasp_generator_class = NicoGraspingGeneratorV0_0

    # create a generator for training data
    train_generator = grasp_generator_class(
        args.grasp_path,
        'train',
        train_image_data_generator,
        image_dir=args.image_dir,
        generate_sets=args.generate_train_val_test,
        train_val_test_proportions=args.train_val_test_proportions,
        batch_size=args.batch_size,
        normalize=True
    )

    # create a generator for testing data
    val_generator = grasp_generator_class(
        args.grasp_path,
        'val',
        val_image_data_generator,
        image_dir=args.image_dir,
        generate_sets=False,
        generate_categories=False,
        batch_size=args.batch_size
    )

    # print model summary
    model_summary = model.summary()
    print(model_summary)

    # save model structure to file
    plot_model(model,
               to_file=os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                    '..', 'images', train_generator.get_name() + '_model_plot.png'),
               show_shapes=True, show_layer_names=True)

    # start training
    loss_history = LossHistory()

    callbacks = [
        # identifier is ignored for checkpoints
        keras.callbacks.ModelCheckpoint(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                     '..', 'snapshots', train_generator.get_name() + '_' +
                                                     args.image_dir + '_' + str(args.epochs) + '_' +
                                                     str(args.batch_size) + '_' + str(args.pretrained_retinanet) +
                                                     '_' + str(args.freeze_retinanet_weights) +
                                                     '_nico_grasping_best.h5'),
                                        monitor='val_loss', verbose=2, save_best_only=True),

        keras.callbacks.CSVLogger(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                               '..', 'logs', train_generator.get_name() + '_' +
                                               args.image_dir + '_' + str(args.epochs) + '_' +
                                               str(args.batch_size) + '_' + str(args.pretrained_retinanet) +
                                               '_' + str(args.freeze_retinanet_weights) +
                                               '_' + str(identifier) + '_nico_grasping_log.csv'),
                                  append=False),

        loss_history,

    ]

    if opt_reducelr_plateau:
        callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1,
                                                           mode='auto', epsilon=0.0001, cooldown=0, min_lr=0))

    hist = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=len(train_generator.image_names) // args.batch_size,
        epochs=args.epochs,
        verbose=1,
        validation_data=val_generator,
        validation_steps=len(val_generator.image_names) // args.batch_size,  # 3000
        callbacks=callbacks,
    )

    # print(hist.history)

    # store final result too
    model.save(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            '..', 'snapshots', train_generator.get_name() + '_' +
                            args.image_dir + '_' + str(args.epochs) + '_' +
                            str(args.batch_size) + str(args.pretrained_retinanet) +
                            '_' + str(args.freeze_retinanet_weights) + '_nico_grasping_final.h5'))

    # store configuration file for storing all details
    store_configuration(args, params, model_summary)


    # delete model to free up memory
    K.clear_session()
    del model

    # return the training report
    return loss_history


def parse_args():
    parser = argparse.ArgumentParser(
        description='Training script for Nico object detection and grasping. '
                    'The output model is stored in /odg/src/keras_fusionnet/scripts/snapshots')
    parser.add_argument('--grasp-path',
                        help='Path to NICO Grasping directory (ie. /odg/datasets/GraspingNico2017).',
                        default="/odg/datasets/GraspingNico2017")
    parser.add_argument('--image-dir',
                        help='name of the image directory (ie. cleaned_samples_v0).',
                        default="balanced_cleaned_samples-v0")
    parser.add_argument('--fusionnet-model-type', help='the model chosen for training',
                        default="fusionnetV1_0")

    parser.add_argument('--pretrained-retinanet',
                        help='selecting whether pretrained RetinaNet weights are to be loaded',
                        default=False, type=bool)
    parser.add_argument('--pretrained-retinanet-model-path',
                        help='the pretrained RetinaNet model to be loaded',
                        default="/home/abawi/PycharmProjects/WTM_JOB/object_detection_grasping/src/keras_retinanet/scripts/resnet50_voc_best.h5")
    parser.add_argument('--freeze-retinanet-weights',
                        help='freezing the loaded weights',
                        default=False, type=bool)

    parser.add_argument('--weights',
                        help='weights to use for initialization (defaults to ImageNet)',
                        default='imagenet')
    parser.add_argument('--batch-size',
                        help='size of the batches', default=20, type=int)
    parser.add_argument('--epochs',
                        help='number of epochs', default=300, type=int)
    parser.add_argument('--identifier',
                        help='the identifier of the current run. This is a string. If None is set, '
                                             'a random identifier is generated', type=str)
    parser.add_argument('--gpu',
                        help='Id of the GPU to use (as reported by nvidia-smi).')

    parser.add_argument('--generate-train-val-test',
                        help='choose whether to generate a training, validation and test set on every run',
                        default=True, type=bool)

    parser.add_argument('--train-val-test-proportions',
                        help='choose the percentage training, validation and test sets respectively (i.e.: 0.8 0.1 0.1',
                        default=None, type=float, nargs=3)

    return parser.parse_args()


def run(args, optimization_params):
    return train(args, optimization_params)

if __name__ == '__main__':
    # set the numpy printing to verbose mode
    # np.set_printoptions(threshold=np.nan)

    # parse arguments
    args = parse_args()
    # train model
    training_report = train(args)

    print('best validation loss', training_report.best_val_loss,
          ' at epoch ', training_report.best_epoch,
          ' with a training loss of ',  training_report.best_train_loss)
