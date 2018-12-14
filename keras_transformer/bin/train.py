import sys
import argparse
import configparser
import os

from keras.optimizers import *
from keras.losses import sparse_categorical_crossentropy
from keras.callbacks import *
from keras.utils.vis_utils import plot_model

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_transformer.bin  # noqa: F401

    __package__ = "keras_transformer.bin"

from ..models.transformer import transformer, Transformer
from ..utils import helper
from ..preprocessing.generator import CSVGenerator
from ..utils.config import read_config_file, write_config_file
from .. import losses, metrics


class LRSchedulerPerStep(Callback):
    def __init__(self, d_model, warmup=4000):
        self.basic = int(d_model) ** -0.5
        self.warm = warmup ** -1.5
        self.step_num = 0

    def on_batch_begin(self, batch, logs=None):
        self.step_num += 1
        lr = self.basic * min(self.step_num ** -0.5, self.step_num * self.warm)
        K.set_value(self.model.optimizer.lr, lr)


class LRSchedulerPerEpoch(Callback):
    def __init__(self, d_model, warmup=4000, num_per_epoch=1000):
        self.basic = int(d_model) ** -0.5
        self.warm = warmup ** -1.5
        self.num_per_epoch = num_per_epoch
        self.step_num = 1

    def on_epoch_begin(self, epoch, logs=None):
        self.step_num += self.num_per_epoch
        lr = self.basic * min(self.step_num ** -0.5, self.step_num * self.warm)
        K.set_value(self.model.optimizer.lr, lr)

def parse_args(args):
    """ Parse the arguments.
    """

    parser = argparse.ArgumentParser(description='Simple training script for a Transformer network.')

    parser.add_argument('annotations',
                        help='Path to training source and target sequences file (both separated by a tab).')
    parser.add_argument('--val-annotations',
                        help='Path to validation source and target sequences file (both separated by a tab).')
    parser.add_argument('--vocab',
                        help='Load an already existing vocabulary file (optional).')

    parser.add_argument('--snapshot-path', help='The snapshot directory.', default='../../snapshots')
    parser.add_argument('--log-path',
                        help='The logging directory.', default='../../results')
    parser.add_argument('--config',
                        help='The configuration file.', default='config.ini')

    parser.add_argument('--batch-size', help='The size of a single batch.', default=64)
    parser.add_argument('--epochs', help='Number of epochs.', default=30)
    parser.add_argument('--steps', help='Number of steps per epoch.', type=int, default=None)


    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    configs = configparser.ConfigParser()
    if args.config is not None:
        configs = read_config_file(args.config)

    snapshot_path = helper.make_dir(args.snapshot_path)
    result_path = helper.make_dir(args.log_path)
    mfile = snapshot_path + 'transformer.h5'

    # store the args and configs
    helper.store_settings(store_object=args, json_file=snapshot_path + 'script_arguments.args')
    write_config_file(configs,  snapshot_path + 'config.ini')

    train_generator = CSVGenerator(args.annotations, batch_size=args.batch_size, tokens_file=args.vocab)
    i_tokens = train_generator.i_tokens
    o_tokens = train_generator.o_tokens

    if args.val_annotations:
        validation_generator = CSVGenerator(args.val_annotations, i_tokens=i_tokens, o_tokens=o_tokens, batch_size=args.batch_size)
        val_size = validation_generator.size()
    else:
        validation_generator = None
        val_size = None

    if args.steps is not None:
        train_size = args.steps
    else:
        train_size = train_generator.size()

    print('seq 1 words:', i_tokens.num())
    print('seq 2 words:', o_tokens.num())

    s2s = Transformer(i_tokens, o_tokens, **configs['init'])
    training_model = transformer(inputs=None, transformer_structure=s2s)
    lr_scheduler = LRSchedulerPerStep(configs['init']['d_model'], 4000)

    training_model.compile(
        loss={'transformer_classification': losses.masked_ce(layer_size=int(configs['init']['len_limit']))},
        optimizer=deserialize({'class_name': configs['optimizer']['class_name'],
                               'config':eval(configs['optimizer']['config'])}))

    model_saver = ModelCheckpoint(mfile, save_best_only=True, save_weights_only=True)
    csv_logger = CSVLogger(result_path + 'results.csv', append=True)

    training_model.summary()
    plot_model(training_model, to_file=snapshot_path + 'architecture.png', show_shapes=True, show_layer_names=True)

    try:
        training_model.load_weights(mfile)
    except:
        print('\n\nnew model')

    training_model.fit_generator(train_generator, epochs=args.epochs, shuffle=False, steps_per_epoch=train_size,
                                 callbacks=[lr_scheduler, model_saver, csv_logger],
                                 validation_data=validation_generator, validation_steps=val_size)


if __name__ == '__main__':

    main(args=None)