import sys
import argparse
import os

from keras.optimizers import *
from keras.losses import sparse_categorical_crossentropy
from keras.callbacks import *
from keras.utils.vis_utils import plot_model

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401

    __package__ = "keras_transformer.bin"

from ..models.transformer import transformer, Transformer
from ..utils import helper
from ..preprocessing.generator import CSVGenerator
from keras_transformer import losses, metrics


class LRSchedulerPerStep(Callback):
    def __init__(self, d_model, warmup=4000):
        self.basic = d_model ** -0.5
        self.warm = warmup ** -1.5
        self.step_num = 0

    def on_batch_begin(self, batch, logs=None):
        self.step_num += 1
        lr = self.basic * min(self.step_num ** -0.5, self.step_num * self.warm)
        K.set_value(self.model.optimizer.lr, lr)


class LRSchedulerPerEpoch(Callback):
    def __init__(self, d_model, warmup=4000, num_per_epoch=1000):
        self.basic = d_model ** -0.5
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

    parser.add_argument('--dataset-name',
                        help='Name of the datset to use.',
                        default='e2e')
    parser.add_argument('--train-file',
                        help='Path to training source and target sequences file (both separated by a tab).',
                        default='/home/abawi/PycharmProjects/THESIS/NEW/datasets/GeneratedData/TransformerMultimodalCSV/virtual/Annotations/annotations.csv')
    parser.add_argument('--valid-file',
                        help='Path to validation source and target sequences file (both separated by a tab).',
                        default='/home/abawi/PycharmProjects/THESIS/NEW/datasets/GeneratedData/TransformerMultimodalCSV/virtual/Annotations/annotations.csv')
    parser.add_argument('--id',
                        help='The unique identifier for the current run.',
                        default=118)
    parser.add_argument('--snapshot-dir',
                        help='The snapshot directory.',
                        default='../../snapshots')
    parser.add_argument('--logging-dir',
                        help='The logging directory.',
                        default='../../results')

    parser.add_argument('--model', help='The model to run [transformer].', default='transformer')
    parser.add_argument('--batch-size', help='The size of a single batch.', default=64)
    parser.add_argument('--epochs', help='Number of epochs.', default=30)
    parser.add_argument('--steps', help='Number of steps per epoch.', default=10000)

    return parser.parse_args(args)


def main(args=None, configs=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    snapshot_path = helper.make_dir(
        os.path.join(args.snapshot_dir, str(args.id))) + os.sep + args.model + '_' + args.dataset_name
    result_path = helper.make_dir(
        os.path.join(args.logging_dir, str(args.id))) + os.sep + args.model + '_' + args.dataset_name
    mfile = snapshot_path + '.model.h5'

    # store the args and configs
    helper.store_settings(store_object=args, json_file=snapshot_path + '.args')
    helper.store_settings(store_object=configs, json_file=snapshot_path + '.configs')

    train_generator = CSVGenerator(args.train_file, batch_size=args.batch_size)
    i_tokens = train_generator.i_tokens
    o_tokens = train_generator.o_tokens

    print('seq 1 words:', i_tokens.num())
    print('seq 2 words:', o_tokens.num())

    s2s = Transformer(i_tokens, o_tokens, **configs['transformer']['init'])
    training_model = transformer(inputs=None, transformer_structure=s2s)
    lr_scheduler = LRSchedulerPerStep(configs['transformer']['init']['d_model'], 4000)

    training_model.compile(
        loss={'transformer_regression': losses.masked_ce(layer_size=configs['transformer']['init']['len_limit'])},
        optimizer=deserialize(configs['transformer']['optimizer']))

    model_saver = ModelCheckpoint(mfile, save_best_only=True, save_weights_only=True)
    csv_logger = CSVLogger(result_path + '.log', append=True)

    training_model.summary()
    plot_model(training_model, to_file=snapshot_path + '.png', show_shapes=True, show_layer_names=True)

    try:
        training_model.load_weights(mfile)
    except:
        print('\n\nnew model')

    training_model.fit_generator(train_generator, epochs=args.epochs, shuffle=False, steps_per_epoch=args.steps,
                                 callbacks=[lr_scheduler, model_saver, csv_logger])


if __name__ == '__main__':
    configs = {
        'transformer': {
            'init': {
                'len_limit': 70,
                'd_model': 256,
                'd_inner_hid': 512,
                'n_head': 4,
                'd_k': 64,
                'd_v': 64,
                'layers': 2,
                'dropout': 0.1,
                'context_alignment_emb': False,
                'share_word_emb': False,
                'dilation': False,
                'dilation_rate': 3,
                'dilation_mode': 'non-linear',
                'dilation_layers': 6
            },
            'optimizer': {
                'class_name': 'adam',
                'config': {
                    'lr': 0.001,
                    'beta_1': 0.9,
                    'beta_2': 0.98,
                    'epsilon': 1e-9,
                    'decay': 0.,
                    'amsgrad': False
                }
            }
        },
        's2srnn': {
            'init': {
                'latent_dim': 256,
                'layers': 3,
            },
            'optimizer': {
                'class_name': 'rmsprop',
                'config': {}
            }
        }
    }

    main(args=None, configs=configs)