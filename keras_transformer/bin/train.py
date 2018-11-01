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

from ..models.transformer import transformer
from ..utils import helper
from ..preprocessing import dataloader as dd
from .. import losses

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

    parser.add_argument('--model', help='The model to run [transformer, s2srnn].', default='transformer')
    parser.add_argument('--batch-size', help='The size of a single batch.', default=64)
    parser.add_argument('--epochs', help='Number of epochs.', default=30)

    # TODO: Add arguments to alter transformer and s2s configs (not important at the moment)

    return parser.parse_args(args)

def main(args=None, configs=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)


    # dataset_name = args.train_file.split(os.sep)[-2]
    snapshot_path = helper.make_dir(os.path.join(args.snapshot_dir, str(args.id))) + os.sep + args.model + '_' + args.dataset_name
    result_path = helper.make_dir(os.path.join(args.logging_dir, str(args.id))) + os.sep + args.model + '_' + args.dataset_name
    mfile = snapshot_path + '.model.h5'

    # store the args and configs
    helper.store_settings(store_object=args, json_file=snapshot_path + '.args')
    helper.store_settings(store_object=configs, json_file=snapshot_path + '.configs')

    itokens, otokens = dd.MakeS2SDict(args.train_file, dict_file=snapshot_path + '_word.txt')
    Xtrain, Ytrain = dd.MakeS2SData(args.train_file, itokens, otokens, h5_file=snapshot_path + '.train.h5')
    Xvalid, Yvalid = dd.MakeS2SData(args.valid_file, itokens, otokens, h5_file=snapshot_path + '.valid.h5')

    print('seq 1 words:', itokens.num())
    print('seq 2 words:', otokens.num())
    print('train shapes:', Xtrain.shape, Ytrain.shape)
    print('valid shapes:', Xvalid.shape, Yvalid.shape)

    if args.model == 's2srnn':
        from ..models.rnn_s2s import RNNSeq2Seq
        s2s = RNNSeq2Seq(itokens,otokens,**configs['s2srnn']['init'])
        s2s.compile(deserialize(configs['s2srnn']['optimizer']))
    elif args.model == 'transformer':
        from ..models.transformer import Transformer, LRSchedulerPerStep
        s2s = Transformer(itokens, otokens,**configs['transformer']['init'])
        prediction_model, training_model = transformer(inputs=None, transformer_structure=s2s)
        # below is the baseline transformer
        # s2s = Transformer(itokens, otokens, len_limit=70, d_model=d_model, d_inner_hid=512,
        #                   n_head=4, d_k=64, d_v=64, layers=2, dropout=0.1, context_emb=False,
        #                   dilation=False, share_word_emb=False)
        lr_scheduler = LRSchedulerPerStep(configs['transformer']['init']['d_model'], 4000)
        # lr_scheduler = LRSchedulerPerEpoch(d_model, 4000, Xtrain.shape[0]/64)  # this scheduler only update lr per epoch
        # s2s.compile(deserialize(configs['transformer']['optimizer']))
        training_model.compile(loss={'tgt_layer': losses.masked_ce()},
                               optimizer=deserialize(configs['transformer']['optimizer']))


    model_saver = ModelCheckpoint(mfile, save_best_only=True, save_weights_only=True)
    csv_logger = CSVLogger(result_path+'.log', append=True)

    training_model.summary()
    plot_model(training_model, to_file=snapshot_path+'.png', show_shapes=True, show_layer_names=True)

    try:
        training_model.load_weights(mfile)
    except:
        print('\n\nnew model')

    if args.model == 's2srnn':
        training_model.fit([Xtrain, Ytrain], None,  batch_size=args.batch_size, epochs=args.epochs,
                      validation_data=([Xvalid, Yvalid], None),
                      callbacks=[model_saver, csv_logger])
    elif args.model == 'transformer':
        training_model.fit([Xtrain, Ytrain], [Ytrain], batch_size=args.batch_size, epochs=args.epochs,
                      validation_data=([Xvalid, Yvalid], [Yvalid]), shuffle=True,
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
                'dilation': True,
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