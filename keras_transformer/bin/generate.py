import sys
import argparse

from keras.callbacks import *

from utils import helper
from preprocessing import dataloader as dd

def parse_args(args):
    """ Parse the arguments.
    """

    parser = argparse.ArgumentParser(description='Simple generation script for a Transformer network.')

    parser.add_argument('--dataset-name',
                        help='Name of the datset to use e.g. [e2e, wmt]',
                        default='e2e')
    parser.add_argument('--id',
                        help='The unique identifier for the current run.',
                        default=116)
    parser.add_argument('--snapshot-dir',
                        help='The snapshot directory.',
                        default='../snapshots')

    parser.add_argument('--model', help='The model to run [transformer, s2srnn].', default='transformer')

    # TODO: Add arguments to alter transformer and s2s properties (not important at the moment)

    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    snapshot_path = helper.make_dir(os.path.join(args.snapshot_dir, str(args.id))) + os.sep + args.model + '_' + args.dataset_name
    mfile = snapshot_path + '.model.h5'

    # load configs
    configs = helper.load_settings(json_file=snapshot_path + '.configs')

    itokens, otokens = dd.MakeS2SDict(None, dict_file=snapshot_path + '_word.txt')

    if args.model == 's2srnn':
        from models.rnn_s2s import RNNSeq2Seq
        s2s = RNNSeq2Seq(itokens,otokens,**configs['s2srnn']['init'])
        s2s.compile()
    elif args.model == 'transformer':
        from models.transformer import Transformer, LRSchedulerPerStep
        s2s = Transformer(itokens, otokens,**configs['transformer']['init'])
        s2s.compile()

    try:
        s2s.model.load_weights(mfile)

    except:
        print('\n\nModel not found or incompatible with network! Exiting now')
        exit(-1)

    start = time.clock()
    print(s2s.decode_sequence_fast(
        dd.parenthesis_split('name[Alimentum] , area[city centre] , familyFriendly[yes] , near[Burger King]',
                             delimiter=" ", lparen="[", rparen="]"), delimiter=' '))
    end = time.clock()
    print("Time per sequence: {} ".format((end - start)))
    while True:
        quest = input('> ')
        print(
            s2s.decode_sequence_fast(dd.parenthesis_split(quest, delimiter=' ', lparen="[", rparen="]"), delimiter=' '))
        rets = s2s.beam_search(dd.parenthesis_split(quest, delimiter=' ', lparen="[", rparen="]"), delimiter=' ')
        for x, y in rets: print(x, y)


if __name__ == '__main__':
    main()