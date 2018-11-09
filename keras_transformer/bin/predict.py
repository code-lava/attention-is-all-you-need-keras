import sys
import argparse

from keras.callbacks import *

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_transformer.bin"

from ..utils import helper
from ..preprocessing import dataloader as dd
from ..models.transformer import transformer, transformer_inference, Transformer
from ..utils.eval import _beam_search, _decode_sequence


def parse_args(args):
    """ Parse the arguments.
    """

    parser = argparse.ArgumentParser(description='Simple generation script for a Transformer network.')

    parser.add_argument('--dataset-name',
                        help='Name of the datset to use e.g. [e2e, wmt]',
                        default='e2e')
    parser.add_argument('--id',
                        help='The unique identifier for the current run.',
                        default=118)
    parser.add_argument('--snapshot-dir',
                        help='The snapshot directory.',
                        default='../../snapshots')

    parser.add_argument('--model', help='The model to run [transformer].', default='transformer')

    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    snapshot_path = helper.make_dir(os.path.join(args.snapshot_dir, str(args.id))) + \
                    os.sep + args.model + '_' + args.dataset_name
    mfile = snapshot_path + '.model.h5'

    # load configs
    configs = helper.load_settings(json_file=snapshot_path + '.configs')

    i_tokens, o_tokens = dd.make_s2s_dict(None, dict_file=snapshot_path + '_word.txt')


    s2s = Transformer(i_tokens, o_tokens,**configs['transformer']['init'])
    model = transformer(inputs=None, transformer_structure=s2s)
    model = transformer_inference(model)
    try:
        model.load_weights(mfile)
        model.compile('adam', 'mse')
    except:
        print('\n\nModel not found or incompatible with network! Exiting now')
        exit(-1)

    start = time.clock()
    padded_line = helper.parenthesis_split('name[Alimentum] , area[city centre] , familyFriendly[yes] , near[Burger King]',
                             delimiter=" ", lparen="[", rparen="]")

    ret = _decode_sequence(model=model,
                           input_seq=padded_line,
                           i_tokens=i_tokens,
                           o_tokens=o_tokens,
                           len_limit=configs['transformer']['init']['len_limit'])
    end = time.clock()
    print("Time per sequence: {} ".format((end - start)))
    print(ret)
    while True:
        quest = input('> ')
        rets = _beam_search(
            model=model,
            input_seq=padded_line,
            i_tokens=i_tokens,
            o_tokens=o_tokens,
            len_limit=configs['transformer']['init']['len_limit'],
            topk=10,
            delimiter=' ')
        for x, y in rets:
            print(x, y)


if __name__ == '__main__':
    main()