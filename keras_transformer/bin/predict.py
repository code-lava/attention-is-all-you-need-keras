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
from ..utils.config import read_config_file

def parse_args(args):
    """ Parse the arguments.
    """

    parser = argparse.ArgumentParser(description='Simple generation script for a Transformer network.')

    parser.add_argument('snapshot', help='Resume training from a snapshot.')
    parser.add_argument('vocab', help='Load an already existing vocabulary file.')
    parser.add_argument('--config', help='The configuration file.', default='config.ini')

    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    mfile = args.snapshot
    print(mfile)
    # load configs
    configs = read_config_file(args.config)

    i_tokens, o_tokens = dd.make_s2s_dict(None, dict_file=args.vocab)


    s2s = Transformer(i_tokens, o_tokens,**configs['init'])
    model = transformer(inputs=None, transformer_structure=s2s)
    # model = transformer_inference(model)
    try:
        model.load_weights(mfile)
        model.compile('adam', 'mse')
    except:
        print('\n\nModel not found or incompatible with network! Exiting now')
        exit(-1)

    start = time.clock()
    padded_line = helper.parenthesis_split('Move the red cube on top of the blue cube',
                             delimiter=" ", lparen="[", rparen="]")

    ret = _decode_sequence(model=model,
                           input_seq=padded_line,
                           i_tokens=i_tokens,
                           o_tokens=o_tokens,
                           len_limit=int(configs['init']['len_limit']))
    end = time.clock()
    print("Time per sequence: {} ".format((end - start)))
    print(ret)
    while True:
        quest = input('> ')
        padded_line = helper.parenthesis_split(quest,
                                               delimiter=" ", lparen="[", rparen="]")
        rets = _beam_search(
            model=model,
            input_seq=padded_line,
            i_tokens=i_tokens,
            o_tokens=o_tokens,
            len_limit=int(configs['init']['len_limit']),
            topk=1,
            delimiter=' ')
        for x, y in rets:
            print(x, y)


if __name__ == '__main__':
    main()