import sys
import argparse
from os.path import basename, splitext

from keras.callbacks import *

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_transformer.bin"

from ..utils import helper, prepare_evaluation
from ..preprocessing import dataloader as dd
from ..utils.evaluation.e2emetrics import measure_scores
from ..models.transformer import transformer, transformer_inference, Transformer
from ..utils.eval import _beam_search, _decode_sequence

def parse_args(args):
    """ Parse the arguments.
    """

    parser = argparse.ArgumentParser(description='Evaluation script which generates sentences and then evaluates using metrics: BLEU, NIST, ROUGE,_L, CIDEr, METEOR.')

    parser.add_argument('--dataset-name',
                        help='Name of the datset to use.',
                        default='e2e')
    parser.add_argument('--valid-file',
                        help='Path to validation source and target sequences file (both separated by a tab).',
                        default='../data/e2e/devset.txt')
    parser.add_argument('--evaluate-metrics',
                        help='If set to true, will evaluate on e2e-metrics and saves the output results.',
                        default=True)
    parser.add_argument('--generate',
                        help='Generates and saves the output sentence to the snapshot directory.',
                        default=True)
    parser.add_argument('--beam-search',
                        help='If set to true,returns the best beam search output.',
                        default=True)
    parser.add_argument('--beam-width',
                        help='Size of the beam width if beam search is used.',
                        default=10)

    parser.add_argument('--id',
                        help='The unique identifier for the current run.',
                        default=119)
    parser.add_argument('--snapshot-dir',
                        help='The snapshot directory.',
                        default='../../snapshots')
    parser.add_argument('--logging-dir',
                        help='The logging directory.',
                        default='../../results')

    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    snapshot_path = helper.make_dir(os.path.join(args.snapshot_dir, str(args.id))) + os.sep + '_' + args.dataset_name
    result_path = helper.make_dir(os.path.join(args.logging_dir, str(args.id))) + os.sep +  '_' + args.dataset_name
    mfile = snapshot_path + '.model.h5'
    golden_file = args.valid_file[0:args.valid_file.rindex('.')] + '.metric_golden.txt'
    baseline_file = snapshot_path + '.metric_baseline.txt'

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

    if args.create_golden:
        prepare_evaluation.create_golden_sentences(args.valid_file, golden_file)

    with open(args.valid_file, 'r') as fval:
        lines = fval.readlines()

    if args.generate:
        outputs = []
        prev_line = ''
        for line_raw_index, line_raw in enumerate(lines):
            line_raw = line_raw.split('\t')
            if prev_line != line_raw[0]:
                padded_line = helper.parenthesis_split(line_raw[0], delimiter=' ', lparen="[", rparen="]")
                if args.beam_search:
                    rets = _beam_search(
                        model=model,
                        input_seq=padded_line,
                        i_tokens=i_tokens,
                        o_tokens=o_tokens,
                        len_limit=configs['transformer']['init']['len_limit'],
                        topk=args.beam_width,
                        delimiter=' ')
                    for x, y in rets:
                        print(x)
                        outputs.append(x)
                        break
                else:
                    ret = _decode_sequence(model=model,
                                           input_seq=padded_line,
                                           i_tokens=i_tokens,
                                           o_tokens=o_tokens,
                                           len_limit=configs['transformer']['init']['len_limit'])
                    outputs.append(ret)
            prev_line = line_raw[0]

        with open(baseline_file, 'w') as fbase:
            for output in outputs:
                fbase.write("%s\n" % output)
        del outputs

    if args.evaluate_metrics:
        data_src, data_ref, data_sys = measure_scores.load_data(golden_file, baseline_file, None)
        measure_names, scores = measure_scores.evaluate(data_src, data_ref, data_sys)
        print(scores)

        if args.beam_search:
            search_method = 'beamsrch'
        else:
            search_method = 'greedy'
        valid_name = splitext(basename(args.valid_file))[0]
        helper.store_settings(scores.__repr__(), result_path + '_' + search_method + '_' + valid_name + '.metric_eval')


if __name__ == '__main__':
    main()