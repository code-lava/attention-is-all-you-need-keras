from random import shuffle
from itertools import compress

from ..preprocessing.dataloader import parenthesis_split

import numpy as np

class MaskableList(list):
    def __getitem__(self, index):
        try: return super(MaskableList, self).__getitem__(index)
        except TypeError: return MaskableList(compress(self, index))

def drop_source_words(input_file, output_file, drop_keep_probabilities, shuffle_mode=2, keep_target=False):
    # shuffle_mode:
    # 0: do not shuffle words
    # 1: shuffle words and add only shuffled words
    # 2: add shuffled and unshuffled words

    with open(input_file) as f:
        lines = f.readlines()

    outputs = []
    for line_raw in lines:
        line_raw = line_raw.split('\t')
        # begin splitting the words
        source = parenthesis_split(line_raw[0].replace('\n',''), ' ', lparen='[', rparen=']')
        print(line_raw[1])
        target = parenthesis_split(line_raw[1 if keep_target else 0].replace('\n',''), ' ', lparen='[', rparen=']')

        random_probs = drop_keep_probabilities
        # shuffle(random_probs)
        for p in random_probs:
            masked_source = MaskableList(source)

            n = 1
            mask = np.random.binomial(n, p, len(source))

            # safety measure to include atleast 1 word in the end
            if np.count_nonzero(mask) <= 1:
                mask = np.random.binomial(n, 0.3, len(source))
                if np.count_nonzero(mask) <= 1:
                    mask = np.random.binomial(n, 0.5, len(source))
                    if np.count_nonzero(mask) <= 1:
                        mask = np.random.binomial(n, 1, len(source))

            masked_source = masked_source[mask]

            if shuffle_mode == 0 or shuffle_mode == 2:
                output = ' '.join(masked_source) + '\t' + ' '.join(target)
                outputs.append(output)
            if shuffle_mode == 1 or shuffle_mode == 2:
                shuffle(masked_source)
                output = ' '.join(masked_source) + '\t' + ' '.join(target)
                outputs.append(output)

    with open(output_file, 'w') as f:
        for output in outputs:
          f.write("%s\n" % output)


if __name__ == '__main__':
    drop_keep_prob = [0.05] # , 0.05, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6, 0.7, 0.7, 0.8, 0.8, 0.9

    drop_source_words('../data/wmt/raw.trainset.txt', '../data/wmt/trainset_noise.txt', drop_keep_prob, shuffle_mode=2,
                      keep_target=False)
    drop_source_words('../data/wmt/raw.devset.txt', '../data/wmt/devset_noise.txt', [0.1, 0.3, 0.5], shuffle_mode=0,
                      keep_target=False)

    ## according to et al. [0.6]
    drop_keep_prob = [0.6, 1]
    drop_source_words('../data/e2e/trainset.txt', '../data/e2e/trainset_noise_google_method.txt', drop_keep_prob, shuffle_mode=2,
                      keep_target=True)
    drop_source_words('../data/e2e/trainset.txt', '../data/e2e/devset_noise.txt', drop_keep_prob, shuffle_mode=0,
                      keep_target=True)