import keras.backend as backend
from keras.layers.core import Lambda

def masked_ce():
    def _masked_ce(y_true, y_pred):
        # backend.reshape(y_true, (102))
        y_true = backend.tf.reshape(y_true[:, 1:], [-1,99])
        y_true = backend.cast(y_true, 'int32')
        # y_true = backend.print_tensor(y_true, 'y_true values')
        # y_pred = backend.print_tensor(y_pred, 'y_pred values')
        # print(y_true.get_shape())
        loss = backend.tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        mask = backend.tf.cast(backend.tf.not_equal(y_true, 0), 'float32')
        loss = backend.tf.reduce_sum(loss * mask, -1) / backend.tf.reduce_sum(mask, -1)
        loss = backend.mean(loss)
        return loss
    return _masked_ce


def get_accu(args):
    y_true,  y_pred = args
    mask = backend.tf.cast(backend.tf.not_equal(y_true, 0), 'float32')
    corr = backend.cast(backend.equal(backend.cast(y_true, 'int32'),
                                      backend.cast(backend.argmax(y_pred, axis=-1), 'int32')), 'float32')
    corr = backend.sum(corr * mask, -1) / backend.sum(mask, -1)
    return backend.mean(corr)