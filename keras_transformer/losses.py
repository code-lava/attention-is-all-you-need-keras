import keras.backend as backend

def masked_ce(layer_size):

    def _masked_ce(y_true, y_pred):
        y_true = backend.tf.reshape(y_true[:, 1:], [-1,layer_size-1])
        y_true = backend.cast(y_true, 'int32')
        loss = backend.tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        mask = backend.tf.cast(backend.tf.not_equal(y_true, 0), 'float32')
        loss = backend.tf.reduce_sum(loss * mask, -1) / backend.tf.reduce_sum(mask, -1)
        loss = backend.mean(loss)
        return loss
    return _masked_ce
