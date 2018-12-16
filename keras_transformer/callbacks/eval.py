"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@ contribution: Fares Abawi (6abawi@informatik.uni-hamburg.de)
"""

import keras
import numpy as np
from ..utils.eval import evaluate


class Evaluate(keras.callbacks.Callback):
    """ Evaluation callback for arbitrary datasets.
    """

    def __init__(
        self,
        generator,
        save_path=None,
        tensorboard=None,
        comet=None,
        verbose=1,
        evaluate_metrics=False
    ):
        """ Evaluate a given dataset using a given model at the end of every epoch during training.

        # Arguments
            generator        : The generator that represents the dataset to evaluate.
            save_path        : The path to save images with visualized detections to.
            tensorboard      : Instance of keras.callbacks.TensorBoard used to log values.
            verbose          : Set the verbosity level, by default this is set to 1.
        """
        self.generator       = generator
        self.save_path       = save_path
        self.tensorboard     = tensorboard
        self.comet           = comet
        self.verbose         = verbose
        self.evaluate_metrics = evaluate_metrics
        super(Evaluate, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # run evaluation
        metrics = evaluate(
            self.generator,
            self.model,
            save_path=self.save_path,
            evaluate_metrics=self.evaluate_metrics
        )

        # compute the e2e_metrics

        if self.tensorboard is not None and self.tensorboard.writer is not None:
            import tensorflow as tf
            summary = tf.Summary()
            for key, value in metrics.items():
                summary_value = summary.value.add()
                summary_value.simple_value = value
                summary_value.tag = key
                self.tensorboard.writer.add_summary(summary, epoch)
        if self.comet is not None:
            for key, value in metrics.items():
                self.comet.log_metric('transformer_'+key, value)

        if self.evaluate_metrics:
            for key, value in metrics.items():
                logs[key] = value

        if self.verbose == 1:
            print('metrics: ', metrics)
