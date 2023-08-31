"""Mixin holding masking dropout fields for RNN cells."""


import tensorflow.compat.v2 as tf
from tensorflow.tools.docs import doc_controls
from keras.layers.rnn.dropout_rnn_cell_mixin import _generate_dropout_mask

from keras import backend


class MaskingDropoutRNNCellMixin:

    def __init__(self, *args, **kwargs):
        self._create_non_trackable_mask_cache()
        super().__init__(*args, **kwargs)

    def _create_non_trackable_mask_cache(self):
        self._masking_dropout_mask_cache = backend.ContextValueCache(
            self._create_masking_dropout_mask
        )

    def reset_masking_dropout_mask(self):
        self._masking_dropout_mask_cache.clear()

    def _create_masking_dropout_mask(self, inputs, training, count=1):
        return _generate_dropout_mask(
            self._random_generator,
            tf.ones_like(inputs),
            self.masking_dropout,
            training=training,
            count=count,
        )

    def get_masking_dropout_mask_for_cell(self, inputs, training, count=1):
        if self.masking_dropout == 0:
            return None
        init_kwargs = dict(inputs=inputs, training=training, count=count)
        return self._masking_dropout_mask_cache.setdefault(kwargs=init_kwargs)

    def __getstate__(self):
        state = super().__getstate__()
        state.pop("_masking_dropout_mask_cache", None)
        return state

    def __setstate__(self, state):
        state["_masking_dropout_mask_cache"] = backend.ContextValueCache(
            self._create_masking_dropout_mask
        )
        super().__setstate__(state)
