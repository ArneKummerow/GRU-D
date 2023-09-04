from __future__ import absolute_import, division, print_function

import tensorflow.compat.v2 as tf

from keras import backend
from keras import activations, constraints, initializers, regularizers
from keras.engine import base_layer
from keras.engine.input_spec import InputSpec
from keras.layers import InputSpec, Layer, Activation
from keras.layers import Bidirectional
from keras.layers.rnn import gru_lstm_utils
from keras.layers.rnn import rnn_utils
from keras.layers.rnn.base_rnn import RNN
from keras.utils import get_custom_objects
from keras.utils import tf_utils

from keras.utils.generic_utils import has_arg
from keras.saving.serialization_lib import serialize_keras_object

from keras.layers.rnn.dropout_rnn_cell_mixin import DropoutRNNCellMixin
from .maksing_dropout_rnn_cell_mixin import MaskingDropoutRNNCellMixin

# from .activations import get_activation

__all__ = ['Bidirectional_for_GRUD', 'GRUDCell', 'GRUD']

get_custom_objects().update(
    {'exp_relu': Activation(lambda x: backend.exp(-backend.relu(x)))})

# @keras_export("keras.layers.GRUDCell", v1=[])
class GRUDCell(
    DropoutRNNCellMixin,
    MaskingDropoutRNNCellMixin,
    base_layer.BaseRandomLayer
):
    """Cell class for the GRU-D layer. An extension of `GRUCell`.
    Notice: Calling with only 1 tensor due to the limitation of Keras.
    Building, computing the shape with the input_shape as a list of length 3.
    # TODO: dynamic imputation
    """

    def __init__(
        self,

        ### GRUCell ###
        units,

        activation="tanh",
        recurrent_activation="sigmoid",

        use_bias=True,

        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",

        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,

        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,

        dropout=0.0,
        recurrent_dropout=0.0,

        ### GRUDCell ###
        x_imputation='zero',

        input_decay='exp_relu',
        hidden_decay='exp_relu',

        use_decay_bias=True,

        feed_masking=True,
        masking_decay=None,

        decay_initializer='zeros',
        decay_regularizer=None,
        decay_constraint=None,

        **kwargs
    ):
        if units <= 0:
            raise ValueError(
                "Received an invalid value for argument `units`, "
                f"expected a positive integer, got {units}."
            )
        if 'implementation' in kwargs and kwargs['implementation'] != 1:
            raise ValueError(
                "Received an invalid value for argument `implementation`, "
                "GRU-D only supports implementation=1, "
                f"got {kwargs['implementation']}."
            )
        if not x_imputation in _SUPPORTED_IMPUTATION:
            raise ValueError(
                "Received an invalid value for argument `x_imputation`, "
                f"expected one of {_SUPPORTED_IMPUTATION}, "
                f"got {kwargs['implementation']}."
            ) 
        if not feed_masking and (
            masking_decay is not None or masking_decay != 'None'):
            raise ValueError(
                "Received an invalid value for argument `feed_masking`, "
                f"`masking_decay` is {masking_decay}, "
                f"but `feed_masking` is {feed_masking}."
            )
            'Mask needs to be fed into GRU-D to enable the mask_decay.'

        # By default use cached variable under v2 mode, see b/143699808.
        if tf.compat.v1.executing_eagerly_outside_functions():
            self._enable_caching_device = kwargs.pop(
                "enable_caching_device", True
            )
        else:
            self._enable_caching_device = kwargs.pop(
                "enable_caching_device", False
            )
        
        super().__init__(units, **kwargs)

        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1.0, max(0.0, dropout))
        self.recurrent_dropout = min(1.0, max(0.0, recurrent_dropout))
        
        self.state_size = None
        self.output_size = self.units

        # GRUDCell
        self.x_imputation = x_imputation

        self.input_decay = activations.get(input_decay)
        self.hidden_decay = activations.get(hidden_decay)

        self.use_decay_bias = use_decay_bias

        self.feed_masking = feed_masking
        if self.feed_masking:
            self.masking_decay = activations.get(masking_decay)
        else:
            self.masking_decay = None
        
        if (self.input_decay is not None
            or self.hidden_decay is not None
            or self.masking_decay is not None):
            self.decay_initializer = initializers.get(decay_initializer)
            self.decay_regularizer = regularizers.get(decay_regularizer)
            self.decay_constraint = constraints.get(decay_constraint)
    
    @tf_utils.shape_type_conversion
    def build(
        self,
        input_shape
    ):
        if not isinstance(input_shape, list) or len(input_shape) != 3:
            raise ValueError(
                "Received an invalid value for argument `input_shape`, "
                f"expected a list of 3 inputs (x, m, s), got {input_shape}."
            )
        
        if input_shape[0] != input_shape[1]:
            raise ValueError(
                "The input x and the masking m should have "
                "the same input shape, but got "
                f"{input_shape[0]} and {input_shape[1]}.")
        
        if input_shape[0][0] != input_shape[2][0]:
            raise ValueError(
                "The input x and the timestamp s should have "
                "the same batch size, but got "
                f"{input_shape[0]} and {input_shape[2]}.")

        super().build(input_shape)

        input_dim = input_shape[0][-1]
        self.true_input_dim = input_dim
        self.state_size = (self.units, input_dim, input_dim)
        default_caching_device = rnn_utils.caching_device(self)

        # GRUCell
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 3),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            caching_device=default_caching_device,
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 3),
            name="recurrent_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
            caching_device=default_caching_device,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units * 3,),
                name="bias",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                caching_device=default_caching_device,
            )
        else:
            self.bias = None

        # GRUDCell
        if self.input_decay is not None:
            self.input_decay_kernel = self.add_weight(
                shape=(input_dim,),
                name='input_decay_kernel',
                initializer=self.decay_initializer,
                regularizer=self.decay_regularizer,
                constraint=self.decay_constraint,
                caching_device=default_caching_device,
            )
            if self.use_decay_bias:
                self.input_decay_bias = self.add_weight(
                    shape=(input_dim,),
                    name='input_decay_bias',
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                    caching_device=default_caching_device,
                )
        if self.hidden_decay is not None:
            self.hidden_decay_kernel = self.add_weight(
                shape=(input_dim, self.units),
                name='hidden_decay_kernel',
                initializer=self.decay_initializer,
                regularizer=self.decay_regularizer,
                constraint=self.decay_constraint,
                caching_device=default_caching_device,
            )
            if self.use_decay_bias:
                self.hidden_decay_bias = self.add_weight(
                    shape=(self.units,),
                    name='hidden_decay_bias',
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                    caching_device=default_caching_device,
                )
        if self.feed_masking:
            self.masking_kernel = self.add_weight(
                shape=(input_dim, self.units * 3),
                name='masking_kernel',
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                caching_device=default_caching_device,
            )
            if self.masking_decay is not None:
                self.masking_decay_kernel = self.add_weight(
                    shape=(input_dim,),
                    name='masking_decay_kernel',
                    initializer=self.decay_initializer,
                    regularizer=self.decay_regularizer,
                    constraint=self.decay_constraint,
                    caching_device=default_caching_device,
                )
                if self.use_decay_bias:
                    self.masking_decay_bias = self.add_weight(
                        shape=(input_dim,),
                        name='masking_decay_bias',
                        initializer=self.bias_initializer,
                        regularizer=self.bias_regularizer,
                        constraint=self.bias_constraint,
                        caching_device=default_caching_device,
                    )

        self.built = True

    def call(self, inputs, states, training=None):
        """
        Args:
            inputs: One tensor which is stacked by 3 inputs (x, m, s)
                x and m are of shape (n_batch * input_dim).
                s is of shape (n_batch, 1).
            states: states and other values from the previous step.
                (h_tm1, x_keep_tm1, s_prev_tm1)
        """
        # Get inputs and states
        # input_x = inputs[:, :self.true_input_dim]
        # input_m = inputs[:, self.true_input_dim:-1]
        # input_s = inputs[:, -1:]
        input_x, input_m, input_s = inputs
        h_tm1, x_keep_tm1, s_prev_tm1 = states
        
        # previous memory ([n_batch * self.units])
        # previous input x ([n_batch * input_dim])
        # and the subtraction term (of delta_t^d in Equation (2))
        # ([n_batch * input_dim])
        input_1m = backend.cast_to_floatx(1.) - input_m
        input_d = input_s - s_prev_tm1
        
        dp_mask = self.get_dropout_mask_for_cell(input_x, training, count=3)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            h_tm1, training, count=3
        )
        if self.feed_masking:
            m_dp_mask = self.get_masking_dropout_mask_for_cell(
                input_m, training, count=3
            )

        # Compute decay if any
        if self.input_decay is not None:
            gamma_di = input_d * self.input_decay_kernel
            if self.use_decay_bias:
                gamma_di = backend.bias_add(gamma_di, self.input_decay_bias)
            gamma_di = self.input_decay(gamma_di)
        if self.hidden_decay is not None:
            gamma_dh = backend.dot(input_d, self.hidden_decay_kernel)
            if self.use_decay_bias:
                gamma_dh = backend.bias_add(gamma_dh, self.hidden_decay_bias)
            gamma_dh = self.hidden_decay(gamma_dh)
        if self.feed_masking and self.masking_decay is not None:
            gamma_dm = input_d * self.masking_decay_kernel
            if self.use_decay_bias:
                gamma_dm = backend.bias_add(gamma_dm, self.masking_decay_bias)
            gamma_dm = self.masking_decay(gamma_dm)

        # Get the imputed or decayed input if needed
        # and `x_keep_t` for the next time step
        if self.input_decay is not None:
            x_keep_t = backend.switch(input_m, input_x, x_keep_tm1)
            x_t = backend.switch(input_m, input_x, gamma_di * x_keep_t)
        elif self.x_imputation == 'forward':
            x_t = backend.switch(input_m, input_x, x_keep_tm1)
            x_keep_t = x_t
        elif self.x_imputation == 'zero':
            x_t = backend.switch(input_m, input_x, backend.zeros_like(input_x))
            x_keep_t = x_t
        elif self.x_imputation == 'raw':
            x_t = input_x
            x_keep_t = x_t
        else:
            raise ValueError(
                "No input decay or invalid x_imputation "
                f"{self.x_imputation}.")

        # Get decayed hidden if needed
        if self.hidden_decay is not None:
            h_tm1d = gamma_dh * h_tm1
        else:
            h_tm1d = h_tm1

        # Get decayed masking if needed
        if self.feed_masking:
            m_t = input_1m
            if self.masking_decay is not None:
                m_t = gamma_dm * m_t

        # Apply the dropout
        if 0. < self.dropout < 1.:
            x_z, x_r, x_h = x_t * dp_mask[0], x_t * dp_mask[1], x_t * dp_mask[2]
            if self.feed_masking:
                m_z, m_r, m_h = (
                    m_t * m_dp_mask[0], m_t * m_dp_mask[1], m_t * m_dp_mask[2])
        else:
            x_z, x_r, x_h = x_t, x_t, x_t
            if self.feed_masking:
                m_z, m_r, m_h = m_t, m_t, m_t
        if 0. < self.recurrent_dropout < 1.:
            h_tm1_z, h_tm1_r = h_tm1d * rec_dp_mask[0], h_tm1d * rec_dp_mask[1]
        else:
            h_tm1_z, h_tm1_r = h_tm1d, h_tm1d

        # Get z_t, r_t, hh_t
        z_t = (backend.dot(x_z, self.kernel[:, :self.units])
             + backend.dot(h_tm1_z, self.recurrent_kernel[:, :self.units]))
        r_t = (backend.dot(x_r, self.kernel[:, self.units: self.units * 2])
             + backend.dot(
                h_tm1_r, self.recurrent_kernel[:,self.units:self.units * 2]))
        
        hh_t = backend.dot(x_h, self.kernel[:, self.units * 2:])

        if self.feed_masking:
            z_t += backend.dot(m_z, self.masking_kernel[:, :self.units])
            r_t += backend.dot(
                m_r, self.masking_kernel[:, self.units:self.units * 2])
            hh_t += backend.dot(m_h, self.masking_kernel[:, self.units * 2:])
        if self.use_bias:
            z_t = backend.bias_add(z_t, self.bias[: self.units])
            r_t = backend.bias_add(r_t, self.bias[self.units : self.units * 2])
            hh_t = backend.bias_add(hh_t, self.bias[self.units * 2 :])
        
        z_t = self.recurrent_activation(z_t)
        r_t = self.recurrent_activation(r_t)
        
        if 0. < self.recurrent_dropout < 1.:
            h_tm1_h = r_t * h_tm1d * rec_dp_mask[2]
        else:
            h_tm1_h = r_t * h_tm1d        
        hh_t = self.activation(hh_t + backend.dot(
                h_tm1_h, self.recurrent_kernel[:, self.units * 2:]))

        # get h_t
        h_t = z_t * h_tm1 + (1 - z_t) * hh_t
        if 0. < self.dropout + self.recurrent_dropout:
            if training is None:
                h_t._uses_learning_phase = True

        # get s_prev_t
        s_prev_t = backend.switch(
            input_m,
            backend.tile(input_s, [1, self.state_size[-1]]),
            s_prev_tm1)
        
        return h_t, [h_t, x_keep_t, s_prev_t]

    def get_config(self):
        config = {
            # GRUCell
            "units": self.units,
            "activation": activations.serialize(self.activation),
            "recurrent_activation": activations.serialize(
                self.recurrent_activation
            ),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(
                self.kernel_initializer
            ),
            "recurrent_initializer": initializers.serialize(
                self.recurrent_initializer
            ),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(
                self.kernel_regularizer
            ),
            "recurrent_regularizer": regularizers.serialize(
                self.recurrent_regularizer
            ),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "recurrent_constraint": constraints.serialize(
                self.recurrent_constraint
            ),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
            "implementation": self.implementation,
            
            # GRUDCell
            'x_imputation': self.x_imputation,
            'input_decay': activations.serialize(self.input_decay),
            'hidden_decay': activations.serialize(self.hidden_decay),
            'use_decay_bias': self.use_decay_bias,
            'feed_masking': self.feed_masking,
            'masking_decay': activations.serialize(self.masking_decay),
            'decay_initializer': initializers.serialize(self.decay_initializer),
            'decay_regularizer': regularizers.serialize(self.decay_regularizer),
            'decay_constraint': constraints.serialize(self.decay_constraint)
        }
        config.update(rnn_utils.config_for_enable_caching_device(self))
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return rnn_utils.generate_zero_filled_state_for_cell(
            self, inputs, batch_size, dtype
        )

    def get_initial_state_by_input_dim(
        self, batch_size_tensor, input_dim, dtype):
        return rnn_utils.generate_zero_filled_state(
            batch_size_tensor=batch_size_tensor,
            state_size=(self.units, input_dim, input_dim),
            dtype=dtype)

        

    # def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    #     if inputs is not None:
    #         batch_size = backend.shape(inputs[0])[0]
    #         dtype = inputs[0].dtype
    # # def get_initial_state(self, inputs):
    #     # initial_state = backend.zeros_like(inputs[0])  # (samples, input_dim)
    #     # initial_state = backend.sum(initial_state, axis=(1,))  # (samples,)
    #     # initial_state = backend.expand_dims(initial_state)  # (samples, 1)
    #     initial_state = backend.zeros((batch_size, 1), dtype=dtype)
    #     # ret = [backend.tile(initial_state, [1, dim]) for dim in self.state_size[:-1]]
    #     ret = [backend.tile(initial_state, [1, dim]) for dim in self.state_size]
    #     # initial_state for s_prev_tm1 should be the same as the first s
    #     # depending on the direction.

    #     # otherwise we take the first s.
    #     # return ret + [backend.tile(inputs[2][:, :], [1, self.state_size[-1]])]
    #     return ret


# @keras_export("keras.layers.GRUD", v1=[])
class GRUD(
    DropoutRNNCellMixin,
    MaskingDropoutRNNCellMixin,
    RNN,
    base_layer.BaseRandomLayer
):
    """Layer class for the GRU-D. An extension of GRU which utilizes
    missing data for better classification performance.
    Notice: constants is not used in GRUD.
    """

    def __init__(
        ### GRU ###
        self,
        units,
        activation='sigmoid',
        recurrent_activation='hard_sigmoid',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        recurrent_initializer='orthogonal',
        bias_initializer='zeros',
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        return_sequences=False,
        return_state=False,
        go_backwards=False,
        stateful=False,
        # unroll=False,
        time_major=False,

        ### GRUD ###
        x_imputation='zero',
        input_decay='exp_relu',
        hidden_decay='exp_relu',
        use_decay_bias=True,
        feed_masking=True,
        masking_decay=None,
        decay_initializer='zeros',
        decay_regularizer=None,
        decay_constraint=None,
        **kwargs
    ):
        # return_runtime is a flag for testing, which shows the real backend
        # implementation chosen by grappler in graph mode.
        self._return_runtime = kwargs.pop("return_runtime", False)

        if "enable_caching_device" in kwargs:
            cell_kwargs = {
                "enable_caching_device": kwargs.pop("enable_caching_device")
            }
        else:
            cell_kwargs = {}

        cell = GRUDCell(
            units=units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            x_imputation=x_imputation,
            input_decay=input_decay,
            hidden_decay=hidden_decay,
            use_decay_bias=use_decay_bias,
            feed_masking=feed_masking,
            masking_decay=masking_decay,
            decay_initializer=decay_initializer,
            decay_regularizer=decay_regularizer,
            decay_constraint=decay_constraint,
            dtype=kwargs.get("dtype"),
            trainable=kwargs.get("trainable", True),
            **cell_kwargs,
        )

        if 'unroll' in kwargs and kwargs['unroll']:
            raise ValueError(
                "Received an invalid value for argument `unroll`, "
                f"GRUD does not support unroll, got unroll={unroll}."
            )

        super().__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            time_major=time_major,
            **kwargs
        )

        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.input_spec = [
            InputSpec(ndim=3), InputSpec(ndim=3), InputSpec(ndim=3)]

    def compute_output_shape(self, input_shape):
        """Even if `return_state` = True, we do not return x_keep and ss
        (the last 2 states) since they are useless.
        """
        output_shape = super().compute_output_shape(input_shape)
        if self.return_state:
            return output_shape[:-2]
        return output_shape

    def compute_mask(self, inputs, mask):
        """Even if `return_state` is True, we do not return x_keep and ss
        (the last 2 states) since they are useless.
        """
        output_mask = super().compute_mask(inputs, mask)
        if self.return_state:
            return output_mask[:-2]
        return output_mask

    def build(self, input_shape):
        # Note input_shape will be list of shapes of initial states
        # if these are passed in __call__.

        if not isinstance(input_shape, list) or len(input_shape) < 3:
            raise ValueError(
                "Received an invalid value for argument `input_shape`, "
                f"expected a list of at least 3, got {input_shape}."
            )
        input_shape = input_shape[:3]

        batch_size = input_shape[0][0] if self.stateful else None
        self.input_spec[0] = InputSpec(
            shape=(batch_size, None, input_shape[0][-1]))
        self.input_spec[1] = InputSpec(
            shape=(batch_size, None, input_shape[1][-1]))
        self.input_spec[2] = InputSpec(
            shape=(batch_size, None, 1))

        # allow GRUDCell to build before we set or validate state_spec
        step_input_shape = [(i_s[0],) + i_s[2:] for i_s in input_shape]
        self.cell.build(step_input_shape)

        # set or validate state_spec
        state_size = list(self.cell.state_size)

        if self.state_spec is not None:
            # initial_state was passed in call, check compatibility
            if [spec.shape[-1] for spec in self.state_spec] != state_size:
                raise ValueError(
                    'An `initial_state` was passed that is not compatible with '
                    '`cell.state_size`. Received `state_spec`={}; '
                    'however `cell.state_size` is '
                    '{}'.format(self.state_spec, self.cell.state_size))
        else:
            self.state_spec = [InputSpec(shape=(None, dim))
                               for dim in state_size]
        if self.stateful:
            self.reset_states()
        self.built = True

    def get_initial_state(self, inputs):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = backend.zeros_like(inputs[0])  # (samples, timesteps, input_dim)
        initial_state = backend.sum(initial_state, axis=(1, 2))  # (samples,)
        initial_state = backend.expand_dims(initial_state)  # (samples, 1)
        ret = [backend.tile(initial_state, [1, dim]) for dim in self.cell.state_size[:-1]]
        # initial_state for s_prev_tm1 should be the same as the first s
        # depending on the direction.
        if self.go_backwards:
            # if go_backwards, we take the last s
            # (we take the largest one in case the padded input can be invalid)
            return ret + [backend.tile(backend.max(inputs[2], axis=1),
                                 [1, self.cell.state_size[-1]])]
        # otherwise we take the first s.
        return ret + [backend.tile(inputs[2][:, 0, :], [1, self.cell.state_size[-1]])]

    def __call__(self, inputs, initial_state=None, **kwargs):
        # We skip `__call__` of `RNN` and `GRU` in this case and directly execute
        # GRUD's great-grandparent's method.
        inputs, initial_state = _standardize_grud_args(inputs, initial_state)

        if initial_state is None:
            return super(RNN, self).__call__(inputs, **kwargs)

        # If `initial_state` is specified and is Keras
        # tensors, then add it to the inputs and temporarily modify the
        # input_spec to include them.

        additional_inputs = []
        additional_specs = []
        kwargs['initial_state'] = initial_state
        additional_inputs += initial_state
        self.state_spec = [InputSpec(shape=backend.int_shape(state))
                           for state in initial_state]
        additional_specs += self.state_spec
        # at this point additional_inputs cannot be empty
        is_keras_tensor = backend.is_keras_tensor(additional_inputs[0])
        for tensor in additional_inputs:
            if backend.is_keras_tensor(tensor) != is_keras_tensor:
                raise ValueError('The initial state or constants of an RNN'
                                 ' layer cannot be specified with a mix of'
                                 ' Keras tensors and non-Keras tensors'
                                 ' (a "Keras tensor" is a tensor that was'
                                 ' returned by a Keras layer, or by `Input`)')

        if is_keras_tensor:
            # Compute the full input spec, including state and constants
            full_input = inputs + additional_inputs
            full_input_spec = self.input_spec + additional_specs
            # Perform the call with temporarily replaced input_spec
            original_input_spec = self.input_spec
            self.input_spec = full_input_spec
            output = super(RNN, self).__call__(full_input, **kwargs)
            self.input_spec = original_input_spec
            return output
        return super(RNN, self).__call__(inputs, **kwargs)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        # We need to rewrite this `call` method by combining `RNN`'s and `GRU`'s.
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None
        self.cell._masking_dropout_mask = None

        inputs = inputs[:3]

        if initial_state is not None:
            pass
        elif self.stateful:
            initial_state = self.states
        else:
            initial_state = self.get_initial_state(inputs)

        if len(initial_state) != len(self.states):
            raise ValueError('Layer has ' + str(len(self.states)) +
                             ' states but was passed ' +
                             str(len(initial_state)) +
                             ' initial states.')
        timesteps = backend.int_shape(inputs[0])[1]

        kwargs = {}
        if has_arg(self.cell.call, 'training'):
            kwargs['training'] = training

        def step(inputs, states):
            return self.cell.call(inputs, states, **kwargs)
        # concatenate the inputs and get the mask

        concatenated_inputs = backend.concatenate(inputs, axis=-1)
        mask = mask[0]
        last_output, outputs, states = backend.rnn(step,
                                             concatenated_inputs,
                                             initial_state,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             unroll=self.unroll,
                                             input_length=timesteps)
        if self.stateful:
            updates = []
            for i, state in enumerate(states):
                updates.append((self.states[i], state))
            self.add_update(updates, inputs)

        if self.return_sequences:
            output = outputs
        else:
            output = last_output

        # Properly set learning phase
        if getattr(last_output, '_uses_learning_phase', False):
            output._uses_learning_phase = True
            for state in states:
                state._uses_learning_phase = True

        if self.return_state:
            states = list(states)[:-2] # remove x_keep and ss
            return [output] + states
        return output

    @property
    def x_imputation(self):
        return self.cell.x_imputation

    @property
    def input_decay(self):
        return self.cell.input_decay

    @property
    def hidden_decay(self):
        return self.cell.hidden_decay

    @property
    def use_decay_bias(self):
        return self.cell.use_decay_bias

    @property
    def feed_masking(self):
        return self.cell.feed_masking

    @property
    def masking_decay(self):
        return self.cell.masking_decay

    @property
    def decay_initializer(self):
        return self.cell.decay_initializer

    @property
    def decay_regularizer(self):
        return self.cell.decay_regularizer

    @property
    def decay_constraint(self):
        return self.cell.decay_constraint

    def get_config(self):
        config = {'x_imputation': self.x_imputation,
                  'input_decay': serialize_keras_object(self.input_decay),
                  'hidden_decay': serialize_keras_object(self.hidden_decay),
                  'use_decay_bias': self.use_decay_bias,
                  'feed_masking': self.feed_masking,
                  'masking_decay': serialize_keras_object(self.masking_decay),
                  'decay_initializer': initializers.get(self.decay_initializer),
                  'decay_regularizer': regularizers.get(self.decay_regularizer),
                  'decay_constraint': constraints.get(self.decay_constraint)}
        base_config = super().get_config()
        for c in ['implementation']:
            del base_config[c]
        return dict(list(base_config.items()) + list(config.items()))


class Bidirectional_for_GRUD(Bidirectional):
    def __call__(self, inputs, initial_state=None, constants=None, **kwargs):
        # We skip the `__call__()` of `Bidirectional`
        # and handle the differences in all cases.

        inputs, initial_state = _standardize_grud_args(
            inputs, initial_state)
        
        if initial_state is None and constants is None:
            return super(Bidirectional, self).__call__(inputs, **kwargs)

        # Applies the same workaround as in `RNN.__call__`
        additional_inputs = []
        additional_specs = []
        if initial_state is not None:
            # Check if `initial_state` can be splitted into half
            num_states = len(initial_state)
            if num_states % 2 > 0:
                raise ValueError(
                    'When passing `initial_state` to a Bidirectional RNN, '
                    'the state should be a list containing the states of '
                    'the underlying RNNs. '
                    'Found: ' + str(initial_state))

            kwargs['initial_state'] = initial_state
            additional_inputs += initial_state
            state_specs = [InputSpec(shape=backend.int_shape(state))
                           for state in initial_state]
            self.forward_layer.state_spec = state_specs[:num_states // 2]
            self.backward_layer.state_spec = state_specs[num_states // 2:]
            additional_specs += state_specs
        if constants is not None:
            kwargs['constants'] = constants
            additional_inputs += constants
            constants_spec = [InputSpec(shape=backend.int_shape(constant))
                              for constant in constants]
            self.forward_layer.constants_spec = constants_spec
            self.backward_layer.constants_spec = constants_spec
            additional_specs += constants_spec

            self._num_constants = len(constants)
            self.forward_layer._num_constants = self._num_constants
            self.backward_layer._num_constants = self._num_constants

        is_keras_tensor = backend.is_keras_tensor(additional_inputs[0])
        for tensor in additional_inputs:
            if backend.is_keras_tensor(tensor) != is_keras_tensor:
                raise ValueError('The initial state of a Bidirectional'
                                 ' layer cannot be specified with a mix of'
                                 ' Keras tensors and non-Keras tensors'
                                 ' (a "Keras tensor" is a tensor that was'
                                 ' returned by a Keras layer, or by `Input`)')

        if is_keras_tensor:
            # Compute the full input spec, including state
            full_input = [inputs] + additional_inputs
            full_input_spec = self.input_spec + additional_specs

            # Perform the call with temporarily replaced input_spec
            original_input_spec = self.input_spec
            self.input_spec = full_input_spec
            output = super(Bidirectional, self).__call__(full_input, **kwargs)
            self.input_spec = original_input_spec
            return output
        return super(Bidirectional, self).__call__(inputs, **kwargs)

def _standardize_grud_args(inputs, initial_state):
    """Standardize `__call__` to a single list of tensor inputs,
    specifically for GRU-D.

    Args:
        inputs: list/tuple of tensors
        initial_state: tensor or list of tensors or None

    Returns:
        inputs: list of 3 tensors
        initial_state: list of tensors or None
    """
    if not isinstance(inputs, list) or len(inputs) <= 2:
        raise ValueError('inputs to GRU-D should be a list of at least 3 tensors.')
    if initial_state is None:
        if len(inputs) > 3:
            initial_state = inputs[3:]
        inputs = inputs[:3]
    def to_list_or_none(x):
        if x is None or isinstance(x, list):
            return x
        if isinstance(x, tuple):
            return list(x)
        return [x]
    # end of `to_list_or_none()`
    
    initial_state = to_list_or_none(initial_state)
    return inputs, initial_state

_SUPPORTED_IMPUTATION = ['zero', 'forward', 'raw']

def _get_grud_layers_scope_dict():
    return {
        'Bidirectional_for_GRUD': Bidirectional_for_GRUD,
        'GRUDCell': GRUDCell,
        'GRUD': GRUD,
    }