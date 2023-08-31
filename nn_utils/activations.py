from __future__ import absolute_import, division, print_function

from keras import activations
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

get_custom_objects().update({'exp_relu': Activation(exp_relu)})

__all__ = ['exp_relu', 'get_activation']

def exp_relu(x):
    return K.exp(-K.relu(x))

# def get_activation(identifier):
#     if identifier is None:
#         return None
#     with custom_object_scope(_get_activations_scope_dict()):
#         return activations.get(identifier)

# def _get_activations_scope_dict():
#     return {
#         'exp_relu': exp_relu,
#     }
