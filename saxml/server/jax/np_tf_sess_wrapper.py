# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""np_tf_sess_wrapper with more efficient PyTree."""

import jax
from jax import tree_util
import numpy as np
from saxml.server.tf import np_tf_sess_wrapper

# nested -> sequence.
np_tf_sess_wrapper.tree_flatten = tree_util.tree_flatten
# tree, sequence -> nested.
np_tf_sess_wrapper.tree_unflatten = tree_util.tree_unflatten
# function, tree -> tree.
np_tf_sess_wrapper.tree_map = tree_util.tree_map


# wrap_tf_session = np_tf_sess_wrapper.wrap_tf_session
# Replace wrap_tf_session with an eager equivalent to circumvent a segfault.
def wrap_tf_session(fun, fix_non_batch_dims=True):
  del fix_non_batch_dims

  def wrapped_fun(*args):
    return jax.tree_map(np.asarray, fun(*args))

  return wrapped_fun


wrap_tf_session_class_member = np_tf_sess_wrapper.wrap_tf_session_class_member
