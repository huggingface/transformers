# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq import utils


class FairseqIncrementalState(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        init_incremental_state(self)


def with_incremental_state(cls):
    cls.__bases__ = (FairseqIncrementalState,) + tuple(b for b in cls.__bases__ if b != FairseqIncrementalState)
    return cls


# In most cases we should register incremental states using @with_incremental_state decorator
# instead of calling into this explicitly in initializer.
def init_incremental_state(obj):
    obj.module_name = obj.__class__.__name__
    utils.INCREMENTAL_STATE_INSTANCE_ID[obj.module_name] = (
        utils.INCREMENTAL_STATE_INSTANCE_ID.get(obj.module_name, 0) + 1
    )
    obj._fairseq_instance_id = utils.INCREMENTAL_STATE_INSTANCE_ID[
        obj.module_name
    ]
