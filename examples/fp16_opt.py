# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Stable version of apex FP16 Optimizer"""
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from loss_scaler import DynamicLossScaler, LossScaler
from fp16util import model_grads_to_master_grads, master_params_to_model_params, clip_grad_norm

FLOAT_TYPES = (torch.FloatTensor, torch.cuda.FloatTensor)
HALF_TYPES = (torch.HalfTensor, torch.cuda.HalfTensor)

def conversion_helper(val, conversion):
    """Apply conversion to val. Recursively apply conversion if `val` is a nested tuple/list structure."""
    if not isinstance(val, (tuple, list)):
        return conversion(val)
    rtn =  [conversion_helper(v, conversion) for v in val]
    if isinstance(val, tuple):
        rtn = tuple(rtn)
    return rtn

def fp32_to_fp16(val):
    """Convert fp32 `val` to fp16"""
    def half_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, FLOAT_TYPES):
            val = val.half()
        return val
    return conversion_helper(val, half_conversion)

def fp16_to_fp32(val):
    """Convert fp16 `val` to fp32"""
    def float_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, HALF_TYPES):
            val = val.float()
        return val
    return conversion_helper(val, float_conversion)

class FP16_Module(nn.Module):
    def __init__(self, module):
        super(FP16_Module, self).__init__()
        self.add_module('module', module.half())

    def forward(self, *inputs, **kwargs):
        return fp16_to_fp32(self.module(*(fp32_to_fp16(inputs)), **kwargs))

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.module.state_dict(destination, prefix, keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)


class FP16_Optimizer(object):
    """
    :class:`FP16_Optimizer` is designed to wrap an existing PyTorch optimizer, 
    and manage static or dynamic loss scaling and master weights in a manner transparent to the user.
    For standard use, only two lines must be changed:  creating the :class:`FP16_Optimizer` instance,
    and changing the call to ``backward``.

    Example::

        model = torch.nn.Linear(D_in, D_out).cuda().half()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        # Name the FP16_Optimizer instance to replace the existing optimizer
        # (recommended but not required):
        optimizer = FP16_Optimizer(optimizer, static_loss_scale = 128.0)
        ...
        # loss.backward() becomes:
        optimizer.backward(loss)
        ...

    Example with dynamic loss scaling::

        ...
        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
                                   # optional arg to control dynamic loss scaling behavior
                                   # dynamic_loss_args={'scale_window' : 500})
                                   # Usually, dynamic_loss_args is not necessary. 

    Args:
        init_optimizer (torch.optim.optimizer):  Existing optimizer created with the parameters to optimize.  Internally, :class:`FP16_Optimizer` replaces the passed optimizer's fp16 parameters, if any, with fp32 master parameters copied from the original ones.  :class:`FP16_Optimizer` also stores references to the original fp16 parameters, and updates these fp16 parameters from the master fp32 copy at the end of each :attr:`step`.  
        static_loss_scale (float, optional, default=1.0):  Loss scale used internally to scale gradients computed by the model.  Any fp16 gradients will be copied to fp32, then downscaled before being applied to the fp32 master params, so ``static_loss_scale`` should not affect learning rate.
        dynamic_loss_scale (bool, optional, default=False):  Use dynamic loss scaling.  If True, this will override any ``static_loss_scale`` option.
        dynamic_loss_args (dict, optional, default=None):  Dict of kwargs that will be forwarded to the internal :class:`DynamicLossScaler` instance's constructor.  Keys of this dict must match kwargs accepted by :class:`DynamicLossScaler`'s constructor.  If ``dynamic_loss_args`` is unspecified, :class:`DynamicLossScaler`'s defaults will be used.
        verbose (bool, optional, default=True):  By default, FP16_Optimizer's constructor prints out the parameters and parameter groups it is ingesting, as a sanity check.  If this becomes annoying (e.g. for large models), it can be disabled by passing ``verbose=False``.  ``verbose=False`` will not disable printing when the loss scale is readjusted during dynamic loss scaling.

    ``init_optimizer`` is expected to have been constructed in the ordinary way.  
    It is recommended (although not required) that the newly constructed :class:`FP16_Optimizer` instance be 
    named to replace ``init_optimizer``, for two reasons:  
    First, it means that references to the same name
    later in the file will not have to change.  
    Second, :class:`FP16_Optimizer` reserves the right (as an implementation detail) to 
    modify ``init_optimizer``.  If you do choose a unique name for the new
    :class:`FP16_Optimizer` instance, you should only work with this new instance,
    because the preexisting optimizer might no longer behave as expected.

    ``init_optimizer`` may be any Pytorch optimizer. 
    It may contain a mixture of fp16 and fp32 parameters organized into any number of 
    ``param_groups`` with different hyperparameters.  The :class:`FP16_Optimizer` constructor will 
    ingest these ``param_groups`` and remember them. 

    Calls to ::

        loss.backward() 

    must be replaced with ::

        optimizer.backward(loss)  

    because :class:`FP16_Optimizer` requires ownership of the backward pass to implement 
    loss scaling and copies to master gradients.

    .. note::
        Loss scaling, either static or dynamic, is orthogonal to learning rate, because gradients
        are downscaled before being applied.  This means that adjusting the loss scale, or using
        dynamic loss scaling, should not require retuning the learning rate or any other 
        hyperparameters.


    **Advanced options**

    **Closures**:  :class:`FP16_Optimizer` can wrap a Pytorch optimizer that receives a closure.
    See docstring for :attr:`step`.

    **Gradient clipping**:  Use :attr:`clip_master_grads`.
    
    **Multiple losses**:  If your model accumulates gradients from multiple losses,
    this can be made more efficient by supplying ``update_master_grads=False``
    to :attr:`backward`.  See docstring for :attr:`backward`.

    **Manually adjusting loss scale**:  The current loss scale can be retrieved or set via ::

        print(optimizer.loss_scale)
        optimizer.loss_scale = new_loss_scale

    For static loss scaling, manually adjusting the loss scale over time is a reasonable
    thing to do.  During later epochs, gradients may become smaller, and a 
    higher loss scale may be required, analogous to scheduling the learning rate.  Dynamic loss
    scaling is more subtle (see :class:`DynamicLossScaler`) and in this case, manually adjusting 
    the loss scale is not recommended.

    **Multi_GPU training**:  If the wrapped ``init_optimizer`` was created from a model wrapped in
    Pytorch DistributedDataParallel or Apex DistributedDataParallel, :class:`FP16_Optimizer` 
    should still work as intended.
    """

    def __init__(self, 
                 init_optimizer, 
                 static_loss_scale=1.0, 
                 dynamic_loss_scale=False,
                 dynamic_loss_args=None,
                 verbose=False):
        if not torch.cuda.is_available:
            raise SystemError("Cannot use fp16 without CUDA.")

        self.verbose = verbose

        self.optimizer = init_optimizer
        # init_state_dict sets up an alternative way to cast per-param state tensors.
        # Stashing here in case https://github.com/pytorch/pytorch/issues/7733 makes it necessary.
        # init_state_dict = init_optimizer.state_dict()

        self.fp16_groups = []
        self.fp32_from_fp16_groups = []
        self.fp32_from_fp32_groups = []
        for i, param_group in enumerate(self.optimizer.param_groups):
            self.maybe_print("FP16_Optimizer processing param group {}:".format(i))
            fp16_params_this_group = []
            fp32_params_this_group = []
            fp32_from_fp16_params_this_group = []
            for i, param in enumerate(param_group['params']):
                if param.requires_grad:
                    if param.type() == 'torch.cuda.HalfTensor':
                        self.maybe_print("FP16_Optimizer received torch.cuda.HalfTensor with {}"
                                         .format(param.size()))
                        fp16_params_this_group.append(param)
                        master_param = param.detach().clone().float()
                        master_param.requires_grad = True
                        param_group['params'][i] = master_param
                        fp32_from_fp16_params_this_group.append(master_param)
                        # Reset existing state dict key to the new master param.
                        # We still need to recast per-param state tensors, if any, to FP32.
                        if param in self.optimizer.state:
                           self.optimizer.state[master_param] = self.optimizer.state.pop(param) 
                    elif param.type() == 'torch.cuda.FloatTensor':
                        self.maybe_print("FP16_Optimizer received torch.cuda.FloatTensor with {}"
                                         .format(param.size()))
                        fp32_params_this_group.append(param)
                        param_group['params'][i] = param
                    else:
                        raise TypeError("Wrapped parameters must be either "
                                        "torch.cuda.FloatTensor or torch.cuda.HalfTensor. "  
                                        "Received {}".format(param.type()))
            
            self.fp16_groups.append(fp16_params_this_group)
            self.fp32_from_fp16_groups.append(fp32_from_fp16_params_this_group)
            self.fp32_from_fp32_groups.append(fp32_params_this_group)

        # Leverage state_dict() and load_state_dict() to recast preexisting per-param state tensors
        self.optimizer.load_state_dict(self.optimizer.state_dict())
        # alternative way to cast per-param state tensors:
        # self.optimizer.load_state_dict(init_state_dict)

        if dynamic_loss_scale:
            self.dynamic_loss_scale = True
            if dynamic_loss_args is not None:
                self.loss_scaler = DynamicLossScaler(**dynamic_loss_args)
            else:
                self.loss_scaler = DynamicLossScaler()
        else:
            self.dynamic_loss_scale = False
            self.loss_scaler = LossScaler(static_loss_scale)

        self.overflow = False
        self.first_closure_call_this_step = True

        self.clip_grad_norm = clip_grad_norm

    def maybe_print(self, msg):
        if self.verbose:
            print(msg)
            
    def __getstate__(self):
        raise RuntimeError("FP16_Optimizer should be serialized using state_dict().")

    def __setstate__(self, state):
        raise RuntimeError("FP16_Optimizer should be deserialized using load_state_dict().")

    def zero_grad(self, set_grads_to_None=False):
        """
        Zero fp32 and fp16 parameter grads.
        """
        # In principle, only the .grad attributes of the model params need to be zeroed,
        # because gradients are copied into the FP32 master params.  However, we zero
        # all gradients owned by the optimizer, just to be safe:
        for group in self.optimizer.param_groups:
             for p in group['params']:
                 if set_grads_to_None:
                     p.grad = None
                 else:
                     if p.grad is not None:
                         p.grad.detach_()
                         p.grad.zero_()

        # Zero fp16 gradients owned by the model:
        for fp16_group in self.fp16_groups:
            for param in fp16_group:
                if set_grads_to_None:
                    param.grad = None
                else:
                    if param.grad is not None:
                        param.grad.detach_() # as in torch.optim.optimizer.zero_grad()
                        param.grad.zero_()

    def _check_overflow(self):
        params = [] 
        for group in self.fp16_groups:
            for param in group:
                params.append(param)
        for group in self.fp32_from_fp32_groups:
            for param in group:
                params.append(param)
        self.overflow = self.loss_scaler.has_overflow(params)

    def _update_scale(self, has_overflow=False):
        self.loss_scaler.update_scale(has_overflow)

    def _master_params_to_model_params(self):
        for fp16_group, fp32_from_fp16_group in zip(self.fp16_groups, self.fp32_from_fp16_groups):
            master_params_to_model_params(fp16_group, fp32_from_fp16_group)

    def _model_params_to_master_params(self):
        for fp16_group, fp32_from_fp16_group in zip(self.fp16_groups, self.fp32_from_fp16_groups):
            master_params_to_model_params(fp32_from_fp16_group, fp16_group)

    # To consider:  Integrate distributed with this wrapper by registering a hook on each variable 
    # that does the overflow check, gradient copy + downscale, and fp32 allreduce in a different stream.
    def _model_grads_to_master_grads(self):
        for fp16_group, fp32_from_fp16_group in zip(self.fp16_groups, self.fp32_from_fp16_groups):
            model_grads_to_master_grads(fp16_group, fp32_from_fp16_group)

    def _downscale_master(self):
        if self.loss_scale != 1.0:
            for group in self.optimizer.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        param.grad.data.mul_(1./self.loss_scale)

    def clip_master_grads(self, max_norm, norm_type=2):
        """
        Clips fp32 master gradients via ``torch.nn.utils.clip_grad_norm``.

        Args:
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
                infinity norm.

        Returns:
            Total norm of the current fp32 gradients (viewed as a single vector).

        .. warning::
            Returns -1 if the most recently computed fp16 gradients overflowed (that is, if ``self.overflow`` is ``True``).
        """
        if not self.overflow:
            fp32_params = []
            for param_group in self.optimizer.param_groups:
                for param in param_group['params']:
                    fp32_params.append(param)
            return self.clip_grad_norm(fp32_params, max_norm, norm_type)
        else:
            return -1

    def state_dict(self):
        """
        Returns a dict containing the current state of this :class:`FP16_Optimizer` instance.
        This dict contains attributes of :class:`FP16_Optimizer`, as well as the state_dict
        of the contained Pytorch optimizer.
        Example::

            checkpoint = {}
            checkpoint['model'] = model.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            torch.save(checkpoint, "saved.pth")
        """
        state_dict = {}
        state_dict['loss_scaler'] = self.loss_scaler
        state_dict['dynamic_loss_scale'] = self.dynamic_loss_scale
        state_dict['overflow'] = self.overflow
        state_dict['first_closure_call_this_step'] = self.first_closure_call_this_step
        state_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        state_dict['fp32_from_fp16'] = self.fp32_from_fp16_groups
        return state_dict

    def load_state_dict(self, state_dict):
        """
        Loads a state_dict created by an earlier call to state_dict(). 
        If ``fp16_optimizer_instance`` was constructed from some ``init_optimizer``, 
        whose parameters in turn came from ``model``, it is expected that the user 
        will call ``model.load_state_dict()`` before
        ``fp16_optimizer_instance.load_state_dict()`` is called.

        Example::

            model = torch.nn.Linear(D_in, D_out).cuda().half()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            optimizer = FP16_Optimizer(optimizer, static_loss_scale = 128.0)
            ...
            checkpoint = torch.load("saved.pth")
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        """
        # I think it should actually be ok to reload the optimizer before the model.
        self.loss_scaler = state_dict['loss_scaler']
        self.dynamic_loss_scale = state_dict['dynamic_loss_scale']
        self.overflow = state_dict['overflow']
        self.first_closure_call_this_step = state_dict['first_closure_call_this_step']
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        # At this point, the optimizer's references to the model's fp32 parameters are up to date.
        # The optimizer's hyperparameters and internal buffers are also up to date.  
        # However, the fp32 master copies of the model's fp16 params stored by the optimizer are still
        # out of date.  There are two options.  
        # 1:  Refresh the master params from the model's fp16 params.  
        # This requires less storage but incurs precision loss.
        # 2:  Save and restore the fp32 master copies separately.
        # We choose option 2.
        # 
        # Pytorch Optimizer.load_state_dict casts saved buffers (e.g. momentum) to the type and device 
        # of their associated parameters, because it's possible those buffers might not exist yet in 
        # the current optimizer instance.  In our case, as long as the current FP16_Optimizer has been 
        # constructed in the same way as the one whose state_dict we are loading, the same master params
        # are guaranteed to exist, so we can just copy_() from the saved master params.
        for current_group, saved_group in zip(self.fp32_from_fp16_groups, state_dict['fp32_from_fp16']):
            for current, saved in zip(current_group, saved_group):
                current.data.copy_(saved.data)

    def step(self, closure=None): # could add clip option.
        """
        If no closure is supplied, :attr:`step` should be called after 
        ``fp16_optimizer_obj.backward(loss)``.
        :attr:`step` updates the fp32 master copy of parameters using the optimizer supplied to
        :class:`FP16_Optimizer`'s constructor, then copies the updated fp32 params into the fp16 params
        originally referenced by :class:`FP16_Optimizer`'s constructor, so the user may immediately run
        another forward pass using their model.

        If a closure is supplied, :attr:`step` may be called without a prior call to 
        :attr:`backward(loss)`.
        This control flow is identical to `ordinary Pytorch optimizer use`_ with closures.
        However, the user should take care that any ``loss.backward()`` call within the closure
        has been replaced by ``fp16_optimizer_obj.backward(loss)``.

        Args:
           closure (optional):  Closure that will be supplied to the underlying optimizer originally passed to :class:`FP16_Optimizer`'s constructor.  closure should call :attr:`zero_grad()` on the :class:`FP16_Optimizer` object, compute the loss, call :attr:`backward(loss)`, and return the loss.

        Example with closure::

            # optimizer is assumed to be an FP16_Optimizer object, previously constructed from an 
            # existing pytorch optimizer.
            for input, target in dataset:
                def closure():
                    optimizer.zero_grad()
                    output = model(input)
                    loss = loss_fn(output, target)
                    # loss.backward() becomes:
                    optimizer.backward(loss)
                    return loss
                optimizer.step(closure)

        .. warning::
            Currently, calling :attr:`step` with a closure is not compatible with dynamic loss scaling.

        .. _`ordinary Pytorch optimizer use`:
            http://pytorch.org/docs/master/optim.html#optimizer-step-closure
        """

        scale = self.loss_scaler.loss_scale
        self._update_scale(self.overflow)

        if self.overflow:
            self.maybe_print("OVERFLOW! Skipping step. Attempted loss scale: {}, reducing to {}"
                .format(scale, self.loss_scale))
            return
        
        if closure is not None:
            retval = self._step_with_closure(closure)
        else:
            retval = self.optimizer.step()

        self._master_params_to_model_params()

        return retval

    def _step_with_closure(self, closure):
        def wrapped_closure():
            # helpful for debugging
            # print("Calling wrapped_closure, first_closure_call_this_step = {}"
            #       .format(self.first_closure_call_this_step))
            if self.first_closure_call_this_step:
                # We expect that the fp16 params are initially fresh on entering self.step(),
                # so _master_params_to_model_params() is unnecessary the first time wrapped_closure()
                # is called within self.optimizer.step().
                self.first_closure_call_this_step = False
            else:
                # If self.optimizer.step() internally calls wrapped_closure more than once,
                # it may update the fp32 params after each call.  However, self.optimizer 
                # doesn't know about the fp16 params at all.  If the fp32 params get updated,
                # we can't rely on self.optimizer to refresh the fp16 params.  We need
                # to handle that manually:
                self._master_params_to_model_params()
            # Our API expects the user to give us ownership of the backward() call by
            # replacing all calls to loss.backward() with optimizer.backward(loss).
            # This requirement holds whether or not the call to backward() is made within a closure.
            # If the user is properly calling optimizer.backward(loss) within "closure," 
            # calling closure() here will give the fp32 master params fresh gradients
            # for the optimizer to play with, so all wrapped_closure needs to do is call 
            # closure() and return the loss.
            temp_loss = closure() 
            while(self.overflow):
                scale = self.loss_scaler.loss_scale
                self._update_scale(self.overflow)
                self.maybe_print("OVERFLOW within closure! Skipping step. Attempted loss scale: {}, "
                      "reducing to {}".format(scale, self.loss_scale))
                temp_loss = closure()
            return temp_loss

        retval = self.optimizer.step(wrapped_closure)

        self.first_closure_call_this_step = True

        return retval

    def backward(self, loss, update_master_grads=True, retain_graph=False):
        """ 
        :attr:`backward` performs the following conceptual steps:

        1. fp32_loss = loss.float() (see first Note below)
        2. scaled_loss = fp32_loss*loss_scale
        3. scaled_loss.backward(), which accumulates scaled gradients into the ``.grad`` attributes of the model's leaves (which may be fp16, fp32, or a mixture, depending how your model was defined).
        4. fp16 grads are then copied to the master params' ``.grad`` attributes (see second Note), which are guaranteed to be fp32.
        5. Finally, master grads are divided by loss_scale.

        In this way, after :attr:`backward`, the master params have fresh gradients,
        and :attr:`step` may be called.

        .. note::
            :attr:`backward` internally converts the loss to fp32 before applying the loss scale.
            This provides some additional safety against overflow if the user has supplied an 
            fp16 loss value.  
            However, for maximum overflow safety, the user should
            compute the loss criterion (MSE, cross entropy, etc) in fp32 before supplying it to 
            :attr:`backward`.

        .. warning::
            The gradients found in a model's leaves after the call to 
            :attr:`backward` should not be regarded as valid in general, 
            because it's possible 
            they have been scaled (and in the case of dynamic loss scaling, 
            the scale factor may change over time).  
            If the user wants to inspect gradients after a call to :attr:`backward`,  
            only the master gradients should be regarded as valid.  These can be retrieved via
            :attr:`inspect_master_grad_data()`.

        Args:
            loss:  The loss output by the user's model.  loss may be either float or half (but see first Note above).
            update_master_grads (bool, optional, default=True):  Option to copy fp16 grads to fp32 grads on this call.  By setting this to False, the user can delay the copy, which is useful to eliminate redundant fp16->fp32 grad copies if :attr:`backward` is being called on multiple losses in one iteration.  If set to False, the user becomes responsible for calling :attr:`update_master_grads` before calling :attr:`step`.
            retain_graph (bool, optional, default=False):  Forwards the usual ``retain_graph=True`` option to the internal call to ``loss.backward``.  If ``retain_graph`` is being used to accumulate gradient values from multiple backward passes before calling ``optimizer.step``, passing ``update_master_grads=False`` is also recommended (see Example below).

        Example::

            # Ordinary operation:
            optimizer.backward(loss)

            # Naive operation with multiple losses (technically valid, but less efficient):
            # fp32 grads will be correct after the second call,  but 
            # the first call incurs an unnecessary fp16->fp32 grad copy.
            optimizer.backward(loss1)
            optimizer.backward(loss2)

            # More efficient way to handle multiple losses:
            # The fp16->fp32 grad copy is delayed until fp16 grads from all 
            # losses have been accumulated.
            optimizer.backward(loss1, update_master_grads=False)
            optimizer.backward(loss2, update_master_grads=False)
            optimizer.update_master_grads()
        """ 
        # To consider:  try multiple backward passes using retain_grad=True to find 
        # a loss scale that works.  After you find a loss scale that works, do a final dummy
        # backward pass with retain_graph=False to tear down the graph.  Doing this would avoid 
        # discarding the iteration,  but probably wouldn't improve overall efficiency.  
        self.loss_scaler.backward(loss.float(), retain_graph=retain_graph)
        if update_master_grads:
            self.update_master_grads()

    def update_master_grads(self):
        """
        Copy the ``.grad`` attribute from stored references to fp16 parameters to 
        the ``.grad`` attribute of the fp32 master parameters that are directly 
        updated by the optimizer.  :attr:`update_master_grads` only needs to be called if
        ``fp16_optimizer_obj.backward`` was called with ``update_master_grads=False``.
        """
        if self.dynamic_loss_scale:
            self._check_overflow()
            if self.overflow: return
        self._model_grads_to_master_grads()
        self._downscale_master()

    def inspect_master_grad_data(self):
        """
        When running with :class:`FP16_Optimizer`, 
        ``.grad`` attributes of a model's fp16 leaves should not be
        regarded as truthful, because they might be scaled.  
        After a call to :attr:`fp16_optimizer_obj.backward(loss)`, if no overflow was encountered,
        the fp32 master params' ``.grad``
        attributes will contain valid gradients properly divided by the loss scale.  However, 
        because :class:`FP16_Optimizer` flattens some parameters, accessing them may be 
        nonintuitive.  :attr:`inspect_master_grad_data`
        allows those gradients to be viewed with shapes corresponding to their associated model leaves.

        Returns:
            List of lists (one list for each parameter group).  The list for each parameter group
            is a list of the ``.grad.data`` attributes of the fp32 master params belonging to that group.                 
        """
        if self.overflow:
            print("Warning:  calling FP16_Optimizer.inspect_master_grad_data while in an overflow state.  "
                  "Gradients are currently invalid (may be inf, nan, or stale).  Returning None.")
            return None
        else:
            # The optimizer owns only references to master params.
            master_grads_data = []
            for param_group in self.optimizer.param_groups:
                master_grads_this_group = []
                for param in param_group['params']:
                    if param.grad is not None:
                        master_grads_this_group.append(param.grad.data)
                    else:
                        master_grads_this_group.append(None)
                master_grads_data.append(master_grads_this_group)
            return master_grads_data


    # Promote loss scale so it can be retrieved or set via "fp16_optimizer_instance.loss_scale"
    def _get_loss_scale(self):
        return self.loss_scaler.loss_scale

    def _set_loss_scale(self, value):
        self.loss_scaler.cur_scale = value

    loss_scale = property(_get_loss_scale, _set_loss_scale)

    # Promote state so it can be retrieved or set via "fp16_optimizer_instance.state"
    def _get_state(self):
        return self.optimizer.state

    def _set_state(self, value):
        self.optimizer.state = value

    state = property(_get_state, _set_state)

    # Promote param_groups so it can be retrieved or set via "fp16_optimizer_instance.param_groups"
    # (for example, to adjust the learning rate)
    def _get_param_groups(self):
        return self.optimizer.param_groups

    def _set_param_groups(self, value):
        self.optimizer.param_groups = value

    param_groups = property(_get_param_groups, _set_param_groups)

