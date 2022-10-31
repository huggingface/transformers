# Copyright 2022 AlQuraishi Laboratory
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
import contextlib
from functools import partialmethod

import numpy as np
import torch

from .tensor_utils import tensor_tree_map


def pad_feature_dict_seq(feature_dict, seqlen):
    """Pads the sequence length of a feature dict. Used for tracing."""
    # The real sequence length can't be longer than the desired one
    true_n = feature_dict["aatype"].shape[-2]
    assert true_n <= seqlen

    new_feature_dict = {}

    feat_seq_dims = {
        "aatype": -2,
        "between_segment_residues": -1,
        "residue_index": -1,
        "seq_length": -1,
        "deletion_matrix_int": -1,
        "msa": -1,
        "num_alignments": -1,
        "template_aatype": -2,
        "template_all_atom_mask": -2,
        "template_all_atom_positions": -3,
    }

    for k, v in feature_dict.items():
        if k not in feat_seq_dims:
            new_feature_dict[k] = v
            continue

        seq_dim = feat_seq_dims[k]
        padded_shape = list(v.shape)
        padded_shape[seq_dim] = seqlen
        new_value = np.zeros(padded_shape, dtype=v.dtype)
        new_value[tuple(slice(0, s) for s in v.shape)] = v
        new_feature_dict[k] = new_value

    new_feature_dict["seq_length"][0] = seqlen

    return new_feature_dict


def trace_model_(model, sample_input):
    # Grab the inputs to the final recycling iteration
    feats = tensor_tree_map(lambda t: t[..., -1], sample_input)

    # Gather some metadata
    n = feats["aatype"].shape[-1]
    msa_depth = feats["true_msa"].shape[-2]
    extra_msa_depth = feats["extra_msa"].shape[-2]
    no_templates = feats["template_aatype"].shape[-2]
    device = feats["aatype"].device

    seq_mask = feats["seq_mask"].to(device)
    pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
    extra_msa_mask = feats["extra_msa_mask"].to(device)
    template_pair_mask = torch.stack([pair_mask] * no_templates, dim=-3)

    # Create some fake representations with the correct shapes
    m = torch.rand(msa_depth + 4, n, model.globals.c_m).to(device)
    z = torch.rand(n, n, model.globals.c_z).to(device)
    t = torch.rand(no_templates, n, n, model.globals.c_t).to(device)
    a = torch.rand(extra_msa_depth, n, model.globals.c_e).to(device)
    msa_mask = torch.randint(0, 1, (msa_depth + 4, n)).to(device)

    # We need to do a dry run through the model so the chunk size tuners'
    # trial runs (which run during the first-ever model iteration) aren't
    # baked into the trace. There's no need to run the entire thing,
    # though; we just need to run one block from each transformer stack.
    evoformer_blocks = model.evoformer.blocks
    model.evoformer.blocks = evoformer_blocks[:1]

    extra_msa_blocks = model.extra_msa_stack.blocks
    model.extra_msa_stack.blocks = extra_msa_blocks[:1]

    if model.template_config.enabled:
        template_pair_stack_blocks = model.template_pair_stack.blocks
        model.template_pair_stack.blocks = template_pair_stack_blocks[:1]

    single_recycling_iter_input = tensor_tree_map(
        lambda t: t[..., :1],
        sample_input,
    )

    with torch.no_grad():
        _ = model(single_recycling_iter_input)

    model.evoformer.blocks = evoformer_blocks
    model.extra_msa_stack.blocks = extra_msa_blocks

    del evoformer_blocks, extra_msa_blocks

    if model.template_config.enabled:
        model.template_pair_stack.blocks = template_pair_stack_blocks
        del template_pair_stack_blocks

    def get_tuned_chunk_size(module):
        tuner = module.chunk_size_tuner
        chunk_size = tuner.cached_chunk_size

        # After our trial run above, this should always be set
        assert chunk_size is not None

        return chunk_size

    # Fetch the resulting chunk sizes
    evoformer_chunk_size = model.globals.chunk_size
    if model.evoformer.chunk_size_tuner is not None:
        evoformer_chunk_size = get_tuned_chunk_size(model.evoformer)

    extra_msa_chunk_size = model.globals.chunk_size
    if model.extra_msa_stack.chunk_size_tuner is not None:
        extra_msa_chunk_size = get_tuned_chunk_size(model.extra_msa_stack)

    if model.template_config.enabled:
        template_pair_stack_chunk_size = model.globals.chunk_size
        if model.template_pair_stack.chunk_size_tuner is not None:
            template_pair_stack_chunk_size = get_tuned_chunk_size(model.template_pair_stack)

    def trace_block(block, block_inputs):
        # Yes, yes, I know
        with contextlib.redirect_stderr(None):
            traced_block = torch.jit.trace(block, block_inputs)

        traced_block = torch.jit.freeze(traced_block, optimize_numerics=True)

        # It would be nice to use this, but its runtimes are extremely
        # unpredictable
        # traced_block = torch.jit.optimize_for_inference(traced_block)

        # All trace inputs need to be tensors. This wrapper takes care of that
        def traced_block_wrapper(*args, **kwargs):
            to_tensor = lambda t: torch.tensor(t) if type(t) != torch.Tensor else t
            args = [to_tensor(a) for a in args]
            kwargs = {k: to_tensor(v) for k, v in kwargs.items()}
            return traced_block(*args, **kwargs)

        return traced_block_wrapper

    def verify_arg_order(fn, arg_list):
        """Because it's difficult to specify keyword arguments of Module
        functions during tracing, we need to pass them as a tuple. As a sanity check, we manually verify their order
        here.
        """
        fn_arg_names = fn.__code__.co_varnames
        # Remove the "self" parameter
        assert fn_arg_names[0] == "self"
        fn_arg_names = fn_arg_names[1:]
        # Trim unspecified arguments
        fn_arg_names = fn_arg_names[: len(arg_list)]
        name_tups = list(zip(fn_arg_names, [n for n, _ in arg_list]))
        assert all([n1 == n2 for n1, n2 in name_tups])

    evoformer_attn_chunk_size = max(model.globals.chunk_size, evoformer_chunk_size // 4)

    # MSA row attention
    msa_att_row_arg_tuples = [
        ("m", m),
        ("z", z),
        ("mask", msa_mask),
        ("chunk_size", torch.tensor(evoformer_attn_chunk_size)),
        ("use_memory_efficient_kernel", torch.tensor(False)),
        ("use_lma", torch.tensor(model.globals.use_lma)),
    ]
    verify_arg_order(model.evoformer.blocks[0].msa_att_row.forward, msa_att_row_arg_tuples)
    msa_att_row_args = [arg for _, arg in msa_att_row_arg_tuples]
    with torch.no_grad():
        for b in model.evoformer.blocks:
            traced_block = trace_block(b.msa_att_row, msa_att_row_args)
            del b.msa_att_row
            b.msa_att_row = traced_block

    # MSA col attention
    msa_att_col_arg_tuples = [
        ("m", m),
        ("mask", msa_mask),
        ("chunk_size", torch.tensor(evoformer_chunk_size)),
        ("use_lma", torch.tensor(model.globals.use_lma)),
        ("use_flash", torch.tensor(model.globals.use_flash)),
    ]
    verify_arg_order(model.evoformer.blocks[0].msa_att_col.forward, msa_att_col_arg_tuples)
    msa_att_col_args = [arg for _, arg in msa_att_col_arg_tuples]
    with torch.no_grad():
        for b in model.evoformer.blocks:
            traced_block = trace_block(b.msa_att_col, msa_att_col_args)
            del b.msa_att_col
            b.msa_att_col = traced_block

    # OPM
    opm_arg_tuples = [
        ("m", m),
        ("mask", msa_mask.float()),
        ("chunk_size", torch.tensor(evoformer_chunk_size)),
        ("inplace_safe", torch.tensor(True)),
    ]
    verify_arg_order(model.evoformer.blocks[0].core.outer_product_mean.forward, opm_arg_tuples)
    opm_args = [arg for _, arg in opm_arg_tuples]
    with torch.no_grad():
        for b in model.evoformer.blocks:
            traced_block = trace_block(b.core.outer_product_mean, opm_args)
            del b.core.outer_product_mean
            b.core.outer_product_mean = traced_block

    # Triangular multiplicative update (out)
    tri_mul_out_arg_tuples = [
        ("z", z),
        ("mask", pair_mask.float()),
        ("inplace_safe", torch.tensor(True)),
        ("_add_with_inplace", torch.tensor(True)),
    ]
    verify_arg_order(model.evoformer.blocks[0].core.tri_mul_out.forward, tri_mul_out_arg_tuples)
    tri_mul_out_args = [arg for _, arg in tri_mul_out_arg_tuples]
    with torch.no_grad():
        for b in model.evoformer.blocks:
            traced_block = trace_block(b.core.tri_mul_out, tri_mul_out_args)
            del b.core.tri_mul_out
            b.core.tri_mul_out = traced_block

    # Triangular multiplicative update (in)
    tri_mul_in_arg_tuples = [
        ("z", z),
        ("mask", pair_mask.float()),
        ("inplace_safe", torch.tensor(True)),
        ("_add_with_inplace", torch.tensor(True)),
    ]
    verify_arg_order(model.evoformer.blocks[0].core.tri_mul_in.forward, tri_mul_in_arg_tuples)
    tri_mul_in_args = [arg for _, arg in tri_mul_in_arg_tuples]
    with torch.no_grad():
        for b in model.evoformer.blocks:
            traced_block = trace_block(b.core.tri_mul_in, tri_mul_in_args)
            del b.core.tri_mul_in
            b.core.tri_mul_in = traced_block

    # Triangular attention (start)
    tri_att_start_arg_tuples = [
        ("x", z),
        ("mask", pair_mask.float()),
        ("chunk_size", torch.tensor(evoformer_attn_chunk_size)),
        ("use_memory_efficient_kernel", torch.tensor(False)),
        ("use_lma", torch.tensor(model.globals.use_lma)),
        ("inplace_safe", torch.tensor(True)),
    ]
    verify_arg_order(model.evoformer.blocks[0].core.tri_att_start.forward, tri_att_start_arg_tuples)
    tri_att_start_args = [arg for _, arg in tri_att_start_arg_tuples]
    with torch.no_grad():
        for b in model.evoformer.blocks:
            traced_block = trace_block(b.core.tri_att_start, tri_att_start_args)
            del b.core.tri_att_start
            b.core.tri_att_start = traced_block

    # Triangular attention (end)
    tri_att_end_arg_tuples = [
        ("x", z.transpose(-2, -3)),
        ("mask", pair_mask.transpose(-1, -2).float()),
        ("chunk_size", torch.tensor(evoformer_attn_chunk_size)),
        ("use_memory_efficient_kernel", torch.tensor(False)),
        ("use_lma", torch.tensor(model.globals.use_lma)),
        ("inplace_safe", torch.tensor(True)),
    ]
    verify_arg_order(model.evoformer.blocks[0].core.tri_att_end.forward, tri_att_end_arg_tuples)
    tri_att_end_args = [arg for _, arg in tri_att_end_arg_tuples]
    with torch.no_grad():
        for b in model.evoformer.blocks:
            traced_block = trace_block(b.core.tri_att_end, tri_att_end_args)
            del b.core.tri_att_end
            b.core.tri_att_end = traced_block

    # evoformer_arg_tuples = [
    #    ("m", m),
    #    ("z", z),
    #    ("msa_mask", msa_mask),
    #    ("pair_mask", pair_mask),
    #    ("chunk_size", torch.tensor(evoformer_chunk_size)),
    #    ("use_lma", torch.tensor(model.globals.use_lma)),
    #    ("use_flash", torch.tensor(model.globals.use_flash)),
    #    ("inplace_safe", torch.tensor(1)),
    #    ("_mask_trans", torch.tensor(model.config._mask_trans)),
    #    ("_attn_chunk_size", torch.tensor(evoformer_attn_chunk_size)),
    # ]
    # verify_arg_order(model.evoformer.blocks[0].forward, evoformer_arg_tuples)
    # evoformer_args = [arg for _, arg in evoformer_arg_tuples]
    # with torch.no_grad():
    #    traced_evoformer_stack = []
    #    for b in model.evoformer.blocks:
    #        traced_block = trace_block(b, evoformer_args)
    #        traced_evoformer_stack.append(traced_block)

    # del model.evoformer.blocks
    # model.evoformer.blocks = traced_evoformer_stack

    #    with torch.no_grad():
    #        for b in model.evoformer.blocks:
    #            _ = b(*evoformer_args)
    #
    #    with torch.no_grad():
    #        for b in model.evoformer.blocks:
    #            _ = b(*evoformer_args)
    #    extra_msa_attn_chunk_size = max(
    #        model.globals.chunk_size, extra_msa_chunk_size // 4
    #    )
    #    extra_msa_arg_tuples = [
    #        ("m", a),
    #        ("z", z),
    #        ("msa_mask", extra_msa_mask),
    #        ("pair_mask", pair_mask),
    #        ("chunk_size", torch.tensor(extra_msa_chunk_size)),
    #        ("use_lma", torch.tensor(model.globals.use_lma)),
    #        ("inplace_safe", torch.tensor(1)),
    #        ("_mask_trans", torch.tensor(model.config._mask_trans)),
    #        ("_attn_chunk_size", torch.tensor(extra_msa_attn_chunk_size)),
    #    ]
    #    verify_arg_order(
    #        model.extra_msa_stack.blocks[0].forward, extra_msa_arg_tuples
    #    )
    #    extra_msa_args = [arg for _, arg in extra_msa_arg_tuples]
    #    with torch.no_grad():
    #        traced_extra_msa_stack = []
    #        for b in model.extra_msa_stack.blocks:
    #            traced_block = trace_block(b, extra_msa_args)
    #            traced_extra_msa_stack.append(traced_block)
    #
    #    del model.extra_msa_stack.blocks
    #    model.extra_msa_stack.blocks = traced_extra_msa_stack

    #    if(model.template_config.enabled):
    #        template_pair_stack_attn_chunk_size = max(
    #            model.globals.chunk_size, template_pair_stack_chunk_size // 4
    #        )
    #        template_pair_stack_arg_tuples = [
    #            ("z", t),
    #            ("mask", template_pair_mask),
    #            ("chunk_size", torch.tensor(template_pair_stack_chunk_size)),
    #            ("use_lma", torch.tensor(model.globals.use_lma)),
    #            ("inplace_safe", torch.tensor(1)),
    #            ("_mask_trans", torch.tensor(model.config._mask_trans)),
    #            ("_attn_chunk_size", torch.tensor(
    #                template_pair_stack_attn_chunk_size
    #            )),
    #        ]
    #        verify_arg_order(
    #            model.template_pair_stack.blocks[0].forward,
    #            template_pair_stack_arg_tuples
    #        )
    #        template_pair_stack_args = [
    #            arg for _, arg in template_pair_stack_arg_tuples
    #        ]
    #
    #        with torch.no_grad():
    #            traced_template_pair_stack = []
    #            for b in model.template_pair_stack.blocks:
    #                traced_block = trace_block(b, template_pair_stack_args)
    #                traced_template_pair_stack.append(traced_block)
    #
    #        del model.template_pair_stack.blocks
    #        model.template_pair_stack.blocks = traced_template_pair_stack

    # We need to do another dry run after tracing to allow the model to reach
    # top speeds. Why, I don't know.
    two_recycling_iter_input = tensor_tree_map(
        lambda t: t[..., :2],
        sample_input,
    )

    with torch.no_grad():
        _ = model(two_recycling_iter_input)
