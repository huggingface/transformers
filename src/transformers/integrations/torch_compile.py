# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Workarounds for ``torch.compile`` quirks we hit in production paths.

``install_compile_patches`` is the single entry point; each helper patches
one upstream torch internal so model code paths compile cleanly. Removable
once the host torch ships the referenced ``pytorch#`` fixes.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def mark_repeated_decoder_layers_as_compile_regions(model: nn.Module) -> None:
    """Wrap the outermost repeated-block class's ``forward`` with
    ``torch.compiler.nested_compile_region`` so dynamo compiles one instance's
    body and stamps it across all N siblings — collapses the dominant codegen
    cost for the layer stack. Idempotent — the class is tagged so re-entry
    no-ops. Structural traversal mirrors ``accelerate.utils.compile_regions``.
    """

    def _is_repeated_block(m: nn.Module) -> bool:
        return isinstance(m, nn.ModuleList) and len(m) > 1 and all(isinstance(child, m[0].__class__) for child in m)

    def _find_repeated_block(module: nn.Module) -> nn.ModuleList | None:
        for child in module.children():
            if _is_repeated_block(child):
                return child  # type: ignore[return-value]
            found = _find_repeated_block(child)
            if found is not None:
                return found
        return None

    stack = _find_repeated_block(model)
    if stack is None:
        return
    layer_cls = type(stack[0])
    if getattr(layer_cls, "_forward_is_nested_compile_region", False):
        return
    layer_cls.forward = torch.compiler.nested_compile_region(layer_cls.forward)
    layer_cls._forward_is_nested_compile_region = True


def install_compile_patches() -> None:
    """Install all three torch-internal patches required for clean
    ``torch.compile(fullgraph=True)`` compilation of models that use
    ``@torch.compiler.nested_compile_region`` with custom mutating ops.
    Idempotent — patches once per process.
    """
    if getattr(install_compile_patches, "_done", False):
        return
    _patch_invoke_subgraph_placeholder()
    _patch_is_compiling_for_session()
    _patch_decompose_auto_functionalized()
    install_compile_patches._done = True


def _patch_invoke_subgraph_placeholder() -> None:
    """Take the recursive-compile branch only for ``torch.export``, not for
    ``torch.compile``. Torch 2.12 ships ``invoke_subgraph_placeholder`` gated
    on ``is_compiling()`` — under ``torch.compile`` that branch fires from
    AOTAutograd's re-entry and bypasses ``FunctionalTensorMode``, causing
    mutating custom ops to trip the C++ functionalize fallback's alias check.
    Upstream changed the gate to ``is_exporting()`` (around pytorch#184614).
    """
    import importlib

    isg = importlib.import_module("torch._higher_order_ops.invoke_subgraph")

    def _patched(func, *args, **kwargs):
        if torch.compiler.is_dynamo_compiling():
            raise RuntimeError("invoke_subgraph should not be called directly in Dynamo")
        if torch.compiler.is_exporting():
            from torch._higher_order_ops.utils import _hop_compile_and_call

            def _wrap(func, args, kwargs):
                return _patched(func, *args, **kwargs)

            return _hop_compile_and_call(_wrap, (func, args, kwargs))
        return func(*args, **kwargs)

    isg.invoke_subgraph_placeholder = _patched


def _patch_is_compiling_for_session() -> None:
    """Backport pytorch#184614 (commit ``80dd86c8cb9``, post-2.12): hold
    ``torch.compiler.is_compiling()`` True for the whole compile session, not
    just the Dynamo bytecode intercept. Without this, callers run inside the
    session but outside the trace (AOTAutograd ``__torch_dispatch__``,
    ``make_fx`` re-entries, custom backends) see the flag as False and pick
    the eager branch in ``invoke_subgraph_placeholder``, bypassing
    ``FunctionalTensorMode``.
    """
    import contextlib

    import torch.compiler as _torch_compiler
    from torch._dynamo import convert_frame as _convert_frame

    if not hasattr(_torch_compiler, "_compile_session_context"):

        @contextlib.contextmanager
        def _compile_session_context():
            old = getattr(_torch_compiler, "_is_compiling_flag", False)
            try:
                _torch_compiler._is_compiling_flag = True
                yield
            finally:
                _torch_compiler._is_compiling_flag = old

        _torch_compiler._compile_session_context = _compile_session_context

    _stock_compile = _convert_frame._compile

    def _patched(*args, **kwargs):
        if kwargs.get("export", False):
            return _stock_compile(*args, **kwargs)
        with _torch_compiler._compile_session_context():
            return _stock_compile(*args, **kwargs)

    _convert_frame._compile = _patched


def _patch_decompose_auto_functionalized() -> None:
    """Pre-process ``auto_functionalized_v2(invoke_subgraph, …)`` FX nodes
    before ``torch._inductor.fx_passes.post_grad.decompose_auto_functionalized``
    runs. The stock pass uses ``Match.replace_by_example`` which silently
    fails to produce a replacement when the inner op is an HOP
    (pytorch#180949). We rewrite those nodes to direct ``invoke_subgraph``
    calls — eager already mutated the static-address buffers, so no
    clone-before-call is needed.
    """
    import operator

    from torch._inductor.fx_passes import post_grad as _post_grad

    invoke_subgraph_op = torch.ops.higher_order.invoke_subgraph
    auto_func_v1 = torch.ops.higher_order.auto_functionalized
    auto_func_v2 = torch.ops.higher_order.auto_functionalized_v2

    _stock_decompose = _post_grad.decompose_auto_functionalized

    def _inline_invoke_subgraph_wrapper(graph: torch.fx.graph.Graph) -> None:
        """Replace ``auto_functionalized_v2(invoke_subgraph, …)`` nodes with a
        direct ``invoke_subgraph(subgraph, identifier, *operands)`` call.

        Operand reconstruction:
        - ``argN`` kwargs that are present give the operand at index N directly.
        - For each ``_argN_base_index = K`` kwarg, operand N is ``_all_bases[K]``
          (the un-aliased base — ``write_view_information_to_args`` only emits
          the ``_base_index`` key when ``get_base()`` returned None, meaning the
          arg IS the base, not a view of it; if it were a view we'd see
          ``_size`` / ``_stride`` / ``_slice_*`` kwargs too, which we don't see
          in any traced HOP case here).
        - ``auto_functionalized_v2[0 .. N_outs-1]`` → ``getitem(new_call, i)``
        - ``auto_functionalized_v2[N_outs ..]`` → ``_all_bases[i - N_outs]``
          (mutation already happened in eager via ``mark_static_address``).
        """
        for target in (auto_func_v1, auto_func_v2):
            for node in list(graph.find_nodes(op="call_function", target=target)):
                if not node.args or node.args[0] is not invoke_subgraph_op:
                    continue
                kw = node.kwargs
                subgraph = kw.get("subgraph")
                identifier = kw.get("identifier")
                if subgraph is None:  # malformed; let the stock pass handle (will assert)
                    continue
                all_bases = list(kw.get("_all_bases", []))

                direct = {int(k[3:]): v for k, v in kw.items() if k.startswith("arg") and k[3:].isdigit()}
                base_map: dict[int, Any] = {}
                for k, v in kw.items():
                    if not (k.startswith("_arg") and k.endswith("_base_index")):
                        continue
                    n = int(k[len("_arg") : -len("_base_index")])
                    if isinstance(v, int):
                        base_map[n] = all_bases[v]
                if not direct and not base_map:
                    continue  # no operands; not our shape
                max_idx = max((*direct, *base_map))
                ordered: list[Any] | None = []
                for i in range(max_idx + 1):
                    if i in direct:
                        ordered.append(direct[i])
                    elif i in base_map:
                        ordered.append(base_map[i])
                    else:
                        # Hole in the operand list — not safe to reconstruct.
                        ordered = None
                        break
                if ordered is None:
                    continue

                with graph.inserting_before(node):
                    new_call = graph.call_function(invoke_subgraph_op, args=(subgraph, identifier, *ordered))
                    # Drop ``eager_input_vals`` when copying meta: it was set for
                    # the ``auto_functionalized_v2`` signature ``(mutable_op,
                    # **kwargs)``, not for ``invoke_subgraph(subgraph,
                    # identifier, *operands)``. ``InvokeSubgraph.create`` slices
                    # ``eager_input_vals[0][2:]`` to derive ``fake_operands``,
                    # so passing the wrong-layout meta yields a too-short
                    # ``fake_operands`` and an ``IndexError`` in the lowering
                    # loop. Without it, ``create`` falls back to reading operand
                    # vals from the FX args directly.
                    new_call.meta.update({k: v for k, v in node.meta.items() if k != "eager_input_vals"})

                # ``auto_functionalized_v2`` output layout:
                #   ``[0 .. N_outs-1]`` = subgraph outputs
                #   ``[N_outs .. N_outs + N_bases - 1]`` = (would-be-cloned) bases
                # ``invoke_subgraph`` returns the subgraph output tuple/list as
                # a single value, so each output access needs a getitem. For
                # the base accesses we point straight at the original buffer
                # (no clone — eager mutation already wrote to
                # ``mark_static_address`` storage). N_outs = max getitem index
                # + 1 - N_bases.
                getitem_users = [u for u in list(node.users) if u.target is operator.getitem]
                n_bases = len(all_bases)
                max_user_idx = max((u.args[1] for u in getitem_users), default=-1)
                n_outs = max_user_idx + 1 - n_bases if getitem_users else 0
                if n_outs <= 0:
                    n_outs = 1  # fallback for subgraphs that return a single value
                for user in getitem_users:
                    idx = user.args[1]
                    if idx < n_outs:
                        with graph.inserting_before(user):
                            replacement = graph.call_function(operator.getitem, args=(new_call, idx))
                        # Carry over the fake-tensor val: downstream inductor
                        # lowerings (e.g. ``InvokeSubgraph.create``) read it
                        # via ``x.meta["val"]`` when ``eager_input_vals`` is
                        # missing.
                        if "val" in user.meta:
                            replacement.meta["val"] = user.meta["val"]
                        user.replace_all_uses_with(replacement)
                    elif idx - n_outs < n_bases:
                        user.replace_all_uses_with(all_bases[idx - n_outs])
                    graph.erase_node(user)
                graph.erase_node(node)
        graph.lint()

    def _patched(graph):
        _inline_invoke_subgraph_wrapper(graph)
        _stock_decompose(graph)

    _post_grad.decompose_auto_functionalized = _patched
