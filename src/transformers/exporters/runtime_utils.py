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
"""Run exported generative models by orchestrating their component graphs through `GenerationMixin.generate`.

`HfExporter.export_for_generation` produces one graph per component; this module plugs the graphs back
together and drives the generation loop from artifacts + configs alone — no model instance, no checkpoint
weights. The model config and the generation config **used at export** are the contract: save them with
the artifacts and hand them back to `ExportedGenerator.from_runners` — the generation config declares the
cache the graphs were traced against (growing `DynamicCache` vs fixed-size `StaticCache` + length), so
nothing is introspected from the graphs. Two public pieces:

- `ModelRunner` — wraps a backend's runtime handle (an `onnxruntime.InferenceSession`, a `torch.export`
  unlifted `module()`, a loaded ExecuTorch program) so it forwards like the module it was exported from:
  `runner(**kwargs) -> {name: tensor}`, torch tensors in and out. One subclass per backend
  (`OnnxModelRunner`, `DynamoModelRunner`, `ExecutorchModelRunner`), each hiding how its backend carries
  tensors (ORT's numpy boundary, executorch's positional buffers) and the KV-cache (flat `input.<name>` /
  `output.<name>` tensors vs the `Cache` pytree).
- `ExportedGenerator` — a `GenerationMixin` over the component runners: a `decode` graph alone is
  decoder-only text generation; add a `text_embed` graph and `Modality` entries (one features graph per
  image / video / audio input) and prefill scatters each modality's features into `inputs_embeds` at its
  placeholder positions, injecting the grid-derived precompute the graphs expect from the config alone.
  `ExportedGenerator.from_runners` assembles either kind from `{component_name: runner}` + the saved
  configs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch.utils._pytree import tree_flatten, tree_leaves, tree_unflatten

from ..generation import GenerationConfig, GenerationMixin
from ..modeling_outputs import BaseModelOutput, CausalLMOutputWithPast


_ORT_TO_TORCH_DTYPE = {
    "tensor(float)": torch.float32,
    "tensor(float16)": torch.float16,
    "tensor(bfloat16)": torch.bfloat16,
    "tensor(double)": torch.float64,
}


class ModelRunner(ABC):
    """Wraps one exported artifact so it forwards like the module it was exported from:
    `runner(**kwargs) -> {name: tensor}`, torch tensors in and out.

    `kwargs` are the exported forward's kwargs — including `past_key_values` as a `Cache` object for a
    decode graph; each backend adapts it to however its graph carries the cache (flattened `input.<name>`
    tensors, positional buffers, or the pytree itself). The returned dict maps output leaf names
    (`logits`, `past_key_values.…`) to tensors, so callers read outputs the same way for every backend.

    `input_names` lists the graph's declared inputs — a caller feeds only what the graph takes;
    `text_input` and `mask_inputs` are derived from it so a generation loop can key its feed. `device` /
    `dtype` are where the backend lands its output tensors — the generator reads them off the decode
    runner, so nobody has to pass them in.
    """

    input_names: tuple[str, ...] = ()
    device: torch.device | str = "cpu"
    dtype: torch.dtype = torch.float32

    @property
    def text_input(self) -> str:
        """The graph's text input: `"decoder_input_ids"` (encoder-decoder decode), `"inputs_embeds"`
        (multi-modal decode) or `"input_ids"`."""
        for name in ("decoder_input_ids", "inputs_embeds"):
            if name in self.input_names:
                return name
        return "input_ids"

    @property
    def mask_inputs(self) -> tuple[str, ...]:
        """The graph's attention-mask input name(s) — several for mixed full/sliding attention."""
        return tuple(n for n in self.input_names if n == "attention_mask" or n.startswith("attention_mask."))

    @abstractmethod
    def __call__(self, **kwargs) -> dict[str, torch.Tensor]:
        """Run the graph on `kwargs`; return its outputs as `{leaf_name: tensor}`."""


def _cache_tensors(past_key_values) -> list[torch.Tensor]:
    """The cache's tensor leaves, in the pytree order the exporter named them."""
    return [t for t in tree_leaves(past_key_values) if isinstance(t, torch.Tensor)]


def _advance_cache(past_key_values, outputs: dict[str, torch.Tensor], num_new_tokens: int):
    """Advance the cache with a decode step's outputs (the `past_key_values.…` entries, in cache-leaf
    order). A fixed-size cache keeps its shapes, so `copy_` in place — preserving the cache object and its
    non-tensor state; a graph that already mutated the cache in place returns the same tensors, making the
    copy a no-op. A growing `DynamicCache` returns longer tensors (its seq axis grew), so rebuild the cache
    from the grown leaves through the registered cache pytree. The tensors themselves tell the two apart.

    Sliding layers additionally keep their running length in a plain python int — `cumulative_length_int`
    on static sliding layers, `cumulative_length` itself on growing ones. It's not a pytree tensor, so the
    decode graph never updates it (the static graph bakes it as a trace-time constant; the growing-cache
    rebuild resurrects the pre-step value from the pytree context). Advance it by the tokens just
    processed, the way the eager `update` does — deliberately NOT read from the static layer's
    `cumulative_length` tensor: `int(tensor)` is a device→host sync (which also blocks CUDA-graph
    capture), and once a sliding layer is full the tensor stops advancing while the int keeps counting."""
    cache_updates = [(name, value) for name, value in outputs.items() if name.startswith("past_key_values")]
    if cache_updates:
        cache_leaves = _cache_tensors(past_key_values)
        # Align updates to cache leaves. ExecuTorch names its cache inputs by flat leaf index
        # (`past_key_values_<N>`) and may prune placeholders its lowering left unused, so index by the
        # suffix and keep the old leaf where no update came back; other backends' dotted names arrive in
        # leaf order. The `.pte` runtime also returns rank-0 updates as python scalars — re-wrap them.
        updated = list(cache_leaves)
        for position, (name, new) in enumerate(cache_updates):
            suffix = name.rsplit("_", 1)[-1]
            index = int(suffix) if suffix.isdigit() else position
            if not isinstance(new, torch.Tensor):
                new = torch.tensor(new, dtype=cache_leaves[index].dtype, device=cache_leaves[index].device)
            updated[index] = new
        if any(old.shape != new.shape for old, new in zip(cache_leaves, updated)):
            _, spec = tree_flatten(past_key_values)
            past_key_values = tree_unflatten(updated, spec)
        else:
            for old, new in zip(cache_leaves, updated):
                if old is not new:
                    old.copy_(new)
    layers = (
        past_key_values.self_attention_cache.layers
        if hasattr(past_key_values, "self_attention_cache")
        else past_key_values.layers
    )
    for layer in layers:
        if hasattr(layer, "cumulative_length_int"):
            layer.cumulative_length_int += num_new_tokens
        elif isinstance(getattr(layer, "cumulative_length", None), int):
            layer.cumulative_length += num_new_tokens
    # An `EncoderDecoderCache` also keeps `is_updated` python flags (pytree context, so the graph never
    # flips them and the growing rebuild resurrects the pre-step values): every decoder step leaves the
    # cross cache written — the prefill graph writes it, decode graphs read it.
    if getattr(past_key_values, "is_updated", None):
        past_key_values.is_updated = dict.fromkeys(past_key_values.is_updated, True)
    return past_key_values


class OnnxModelRunner(ModelRunner):
    """`ModelRunner` backed by an `onnxruntime.InferenceSession`. Torch tensors in, torch tensors out —
    the numpy boundary ORT requires is confined here. The KV-cache rides as matched `input.<name>` /
    `output.<name>` graph inputs/outputs, so a `past_key_values` kwarg is flattened into the feed and the
    `output.` prefix stripped from the results (back to plain leaf names)."""

    def __init__(self, session):
        self._session = session
        self._output_names = [o.name for o in session.get_outputs()]
        self.input_names = tuple(i.name for i in session.get_inputs())
        self._cache_names = [n for n in self.input_names if n.startswith("input.")]
        # Where the loop's tensors live / at what precision: the session's execution provider and the
        # graph's float (logits) output type. ORT still hands back host numpy, so `__call__` bridges —
        # feeds to CPU, outputs back to this device.
        self.device = "cuda" if any("CUDA" in p for p in session.get_providers()) else "cpu"
        logits = next((o for o in session.get_outputs() if o.name == "logits"), None)
        self.dtype = _ORT_TO_TORCH_DTYPE.get(logits.type, torch.float32) if logits is not None else torch.float32

    def __call__(self, **kwargs) -> dict[str, torch.Tensor]:
        from .utils import get_leaf_tensors

        cache = kwargs.pop("past_key_values", None)
        if cache is not None:
            kwargs.update({name: t.detach() for name, t in zip(self._cache_names, _cache_tensors(cache))})
        # Non-tensor kwargs are pytrees (`encoder_outputs`, …) — the graph declares their leaves by dotted
        # path, the exporter's own naming.
        for name in [n for n, v in kwargs.items() if not isinstance(v, torch.Tensor)]:
            kwargs.update({f"{name}.{leaf}": t for leaf, t in get_leaf_tensors(kwargs.pop(name)).items()})
        feed = {name: tensor.detach().cpu().numpy() for name, tensor in kwargs.items()}
        outputs = self._session.run(None, feed)
        return {
            name.removeprefix("output."): torch.from_numpy(value).to(self.device)
            for name, value in zip(self._output_names, outputs)
        }


class DynamoModelRunner(ModelRunner):
    """`ModelRunner` backed by a `torch.export` unlifted module (`ExportedProgram.module()`) — the
    runnable, like ORT's session. Kwargs pass straight through (the KV-cache stays a `Cache` pytree the
    graph consumes natively); the output object is flattened to its named tensor leaves."""

    def __init__(self, module):
        self._module = module
        # The module requires the exact kwarg set it was traced with (including baked scalars like
        # `max_seqlen` that aren't graph placeholders) — its input pytree spec carries those kwarg names.
        self.input_names = tuple(module._in_spec.child(1).context)
        # Outputs land wherever the exported weights live.
        weight = next(self._module.parameters(), None)
        if weight is not None:
            self.device, self.dtype = weight.device, weight.dtype

    def __call__(self, **kwargs) -> dict[str, torch.Tensor]:
        from .utils import get_leaf_tensors

        # The module rejects kwargs it wasn't traced with — e.g. a prefill graph whose capture predates the
        # cache (models that create it inside the first forward); its cache still rides out on the outputs.
        if "past_key_values" not in self.input_names:
            kwargs.pop("past_key_values", None)
        output = self._module(**kwargs)
        if isinstance(output, torch.Tensor):
            return {"output": output}
        return get_leaf_tensors(output)


class ExecutorchModelRunner(ModelRunner):
    """`ModelRunner` backed by a loaded ExecuTorch runtime program
    (`Runtime.get().load_program(...)`). Unlike ONNX, a `.pte` carries no `input.`/`output.` convention:
    inputs are **positional** (in the source graph's flat order) and `execute` returns the lowering's
    mutated-input copies first, the model's own outputs last. A loaded method's metadata carries only counts
    and tensor shapes, so the exporter bakes the input names and the user-output count into the `.pte`
    itself as constant methods (`_signature_constant_methods`) — the program is self-describing, exactly
    like an ONNX session or a `torch.export` module, and this runner needs nothing else. Cache inputs are
    the `past_key_values*` ones, each named by its flat leaf index; the model's outputs are
    `[logits, *cache_updates]`, in cache-input order.

    Multi-token decode works: XNNPACK needs a *bounded* dynamic sequence dim, which `_fix_range_constraints`
    already supplies (it caps unbounded `Dim.AUTO` extents), so one graph serves prefill and decode.
    """

    def __init__(self, program):
        self._method = program.load_method("forward")
        self.input_names = tuple(program.load_method("input_names").execute(()))
        num_user_outputs = program.load_method("num_user_outputs").execute(())[0]
        total_outputs = self._method.metadata.num_outputs()
        self._user_output_indices = range(total_outputs - num_user_outputs, total_outputs)
        self._cache_names = [n for n in self.input_names if n.startswith("past_key_values")]

    def __call__(self, **kwargs) -> dict[str, torch.Tensor]:
        from .utils import get_leaf_tensors

        cache = kwargs.pop("past_key_values", None)
        if cache is not None:
            # Cache inputs are named by flat leaf index (`past_key_values_<N>`) — index rather than zip,
            # since the lowering may have pruned leaves it left unused (sliding caches' scalars).
            leaves = _cache_tensors(cache)
            kwargs.update({name: leaves[int(name.rsplit("_", 1)[-1])] for name in self._cache_names})
        # Remaining non-tensor kwargs are pytrees (`encoder_outputs`, mask dicts) — the graph names their
        # leaves by underscore-joined path (`encoder_outputs_last_hidden_state`).
        for name in [n for n, v in kwargs.items() if not isinstance(v, torch.Tensor)]:
            value = kwargs.pop(name)
            kwargs.update({f"{name}_{leaf.replace('.', '_')}": t for leaf, t in get_leaf_tensors(value).items()})
        outputs = self._method.execute(tuple(kwargs[name].contiguous() for name in self.input_names))
        outputs = [outputs[i] for i in self._user_output_indices]
        if cache is not None:
            return {"logits": outputs[0], **dict(zip(self._cache_names, outputs[1:]))}
        return {f"output.{i}": value for i, value in enumerate(outputs)}


@dataclass
class Modality:
    """Routes one input modality (image / video / audio) of an `ExportedGenerator`:

    - `token_id`: the placeholder id in `input_ids` its features scatter into (`config.image_token_id`, …).
    - `runner`: the exported `get_<modality>_features` graph.
    - `input_keys`: the generate kwargs that belong to it (e.g. `("pixel_values", "image_grid_thw")`); the
      first is the presence key — the modality runs only when it's passed.
    """

    token_id: int
    runner: ModelRunner
    input_keys: tuple


class _ExportedEncoder:
    """`get_encoder()` stand-in over the exported encoder graph. `generate` filters its kwargs by
    `forward`'s signature (a wildcard here, so nothing is dropped), always adds the `output_*` /
    `return_dict` flags, and expects a `ModelOutput` back — the wrapper feeds the graph only what it
    declares and wraps its features as `BaseModelOutput.last_hidden_state`, the form the decoder graphs
    were traced with."""

    def __init__(self, runner: ModelRunner):
        self._runner = runner

    def forward(self, **kwargs):
        feed = {k: v for k, v in kwargs.items() if k in set(self._runner.input_names)}
        return BaseModelOutput(last_hidden_state=next(iter(self._runner(**feed).values())))

    def __call__(self, **kwargs):
        return self.forward(**kwargs)


class ExportedGenerator(GenerationMixin):
    """Drive exported component graphs through `generate`, from artifacts + configs alone.

    A `decode` runner alone is decoder-only text generation. Pass `text_embed` (the `input_ids -> inputs_embeds`
    graph) and `modalities` for a multi-modal model: prefill computes each modality's features and scatters
    them into `inputs_embeds` at its placeholder positions; decode steps are text-only. Pass `prefill=` to
    drive a fixed query=1 `decode` graph from a separate dynamic prefill graph (the shape CUDA-graph capture
    / io-binding want); when omitted, `decode` serves both (it must then be a multi-token graph, i.e.
    exported with `multi_token_decode=True`).

    `generation_config` must be the one the model was **exported with**: it declares the cache the decode
    graph was traced against (no `cache_implementation` → growing `DynamicCache`; `"static"` → fixed-size
    `StaticCache`, whose `max_cache_len` a static-cache export should pin explicitly).

    Example:
        programs = OnnxExporter().export_for_generation(model, inputs,
                                                        OnnxConfig(dynamic=True, external_data=False),
                                                        generation_config=generation_config,
                                                        multi_token_decode=True)
        session = ort.InferenceSession(programs["decode"].model_proto.SerializeToString())
        runtime = ExportedGenerator(model.config, generation_config, OnnxModelRunner(session))
        # Called like a normal model — the runtime builds the cache the exported graph needs.
        ids = runtime.generate(input_ids=prompt, max_new_tokens=32)
    """

    base_model_prefix = ""
    main_input_name = "input_ids"

    _supports_cache_class = True

    def __init__(
        self,
        config,
        generation_config,
        decode: ModelRunner,
        *,
        prefill: ModelRunner | None = None,
        encoder: ModelRunner | None = None,
        text_embed: ModelRunner | None = None,
        modalities: list[Modality] = (),
    ):
        self.config = config
        self.generation_config = generation_config
        self._decode_runner = decode
        # (`_prefill`/`_decode` without the suffix would shadow `GenerationMixin` methods `generate` calls.)
        self._prefill_runner = prefill if prefill is not None else decode
        self._encoder = _ExportedEncoder(encoder) if encoder is not None else None
        self._text_embed = text_embed
        self._modalities = list(modalities)
        self._modality_keys = {key for modality in self._modalities for key in modality.input_keys}
        # Everything the component graphs declare: `generate` kwargs matching these are fed straight through
        # (a model may take extra per-step tensors, e.g. `token_type_ids`), so they count as consumed.
        runners = [decode, self._prefill_runner, text_embed, *(m.runner for m in self._modalities)]
        self._graph_inputs = {name for runner in runners if runner is not None for name in runner.input_names}
        # `GenerationMixin` bookkeeping (where it builds tensors, at what precision) — read off the decode
        # runner, whose backend already decided where its outputs land.
        self._device = torch.device(decode.device)
        self._dtype = decode.dtype

    @classmethod
    def from_runners(
        cls, runners: dict[str, ModelRunner], config, generation_config: GenerationConfig | None = None
    ) -> ExportedGenerator:
        """Assemble the generator from `{component_name: runner}` (the names
        `HfExporter.export_for_generation` produces) + the configs — text-only from a `"decode"` runner,
        multi-modal when `"embed_tokens"` and `"<modality>_encoder"` runners are present; each modality's
        precompute is built from `config` alone. `generation_config` must be the one the model was
        **exported with** (it declares the cache the graphs were traced against — save it with the
        artifacts); when `None`, the model config's own generation defaults apply (a growing cache). To
        load from disk, build each `ModelRunner` from its saved artifact (e.g.
        `OnnxModelRunner(onnxruntime.InferenceSession(path))`,
        `DynamoModelRunner(torch.export.load(path).module())`) and pass a config from
        `AutoConfig.from_pretrained(...)`."""
        from .utils import _MODALITY_SPECS

        if generation_config is None:
            generation_config = GenerationConfig.from_model_config(config)
        if "embed_tokens" not in runners:
            return cls(
                config,
                generation_config,
                runners["decode"],
                prefill=runners.get("prefill"),
                encoder=runners.get("encoder"),
            )

        modalities = []
        for name, _getter, input_key, grid_key, token_field in _MODALITY_SPECS:
            if name not in runners:
                continue
            token_id = getattr(config, token_field, None)
            if token_id is None:  # older configs name it `<modality>_token_index`
                token_id = getattr(config, f"{token_field[:-3]}_index", None)
            input_keys = (input_key,) + ((grid_key,) if grid_key is not None else ())
            modalities.append(Modality(token_id, runners[name], input_keys))
        return cls(
            config,
            generation_config,
            runners["decode"],
            prefill=runners.get("prefill"),
            text_embed=runners["embed_tokens"],
            modalities=modalities,
        )

    # ── GenerationMixin plumbing (a real PreTrainedModel provides all of this) ──
    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    def can_generate(self):
        return True

    def is_remote_code(self):
        return False

    def get_experts_implementation(self):
        return {}

    def get_output_embeddings(self):
        return None

    def get_encoder(self):
        return self._encoder

    def get_compiled_call(self, compile_config):
        """`generate` swaps in a compiled forward for the decode loop when the cache is compileable. The
        graphs are already compiled (that is what exporting them was), so hand back the plain call."""
        return self.__call__

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def _validate_model_kwargs(self, model_kwargs):
        # Consumed without being named on `forward`: the modality inputs (pixel_values, input_features, …)
        # in `_merge_modalities`, `mm_token_type_ids` in `_prepare_position_ids_for_generation` (M-RoPE),
        # and anything the graphs declare as an input, which `forward` feeds through.
        consumed = self._graph_inputs | self._modality_keys
        if self._text_embed is not None:
            consumed |= {"mm_token_type_ids"}
        super()._validate_model_kwargs({k: v for k, v in model_kwargs.items() if k not in consumed})

    def _prepare_cache_for_generation(
        self, generation_config, model_kwargs, generation_mode, batch_size, max_cache_length
    ):
        """Build the cache the exported decode graph expects, so `generate` is called like a normal model
        (`generate(input_ids=..., max_new_tokens=N)` — no hand-rolled cache).

        `generate`'s own builder decides everything (kind, size, `EncoderDecoderCache` pairing, a
        source-length static cross cache, …) from the generation config — the same config that built the
        cache at export capture, which is the caller's contract — so the traced and runtime caches match by
        construction. The one addition is materialization: `torch.export` bakes real tensors into the
        graph's input spec, so the lazily-uninitialized layers `generate` hands back have to be filled in
        (`materialize_cache_layers`, the helper the capture uses too)."""
        from .utils import materialize_cache_layers

        super()._prepare_cache_for_generation(
            generation_config, model_kwargs, generation_mode, batch_size, max_cache_length
        )
        for cache_name in ("past_key_values", "cache_params"):
            if (cache := model_kwargs.get(cache_name)) is not None:
                materialize_cache_layers(cache, batch_size, self.config, self._dtype, self._device)

    # ── decode orchestration ──
    def _merge_modalities(self, input_ids, kwargs) -> torch.Tensor:
        """Embed `input_ids` and scatter each present modality's features into its placeholder rows.
        Presence keys on the primary input (`input_keys[0]`: pixel_values / input_features); `generate`
        drops it after prefill but may keep a stale grid kwarg, so don't trigger on the aux keys. The eager
        `get_<modality>_features` computes its grid-derived tensors (`cu_seqlens` / `window_index` / …)
        internally; the export moved them out of the graph, so they're injected here from `self.config`
        alone (`precompute_export_inputs`), after renaming generate's grid kwarg (`image_grid_thw` /
        `video_grid_thw`) to the graph's `grid_thw` input."""
        from .utils import cast_leaf_tensors, precompute_export_inputs

        inputs_embeds = next(iter(self._text_embed(input_ids=input_ids).values()))
        for modality in self._modalities:
            if kwargs.get(modality.input_keys[0]) is None:
                continue
            inputs = {
                "grid_thw" if key.endswith("_grid_thw") else key: kwargs[key]
                for key in modality.input_keys
                if kwargs.get(key) is not None
            }
            # Cast BEFORE the precompute: the graph was traced with the model's dtype for the modality
            # inputs (`pixel_values` …) but with whatever dtype the precompute itself produces (fp32
            # interpolation weights), so casting after would hand the graph bf16 weights it never saw.
            inputs = cast_leaf_tensors(inputs, dtype=modality.runner.dtype, device=self._device)
            inputs = precompute_export_inputs(self.config, inputs)
            # Feed every input the graph declares — including scalars like `max_seqlen` (dynamo's
            # `module()` requires the full traced kwarg set, not just the tensor placeholders).
            feed = {k: v for k, v in inputs.items() if k in set(modality.runner.input_names)}
            features = next(iter(modality.runner(**feed).values()))
            mask = (input_ids == modality.token_id).unsqueeze(-1)
            inputs_embeds = inputs_embeds.masked_scatter(mask, features.to(inputs_embeds.dtype))
        return inputs_embeds

    def _mask_feed(self, runner, attention_mask, position_ids, cache_len):
        """Feed the decode graph's mask input(s). `generate` hands us either a dict of 4D bool masks (one
        per attention type, for mixed full/sliding models) or a single 4D mask; when it drops a mask as
        redundant (`None` — the whole kwarg or one dict slot) the traced graph still takes a tensor there,
        so rebuild the full causal mask the eager model would have built internally."""
        if isinstance(attention_mask, dict):
            attention_mask = {
                layer_type: mask if mask is not None else self._causal_mask(position_ids, cache_len)
                for layer_type, mask in attention_mask.items()
            }
            # Mixed full/sliding models: ONNX declares one input per attention type
            # (`attention_mask.<type>`, flattened by the exporter); dynamo takes the whole dict as a single
            # `attention_mask` pytree kwarg.
            if any(n.startswith("attention_mask.") for n in runner.mask_inputs):
                return {f"attention_mask.{layer_type}": mask for layer_type, mask in attention_mask.items()}
            return {"attention_mask": attention_mask}
        # A graph may take no explicit mask — e.g. a prefill graph builds the causal mask internally from
        # positions on an empty, unpadded cache. Nothing to feed then.
        if not runner.mask_inputs:
            return {}
        if attention_mask is None:
            return {runner.mask_inputs[0]: self._causal_mask(position_ids, cache_len)}
        return {runner.mask_inputs[0]: attention_mask}

    def _causal_mask(self, position_ids, cache_len):
        """Full causal mask `[batch, 1, query, cache_len]` from the positions (per batch row) — what the
        eager model builds internally when `generate` drops the mask as redundant. M-RoPE positions carry
        extra axes in front; the text row (axis 0) is the sequence position. Assumes no left-padding — the
        common single-sequence case."""
        text_positions = position_ids if position_ids.dim() == 2 else position_ids[0]
        positions = torch.arange(cache_len, device=text_positions.device)
        return (positions <= text_positions[..., None])[:, None]

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        """Multi-modal M-RoPE: build the `[text; 3 vision]` 4-axis `position_ids` the exported decode graph
        expects — what `generate` normally gets from a VLM's own override of this method. Mirrors that
        override from the config alone: the text row from `super()` (GenerationMixin), the 3 vision rows
        from `modeling_rope_utils.get_mrope_index` — the very call the model's own override makes — and the
        decode step advances the text row by the cached rope-delta. Models that don't declare M-RoPE (plain
        decoders, VLMs with 1D text positions like Llava) keep the standard positions."""
        from ..modeling_rope_utils import get_mrope_index, uses_mrope

        text_positions = super()._prepare_position_ids_for_generation(inputs_tensor, model_kwargs)
        if self._text_embed is None or not uses_mrope(self.config):
            return text_positions

        cache = model_kwargs.get("past_key_values")
        past_length = cache.get_seq_length() if cache is not None else 0
        if past_length != 0 and getattr(self, "_rope_deltas", None) is not None:
            return text_positions[None, ...] + self._rope_deltas

        if model_kwargs.get("input_ids") is not None and model_kwargs["input_ids"].shape[1] > 0:
            inputs_tensor = model_kwargs["input_ids"]
        # No `attention_mask`: unpadded single sequence, and `generate` has already turned the mask into the
        # per-layer form the attention needs, not the 2D form `get_rope_index` wants. `None` = all valid.
        vision_positions, self._rope_deltas = get_mrope_index(
            self.config,
            inputs_tensor,
            model_kwargs["mm_token_type_ids"],
            image_grid_thw=model_kwargs.get("image_grid_thw"),
            video_grid_thw=model_kwargs.get("video_grid_thw"),
            second_per_grid_ts=model_kwargs.get("second_per_grid_ts"),
        )
        return torch.cat([text_positions[None, ...], vision_positions], dim=0)

    def forward(
        self,
        past_key_values=None,
        input_ids=None,
        decoder_input_ids=None,
        position_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # First step (empty cache) is the prefill; subsequent steps are decode. With no dedicated prefill
        # graph the two runners are the same object, so this just always runs `decode`. For an
        # encoder-decoder model the split is load-bearing beyond shapes: the prefill graph *writes* the
        # cross cache from `encoder_outputs`, decode graphs read it (the modeling's `is_updated` python
        # branch bakes at trace time). With caching off (`use_cache=False`) there is no cache at all and
        # every step re-feeds the whole sequence — that is a prefill each time.
        if past_key_values is None:
            runner = self._prefill_runner
        else:
            runner = self._prefill_runner if past_key_values.get_seq_length() == 0 else self._decode_runner
        text_ids = decoder_input_ids if decoder_input_ids is not None else input_ids
        text = text_ids if self._text_embed is None else self._merge_modalities(text_ids, kwargs)
        feed = {runner.text_input: text}
        if position_ids is not None and "position_ids" in runner.input_names:
            feed["position_ids"] = position_ids
        if encoder_outputs is not None and any(n.startswith("encoder_outputs") for n in runner.input_names):
            feed["encoder_outputs"] = encoder_outputs
        # Extra per-step inputs the graph declares (`token_type_ids`, per-model aux masks, …) come from
        # `generate`'s kwargs. `generate` slices sequence kwargs to the current step only for the ones named
        # on the real model's `forward` — ours takes them generically, so trim them here the same way.
        for name, value in kwargs.items():
            if name in runner.input_names and name not in feed:
                if isinstance(value, torch.Tensor) and value.dim() == 2 and value.shape[1] > text.shape[1]:
                    value = value[:, -text.shape[1] :]
                feed[name] = value
        cache_len = past_key_values.get_max_length() if past_key_values is not None else text.shape[1]
        feed.update(self._mask_feed(runner, attention_mask, position_ids, cache_len))
        if past_key_values is not None:
            feed["past_key_values"] = past_key_values
        outputs = runner(**feed)
        if past_key_values is not None:
            past_key_values = _advance_cache(past_key_values, outputs, num_new_tokens=text.shape[1])
        return CausalLMOutputWithPast(logits=outputs["logits"], past_key_values=past_key_values)
