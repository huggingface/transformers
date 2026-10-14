<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Layer execution plans

An execution plan repeats decoder layers while sharing their original parameters. Load a checkpoint normally, then
enable a plan on its text decoder. The original registered layers, parameter identities, weight keys, and
`num_hidden_layers` are preserved. More executions increase computation and cache storage; training activation memory
also depends on checkpointing. Additional computation does not guarantee better model quality.

Built-in adapters support Llama, the Qwen3.5 dense text decoder, Gemma3n's text decoder (including AltUp streams and
native KV sharing), and RecurrentGemma. Multimodal wrappers can apply a plan to their text backbone. Encoder stacks,
encoder-decoder models, and changing-width layers are excluded. Other constant-width decoders use the adapter protocol
below; constant width alone does not establish compatibility.

## Repeat layers

Layer indices are **zero-based**. `RepeatRange.start` is inclusive, `stop` is exclusive, and `total_passes` counts
**all executions**, including the first pass. Multiple repeat ranges must not overlap.

```python
from transformers import AutoModelForCausalLM, RepeatRange

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-9B", dtype="auto")

# Source layers 0, 1, 1, 2, ...: execute the second layer twice.
model.set_layer_execution_plan(repeats=[RepeatRange(start=1, stop=2, total_passes=2)])
print(model.get_layer_execution_plan().layer_order)
# model(...) and model.generate(...) now follow this order.
```

The model methods infer the source layer count. Other recipes use the same interface:

```python
# Source layers 0, 1, 2, 1, 2, 3, ...: repeat the second through third layers as a block.
model.set_layer_execution_plan(repeats=[RepeatRange(start=1, stop=3, total_passes=2)])

# Execute the entire decoder stack twice: 0, 1, ..., n-1, 0, 1, ..., n-1.
num_layers = model.config.get_text_config().num_hidden_layers
model.set_layer_execution_plan(list(range(num_layers)) * 2)

# Explicit orders can repeat, reorder, or omit source layers.
model.set_layer_execution_plan([0, 1, 1, *range(2, num_layers)])
```

Embedding and stack input projections run once, each decoder block runs according to the plan, and final normalization,
output projections, and the output head run once after the complete plan. `output_hidden_states=True` records the
logical executions rather than the number of
unique source layers. The getter returns an immutable snapshot, so inspecting or keeping a plan cannot change the
model configuration. `LayerExecutionPlan.from_repeats` remains available when constructing plans independently of a model.

## Independent cache states

Every logical execution position owns an independent cache layer by default. Repeated visits to the same source layer
share weights and retain separate KV, convolution, and recurrent histories. The same execution position retains its history
when processing subsequent tokens. Token positions advance once per decoder forward, not once per depth loop.

Normal `generate` creates a `LayerExecutionCache` automatically. For manual cached decoding:

```python
from transformers import LayerExecutionCache

cache = LayerExecutionCache(model.config)
outputs = model(input_ids=prompt_ids, past_key_values=cache, use_cache=True)
outputs = model(input_ids=next_token_ids, past_key_values=cache, use_cache=True)
```

Keep the plan fixed while using a cache. After changing plans, create a new cache and recompute the prefix. A cache
from another plan, or an ordinary `DynamicCache`, is rejected. The same plan and model use these backends:

| `cache_implementation` | Storage |
|---|---|
| `dynamic` (default) | Native growing KV and recurrent cache layers |
| `static` | Preallocated attention storage; pass `max_cache_len` for manual decoding |
| `offloaded` / `offloaded_static` | Execution-boundary CUDA/CPU transfers, including recurrent and custom state; CPU runs need no transfer |
| `quantized` | Full-attention KV uses `quanto`, `hqq`, or the portable packed `torch` backend; sliding KV and recurrent state keep native precision |

```python
cache = LayerExecutionCache(model.config, cache_implementation="static", max_cache_len=4096)
tokens = model.generate(prompt_ids, max_new_tokens=64, cache_implementation="static")

tokens = model.generate(
    prompt_ids, max_new_tokens=64,
    cache_implementation="quantized",
    cache_config={"backend": "torch", "nbits": 4, "q_group_size": 64, "residual_length": 128},
)
```

Quantization changes numerical results. The portable backend packs 2/4/8-bit codes without an optional native
extension; Quanto and HQQ require their packages and platform-compatible kernels. Quantizing KV is separate from
quantizing model weights. Static decode is covered by a `torch.compile(..., backend="eager", fullgraph=True)` test;
compiler performance and accelerator kernels require platform-specific validation.

## Native cross-layer KV dependencies

Some architectures omit K/V projection parameters on consumer layers. Retain their native dependency explicitly:

```python
from transformers import LayerExecutionPlan

# gemma3n is an already loaded Gemma3n text model or causal-LM wrapper.
count = gemma3n.config.get_text_config().num_hidden_layers
gemma3n.set_layer_execution_plan(LayerExecutionPlan(tuple(range(count)) * 2, kv_sharing="native"))
```

Each producer visit still owns a separate cache. A consumer reads the **most recent earlier visit** of its native
producer. The consumer stores no redundant KV history. `kv_dependencies` can select a particular producer execution
instead: its entries are logical execution indices, with `None` selecting the default binding. For four source
layers where 2 reads 0 and 3 reads 1, the order `(0, 1, 2, 3, 0, 1, 2, 3)` binds producers to
`(None, None, 0, 1, None, None, 4, 5)`. Set an explicit entry of `0` at position 6 to reuse the first visit of layer 0.
The compiler rejects missing, future, or incorrect producers before enabling a plan. The default independent policy
rejects architectures requiring native sharing rather than inventing missing projection weights.

## Assisted generation and continuous batching

Assisted generation records token boundaries so rejected candidates roll back **all** execution-local state:

```python
tokens = model.generate(prompt_ids, assistant_model=assistant, max_new_tokens=64)
```

This also works when the assistant has its own loop plan. Candidate verification with recurrent state runs token by
token, then releases accepted history; it provides correct rollback but does not promise a speculative speedup.
For manual rollback, call `cache.activate_past_recording()` before speculative work, `cache.crop(-rejected_tokens)`,
and `cache.commit_past()` after committing a prefix. Committing releases older rollback boundaries. Static assisted
generation retains the generation API's native restriction.

```python
from transformers import ContinuousBatchingConfig

results = model.generate_batch(
    [[1, 2, 3], [4, 5]], max_new_tokens=32,
    continuous_batching_config=ContinuousBatchingConfig(max_requests_per_batch=8),
)

with model.continuous_batching_context_manager(warmup=False) as manager:
    request_id = manager.add_request([1, 2, 3], max_new_tokens=32, streaming=True)
    for result in manager.request_id_iter(request_id):
        print(result.generated_tokens)
```

The portable manager continuously admits requests and batches equal-length dynamic-cache requests. Every request has
its own cache and logits processors; completion and cancellation release its execution state. Static, quantized,
and offloaded backends run as individual request groups. `max_requests_per_batch` limits resident active requests
(default 16); pending requests wait without allocating caches. Normal `stop()` drains pending work;
`stop(hard_stop=True)` fails it immediately. Pause, callbacks, cancellation, streaming, and worker synchronization
use the existing manager API. Stop the manager before changing a plan. Its portable path uses ordinary attention
kernels, without paged block sharing, CUDA graph warmup, or asynchronous GPU scheduling.

Pure-attention models can select the native paged scheduler with `generation_config.cache_implementation="paged"`.
Paged cache allocation expands logical cache metadata without changing the model's source-layer configuration.
Native paged allocation supports its existing attention types; recurrent architectures use the portable manager.
Pipeline execution uses the portable manager; explicitly combining it with the native paged scheduler is rejected.
Initialize/start the manager on every TP or PP rank, and submit requests on its driver.

## Training and checkpoints

The execution graph contains every repeated call. Gradients from all uses flow into the original shared parameters;
there are no detached loop boundaries or extra optimizer parameters. Gradient-enabled training disables caching and
requires `past_key_values=None`. Generation under `torch.no_grad()` or `torch.inference_mode()` can use independent
execution caches while the model remains in training mode, as needed by reinforcement learning trainers. Disable
gradient checkpointing during generation and restore it before training. Non-reentrant gradient checkpointing can
reduce activation memory:

```python
model.train()
model.gradient_checkpointing_enable()  # Defaults to use_reentrant=False.
loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
loss.backward()
```

`save_pretrained` writes the execution order as `layer_execution_plan` and nondefault dependency options as
`layer_execution_options` in the text configuration. Loading with
`from_pretrained` automatically restores it while keeping the original weight keys. This requires a Transformers
version containing layer execution support. Built-in adapters also support pickling for process spawning.

```python
model.save_pretrained("loop-checkpoint")
reloaded = AutoModelForCausalLM.from_pretrained("loop-checkpoint")
assert reloaded.get_layer_execution_plan() == model.get_layer_execution_plan()

# Restore the original decoder computation using the current weights.
model.set_layer_execution_plan(None)
assert model.get_layer_execution_plan() is None
```

Use the setter to update a loaded model; editing `config.json` applies on the next load. Changing or disabling a plan
does not reset learned weights. Reconfigure between complete forward/backward steps, and discard any existing cache.

### TRL references and PEFT adapter checkpoints

The recorded SFT/DPO/GRPO paths use unmodified TRL 1.15.0 and Accelerate 1.15.0. They require this Transformers
checkout, which provides model plan dispatch, execution-local caches, Trainer checkpoint handling, and automatic
ZeRO-3 shared-gradient integration. No TRL or Accelerate source fork is required for these tested configurations.

An in-memory policy plan is not automatically copied to a reference model that a trainer reloads from the original
checkpoint. Saving only a PEFT adapter also does not serialize an in-memory base plan: automatic adapter loading
follows the base model path recorded in the adapter configuration. When all of these models should use the same
plan, save a planned base checkpoint and reload it before constructing the trainer or adapter:

```python
model.save_pretrained("planned-base")
policy = AutoModelForCausalLM.from_pretrained("planned-base")
# Pass policy to the trainer, or create the PEFT adapter from policy.
```

Keep that base checkpoint accessible for adapter reloads. Alternatively, explicitly configure the DPO reference and
restore the base plan before loading an adapter. Policy and reference plans may intentionally differ. The public
plan functions accept standard PEFT wrappers. Single-GPU SFT, DPO, and GRPO with Hugging Face generation were checked
on TRL 1.15.0 / PEFT 0.21.2 with Llama and Qwen3.5, including original Qwen3.5-9B full-parameter and LoRA training.
GRPO used `use_vllm=False`. A two-H20 follow-up has completed SFT/DPO/GRPO × full/LoRA/NF4 QLoRA ×
DDP/ZeRO-2/ZeRO-3 with the second layer repeated for both tiny Qwen3.5 and original Qwen3.5-9B (27 cases each),
including separate-process training-state resume. Tiny cases also compare resumed and uninterrupted updates.
An additional 15 tiny training cases cover Llama QLoRA across those trainers/backends and Qwen3.5 SFT QLoRA
range/whole-stack repetition. Six nonzero-reference DPO LoRA/QLoRA cases check frozen-adapter preservation across
cold resume on all three backends. Twelve additional original 9B SFT LoRA/QLoRA cases cover range/whole-stack
repetition across all three backends and dynamic/static/offloaded/offloaded-static generation caches. The complete
selected matrix has 87 passing configurations, 180 training phases, and 360 rank records. Interrupted and corrected
attempts are preserved separately. Full-parameter 9B runs use SGD; tiny and adapter runs use AdamW. These results do
not establish full-parameter 9B AdamW, load-time ZeRO-3 initialization, optimizer offload, or other TRL combinations.
vLLM loop execution is not implemented or runtime-validated, including its Transformers backend and TRL's vLLM
server/colocate modes. These paths require separate integration of the execution plan and vLLM-managed state.
For trainer gradient checkpointing, use `gradient_checkpointing_kwargs={"use_reentrant": False}`.
TRL 1.15.0's GRPOTrainer initially overrides this to reentrant checkpointing for PEFT with ZeRO-3. Its generation
utility later re-enables checkpointing without arguments, selecting this checkout's non-reentrant default. The
checks follow those ordinary utilities. Custom contexts should explicitly restore their intended keyword arguments.
Native non-reentrant ZeRO training is also checked against expanded layers.

Trainer preserves existing adapter dtypes, restores both a root-level PEFT `default` adapter and named directories,
and selects the active adapter's path for best-checkpoint loading. Multi-adapter DeepSpeed checkpoints retain frozen
reference parameters, and ZeRO-3 exports gather their weights. DeepSpeed's exclusion flag covers all frozen parameters,
so these training checkpoints can include the frozen base and be larger than single-adapter checkpoints.
Trainer saves full-model checkpoints in their current
runtime key format for direct `state_dict` resume. NF4 validation uses BF16 compute and quantization storage, places
each rank's quantized model on its own GPU, and reapplies the same k-bit preparation and backend precision policy
when reloading. DeepSpeed 0.19.7 casts parameters while preserving FP32 buffers such as RoPE frequencies.

### Accelerate integration

Accelerate 1.15.0 is exercised through Trainer on single-GPU, two-GPU DDP, and ZeRO-2/ZeRO-3 paths. Set the plan before
Trainer or `Accelerator.prepare(...)`; the underlying model retains its shared parameters and performs the loop.
For a standalone Accelerate/DeepSpeed ZeRO-3 loop, call `configure_deepspeed_layer_execution` on the engine returned
by `prepare(...)` before training. Trainer installs this coordinator automatically; standalone `prepare(...)` does not.
Preserve the planned model configuration when restoring a custom loop's training checkpoint.

These Trainer checks do not establish all standalone Accelerate plugins or launch configurations. Native PyTorch
FSDP2/TP/PP checks are separate; the corresponding Accelerate/TRL paths, combined parallel layouts, other accelerators,
and physical multi-host operation require independent integration or validation.

## Distributed execution

The executor calls the original blocks through their normal module interface. DDP reduction hooks, FSDP sharding hooks,
and tensor-parallel transforms remain attached to the same registered parameters. No extra parameter copies or
optimizer groups are introduced. Set a plan before constructing your distributed trainer. The standalone functions
also accept an existing DDP or FSDP container and return that same container:

```python
from transformers import get_layer_execution_plan, set_layer_execution_plan

# ddp_model is an existing DistributedDataParallel wrapper.
set_layer_execution_plan(ddp_model, repeats=[RepeatRange(1, 2)])
print(get_layer_execution_plan(ddp_model).layer_order)
```

Trainer configures repeated source gradients automatically after DeepSpeed initialization. For a manual engine,
call the integration after setting the plan and initializing DeepSpeed:

```python
import deepspeed
from transformers.layer_execution.deepspeed import configure_deepspeed_layer_execution

# Supply resolved numeric batch sizes and your optimizer settings in deepspeed_config.
engine, optimizer, _, _ = deepspeed.initialize(model=model, config=deepspeed_config)
configure_deepspeed_layer_execution(engine)
```

For ZeRO-3, this defers reduction of a repeated source parameter until its full backward contribution has accumulated,
then uses the optimizer's ordinary reduction/partitioning. Source registrations and optimizer groups remain unchanged.
Use `set_layer_execution_plan(engine, ...)` for later plan changes between complete steps. The integration is verified
with DeepSpeed 0.19.7; other releases require checks of the reduction API and backward-hook ordering.

All ranks must use the same plan. Synchronize changes between complete training steps. For manual DDP, set
`find_unused_parameters=True` if an explicit order omits any source layer. `Trainer` selects this setting automatically
for an initially configured plan, unless you override `ddp_find_unused_parameters`. Set it explicitly if later plans
may omit layers. Non-reentrant checkpointing supports repeated calls; explicit `use_reentrant=True` requires DDP's
`static_graph=True` and a fixed execution graph. See the [PyTorch DDP documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html).

The detailed CUDA and original-weight measurements below describe the pre-sync
[`loop-baseline-2026-10-11`](https://github.com/bebetterest/transformers_loop/tree/loop-baseline-2026-10-11)
at `e709e88b9d8db18a1648c023ba9d10f732dd2237`. Later upstream merges have separate regression results in
[fork maintenance](https://github.com/bebetterest/transformers_loop/blob/main/MAINTENANCE.md).

CPU validation uses tiny models on PyTorch 2.14.1. CUDA validation uses NVIDIA H20 devices on PyTorch 2.12.1 / CUDA 13.0,
including a two-device NCCL 2.29.7 group with NVLink. The checks cover the original BF16 text weights from
`Qwen/Qwen3.5-9B` at revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`:

| Path | Validation |
|---|---|
| Single CPU | Expanded-model reference, shared gradients, independent KV and recurrent states, greedy/beam generation, original checkpoints and text extraction from a multimodal checkpoint |
| Two-process CPU DDP | Gradient accumulation with `no_sync`, non-reentrant checkpointing, fixed-graph reentrant checkpointing, omitted layers, plan changes, pickle-based spawning, AdamW updates and cached generation |
| Two-process CPU FSDP2 | Per-block parameter sharding and resharding, repeated calls, checkpointing, shared gradients, AdamW updates and cached generation |
| Two-process CPU tensor parallelism | Native TP placements and transforms, checkpointing, shared gradients, AdamW updates and cached generation |
| Two-process CPU Trainer | Checkpointing, gradient accumulation, shared parameter updates and omitted layers |
| Two-process CPU pipeline | Stage-boundary communication across loop passes, native KV dependencies, AltUp streams, external recurrent state, projected OPT inputs, tied embeddings, accumulation, checkpointing, optimizer steps and cached generation |
| Two-process CPU continuous batching | TP/PP driver-only admission, request-local caches, synchronized tokens and shutdown |
| Single CUDA | FP32/BF16 shared gradients against independent logical layers, checkpointing, dynamic/static offload, Torch/Quanto/HQQ quantized caches, beam reorder/crop/reset, Inductor fullgraph static decode, and BF16 Trainer checkpoint/resume |
| CUDA stateful adapters | Gemma3n native KV dependencies and AltUp streams, RecurrentGemma external state, independent expanded gradient references with and without checkpointing |
| Original Qwen3.5-9B CUDA | Second-layer, range and whole-stack repetition; original parameter identities and keys; independent cache buffers; dynamic/static/offloaded/offloaded-static prefill/decode; greedy cache/recompute agreement; restoring the native forward; one full-gradient checkpointed SGD update and generation afterward |
| Original Qwen3.5-9B, two-device CUDA | DDP and native FSDP2/TP/PP loading, three repeat plans, all four unquantized cache backends, independent logical buffers, all-parameter checkpointed SGD updates and generation afterward; TP gradients checked against separately instantiated logical TP blocks |
| Two-device CUDA DDP | Shared gradients against expanded layers, accumulation, non-reentrant and fixed-graph reentrant checkpointing, omitted layers, plan changes, pickle spawning, AdamW updates and dynamic/static/offloaded/offloaded-static generation |
| Two-device CUDA DataParallel | Replica-bound forwards and shared gradients against independent logical layers, with and without checkpointing |
| Two-device CUDA FSDP2 / TP | Per-block sharding and resharding or native TP placements, expanded shared-gradient references, checkpointing, AdamW updates and all four unquantized cache backends |
| Two-device CUDA pipeline | Tied embeddings, captured hidden states and attentions, auxiliary losses, accumulation, checkpointing, complete model save/reload, native KV dependencies, AltUp, external recurrent state and projected adapter inputs |
| Two-device CUDA Trainer | DDP checkpoint/resume, standalone pipeline FP32/BF16 clipping and exact checkpoint/resume, native load-time TP/FSDP2 training and model-only checkpoint/reload |
| Two-device CUDA continuous batching | TP/PP driver-only admission, request-local caches, synchronized tokens, background-thread device selection and shutdown |
| Two simulated nodes on one host | Independent torchrun agents with one visible GPU each, distinct global ranks and local rank zero; Socket data channels; DDP/FSDP2/TP/PP gradients, four cache backends, Trainer checkpoints, stateful adapters and continuous batching |
| Original Qwen3.5-9B, two simulated nodes | Original-weight DDP/FSDP2/TP/PP loading, second-layer/range/whole-stack repeats, all four unquantized cache backends, shared-gradient checkpointed SGD updates and generation afterward; distinct GPUs and Socket data channels verified |
| Multiple physical hosts | Deferred; the two-node launch checks run on one physical host |

The distributed checks compare against independently expanded logical layers and sum their gradients into each source
parameter. Floating-point comparisons allow the different reduction order used by parallel kernels. Additional
two-device NF4 and ZeRO-2/ZeRO-3 regressions check shared gradients, optimizer updates, independent caches and adapter
reload. These do not establish every TRL/distributed combination; the version-bounded matrix is described above.
FSDP1, combined parallel layouts, optimized recurrent FLA/causal-convolution kernels, CPU/NVMe optimizer offload and
other weight quantizers still need separate validation. Models without an enabled plan retain their native execution
paths.

The simulated-node regression starts two independent `torchrun` agents with `--nnodes=2` and
`--nproc-per-node=1`. Each worker sees one distinct physical GPU, has `LOCAL_RANK=0`, and receives its global rank
from the launcher. This checks that global rank is used for data partitions and process groups, while local rank
selects the GPU. CUDA tests temporarily disable P2P, shared-memory transport and NVLS, force `NCCL_NET=Socket`,
and check the NCCL logs for Socket data channels on both ranks. These settings apply only to test subprocesses.

Run the CPU launcher check and the 19 CUDA cases with:

```bash
TRANSFORMERS_TEST_DEVICE=cuda python -m pytest tests/test_layer_execution_multinode.py -q
```

The simulation uses one host's network stack and a shared filesystem. Physical NICs, IB/RoCE, separate host
filesystems, network interruptions and elastic worker restarts still require their own validation. GPU tests skip
when two CUDA devices are unavailable; the CPU launcher test can run without GPUs.

The original-weight 9B checks also pass with this two-agent layout and Socket transport. They use the same
plans, cache backends, gradient references and numerical tolerances as the two-device checks below. Both workers
use local device zero while their global ranks differ. The observed maximum logit differences remain `0.375` for
TP and `0.203125` for DDP/FSDP2/PP, with identical checked greedy tokens.

When constructing a manual TP optimizer, put ordinary tensors and DTensors from different meshes in separate
parameter groups, or set `foreach=False` and avoid fused updates over mixed tensor types. `Trainer` separates these
groups automatically. Native `DistributedConfig` load-time TP/FSDP2 currently requires `save_only_model=True` or
`save_strategy="no"` in `Trainer`: saving and reloading model weights works, but restoring optimizer training state is
not supported by that native loading path. DDP and the standalone loop pipeline support full Trainer checkpoint/resume.

The 9B check loads all 8,953,803,264 text parameters without missing weights; its four checkpoint shards are checked
against the fixed revision's LFS SHA-256 hashes. It uses 13 input tokens and short greedy continuations, so it verifies
execution and training mechanics rather than model quality or long-run convergence. Identity execution, disabling
the plan, and each cache prefill match exactly in this check. BF16 cached decoding allows `atol=0.25, rtol=0.025`
against full-prefix recomputation; the observed maximum logit difference is `0.203125`, and greedy tokens agree.
The full-gradient update repeats the second layer and uses gradient checkpointing with SGD, peaking at 38.3 GiB of
allocated CUDA memory. The quantized-cache and Inductor checks use tiny models, not the 9B checkpoint.

The two-device 9B checks allow `atol=0.5, rtol=0.03` for BF16 logits against single-device full-prefix recomputation.
The maximum observed difference is `0.375` for TP and `0.203125` for DDP/FSDP2/PP; greedy tokens agree in all checked
plans and cache backends. Training compares 128 gradient elements and the full gradient norm of every source
parameter. DDP/FSDP2 use a single-device reference with the same per-rank microbatch shape, PP uses the same global
batch, and TP uses independently instantiated logical blocks on the same TP mesh. BF16 TP reduction differs from
single-device BF16 arithmetic; its observed gradient-probe difference is `0.013671875`, while the loop model's loss
and all source gradient norms match the expanded TP reference. These are short correctness checks, not convergence
or parallel throughput benchmarks.

Pipeline execution keeps source parameters on their owning stage and communicates the hidden/state pytree when the
next execution changes owner, including jumps back to an earlier stage. It is a sequential schedule with autograd
communication; overlapped GPipe/1F1B scheduling is not implemented. The pipeline runtime checks that ranks agree on
the plan. Manual pipeline forwards must receive identical inputs on every stage. Use `apply_pipeline_parallelism` on an enabled loaded model, or load a loop checkpoint with the existing
pipeline `DistributedConfig`. Run `save_pretrained` on every pipeline rank to gather a complete original-weight
checkpoint. `Trainer` supports a standalone pipeline group: batches are replicated across stages, gradient clipping
uses the global norm, and optimizer states are saved per stage. Resume with the same pipeline partition. Combined
PP/data/tensor/FSDP Trainer layouts need a dedicated integration and are rejected when the training world does not
match the standalone PP group.

Captured hidden states and attentions are returned in logical execution order on every stage, including selected
hidden-state indices. Capturing adds per-step communication. Losses computed from these returned tensors are averaged
across the replicated stages, just like the logits loss; compute the same objective on every rank.

Run the feature and distributed tests without downloading model weights:

```bash
HF_HUB_OFFLINE=1 pytest tests/test_layer_execution*.py -q
```

## Add a decoder adapter

Constant width is necessary, but does not make an arbitrary decoder automatically compatible. New model types need an
adapter to preserve their input transformations, block calling conventions, masks, cache protocol, and output semantics.
Subclass `DecoderLayerExecutionAdapter` and register it with `register_layer_execution_adapter`. The following hooks
keep these differences out of the plan and cache engine:

| Adapter hook | Responsibility |
|---|---|
| `get_layers(decoder)` | Return original registered modules, including modules in a nested container. Defaults to `layer_container_name="layers"`. |
| `validate(decoder)` | Check stack constraints before modifying the model. The default checks constant width using `per_layer_config`, rather than embedding weights or widths. |
| `prepare(decoder, inputs_embeds, attention_mask, position_ids, cache, **kwargs)` | Return `(hidden_states, context)` after one-time input projection, position embedding, or layout changes. |
| `layer_kwargs(decoder, source_index, context)` | Select model-specific arguments for the source layer. These override matching caller kwargs. |
| `execute_layer(layer, hidden_states, layer_kwargs, cache, use_cache)` | Call the module with its original hooks and the routed cache. Override for keyword-only inputs, renamed cache arguments, or kwargs filtering. |
| `extract_hidden_states(layer_output)` | Extract the next tensor state from tensor, tuple, or structured outputs. |
| `execute_step(...)` / `extract_state(...)` | Evolve a tensor pytree or `LayerExecutionState` without dropping independently evolving streams. Defaults preserve existing single-tensor adapters. |
| `kv_dependencies(decoder)` | Declare native consumer-source to producer-source mappings. |
| `step_kwargs(...)` / `finish_step(...)` | Read/publish dependencies by logical execution index; use a `UserDict` for mutable KV publication across FSDP hooks. |
| `slice_token_kwargs(...)` | Slice model-specific sequence inputs when speculative verification records individual token boundaries. |
| `supports_compile` | Opt into automatic generation compilation after validating the adapter's complete static decode path. Defaults to `False`. |
| `finalize(decoder, hidden_states, cache, context)` | Apply final norm/output projection and construct the model's output, including model-specific auxiliary data. |

The stack's prepared tensor may differ in width or layout from its input embeddings. Every block must preserve the
**prepared state streams' shapes and pytree structure**. The executor also preserves the original decoder forward signature and positional argument
order. For models using a different cache input/output name, set `cache_name` on the adapter; the tested names are
`past_key_values` and `cache_params`.

For example, this GPT-2 adapter supports the standard decoder without cross-attention:

```python
import torch
from transformers import DecoderLayerExecutionAdapter, register_layer_execution_adapter
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions


class GPT2ExecutionAdapter(DecoderLayerExecutionAdapter):
    layer_container_name = "h"

    def validate(self, decoder):
        super().validate(decoder)
        if decoder.config.add_cross_attention:
            raise ValueError("This adapter requires GPT-2 without cross-attention.")

    def prepare(self, decoder, inputs_embeds, attention_mask, position_ids, cache, **kwargs):
        if position_ids is None:
            offset = cache.get_seq_length() if cache is not None else 0
            position_ids = (torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + offset).unsqueeze(0)
        hidden_states = inputs_embeds + decoder.wpe(position_ids)
        if kwargs.get("token_type_ids") is not None:
            hidden_states = hidden_states + decoder.wte(kwargs["token_type_ids"])
        return decoder.drop(hidden_states), {
            "position_ids": position_ids,
            "attention_mask": create_causal_mask(decoder.config, inputs_embeds, attention_mask, cache),
        }

    def extract_hidden_states(self, output):
        return output if isinstance(output, torch.Tensor) else output[0]

    def finalize(self, decoder, hidden_states, cache, context):
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=decoder.ln_f(hidden_states), past_key_values=cache
        )


register_layer_execution_adapter("gpt2", GPT2ExecutionAdapter)
# Load normally, then use model.set_layer_execution_plan(...).
```

Define custom adapters in an importable Python module for process spawning, and register them in each process before
loading a loop checkpoint. Checkpoints store the plan and original weights, not executable adapter code. Adapter
instances must not hold mutable forward state; keep per-forward context in `prepare` and per-execution history in the
cache. Always call `layer(...)` rather than `layer.forward(...)` to retain checkpointing and distributed hooks.

An adapter must describe a causal decoder-only stack and declare its cross-layer dependencies. The cache engine uses
registered native cache layer types. For persistent state outside the native cache protocol, store tensor histories
in the execution view's `cache.state` dictionary. RecurrentGemma demonstrates this integration without request state
on shared parameter objects. No-cache training must keep captured inputs immutable during checkpoint recomputation.
Use `LayerExecutionState(hidden_states, streams={"auxiliary": tensor})` or a tensor pytree for multiple streams; every
leaf remains attached to autograd and participates in pipeline transfers. Gemma3n demonstrates a stacked AltUp layout.
Architecture-specific skip connections and extra output semantics belong in the adapter. Width checks alone cannot
establish compatibility. Preserve or explicitly reject behavior such as LayerDrop,
cross-attention, auxiliary outputs, and cached chunk restrictions in the adapter.

Extension tests cover GPT-2's learned positions and tuple outputs, OPT's different embedding/decoder widths and nested
containers, and Mamba's pure recurrent stack, `cache_params`, and custom hidden-state outputs. OPT and Mamba tests also
compare shared gradients, checkpointing, cached generation, save/load, and pickle against native models with
independently expanded layers, including two-process CPU DDP and FSDP2 with checkpointing. Nested language-model
wrappers and custom adapters pickled into a fresh process are also covered. The example Mamba adapter allows
single-token cached continuation because the current
native full-sequence scan does not consume the previous recurrent state for cached chunks. These examples validate
the extension interface; they are not additional built-in adapters.

The implementation is split by responsibility under `src/transformers/layer_execution/`: `plan.py` compiles orders
and dependencies; `state.py` defines structured streams; `cache.py` routes state and rollback; `backends.py` selects
storage; `adapters.py` and `model_adapters.py` handle model semantics; `pipeline.py` handles communication, gradients,
and checkpoint gathering; `continuous.py` and `paged.py` handle request scheduling and paged routing; `executor.py`
configures models and runs shared modules. `deepspeed.py` completes repeated source gradients before ZeRO-3 reduction.
Adding an adapter does not require rewriting the plan or cache engine.

## API

[[autodoc]] RepeatRange

[[autodoc]] LayerExecutionPlan
    - from_repeats

[[autodoc]] LayerExecutionCache

[[autodoc]] LayerExecutionState

[[autodoc]] LayerExecutionStep

[[autodoc]] set_layer_execution_plan

[[autodoc]] get_layer_execution_plan

[[autodoc]] DecoderLayerExecutionAdapter

[[autodoc]] register_layer_execution_adapter
