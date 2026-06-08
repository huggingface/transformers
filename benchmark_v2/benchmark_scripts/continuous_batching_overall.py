"""
Continuous batching overall benchmark suite.

Runs CB in-process across many configurations (GSM8K prompts and synthetic
data) and can compare throughput against a previously-saved run.
"""

import argparse
import gc
import json
import os
import time
import types
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTask, LightevalTaskConfig
from lighteval.tasks.prompt_manager import PromptManager
from lighteval.tasks.registry import Registry
from lighteval.tasks.requests import Doc
from tabulate import tabulate

from transformers import AutoModelForCausalLM, AutoTokenizer, ContinuousBatchingConfig, GenerationConfig


# Defaults
RESULTS_DIR = Path(__file__).parent.parent / "benchmark_results/cb_overall/"


# Auxiliary functions
def _fmt(val: Any, spec: str = "", missing: str = "X") -> str:
    """Format `val` per `spec`, or return `missing` if val is None."""
    return format(val, spec) if val is not None else missing


def _config_summary(cfg: Any) -> dict[str, Any]:
    """Extract a JSON-friendly summary of a dataclass/config object."""
    raw = cfg.to_dict() if hasattr(cfg, "to_dict") else cfg.__dict__
    return {k: v for k, v in raw.items() if isinstance(v, (int, float, str, bool, type(None)))}


# Data-related functions
def _build_gsm8k_platinum_module() -> types.ModuleType:
    """Define the gsm8k_platinum custom task inline so lighteval's Registry can pick it up via `custom_tasks=`."""

    def gsm8k_platinum_prompt(line, task_name=None):
        return Doc(
            task_name=task_name,
            query=f"Question: {line['question']}\nAnswer:",
            choices=[f" {line['answer']}"],
            gold_index=0,
        )

    metrics = list(Registry().load_all_task_configs()["gsm8k"].metrics)

    mod = types.ModuleType("_gsm8k_platinum_inline")
    mod.TASKS_TABLE = [  # type: ignore
        LightevalTaskConfig(
            name="gsm8k_platinum",
            prompt_function=gsm8k_platinum_prompt,
            hf_repo="madrylab/gsm8k-platinum",
            hf_subset="main",
            evaluation_splits=("test",),
            few_shots_split="test",
            few_shots_select="random_sampling",
            generation_size=256,
            stop_sequence=["Question:"],
            metrics=metrics,
        ),
    ]
    return mod


def _build_lighteval_inputs_scorer(
    tokenizer: AutoTokenizer,
    *,
    task_spec: str,
    task_name: str,
    use_chat_template: bool,
    custom_tasks: Any = None,
    primary_metric: str | None = None,
    stop_sequences: tuple[str, ...] = (),
) -> tuple[list[list[int]], Callable[[Any], float]]:
    """Tokenize prompts and build a per-sample scorer for any lighteval task."""
    r = Registry(tasks=task_spec, **({"custom_tasks": custom_tasks} if custom_tasks else {}))
    metric = r.task_to_configs[task_name][0].metrics[0]
    tasks_dict = r.load_tasks()
    LightevalTask.load_datasets(tasks_dict, 1)
    docs = next(iter(tasks_dict.values())).get_docs()

    pm = PromptManager(use_chat_template=use_chat_template, tokenizer=tokenizer, system_prompt=None)
    prompts = [pm.prepare_prompt(doc) for doc in docs]
    inputs = tokenizer(prompts, add_special_tokens=not use_chat_template)["input_ids"]

    def score(outputs) -> float:
        scores = []
        for doc, (_, out) in zip(docs, outputs.items()):
            text = tokenizer.decode(out.generated_tokens, skip_special_tokens=True)  # type: ignore
            for s in stop_sequences:
                text = text.split(s, 1)[0]
            value = metric.sample_level_fn.compute(doc, ModelResponse(text=[text]))
            # Grouped metrics return a dict keyed by sub-metric — pick the primary one.
            scores.append(value[primary_metric] if isinstance(value, dict) else value)
        return sum(scores) / len(scores)

    return inputs, score


def get_tokenized_gsm8k(
    tokenizer: AutoTokenizer, n_fewshot: int = 8
) -> tuple[list[list[int]], Callable[[Any], float]]:
    """GSM8K-Platinum few-shot inputs and scorer using the same lighteval extractive_match as the gsm8k task."""
    return _build_lighteval_inputs_scorer(
        tokenizer,
        task_spec=f"gsm8k_platinum|{n_fewshot}",
        task_name="gsm8k_platinum",
        use_chat_template=False,
        custom_tasks=_build_gsm8k_platinum_module(),
        stop_sequences=("Question:",),
    )


def get_tokenized_ifeval(tokenizer: AutoTokenizer) -> tuple[list[list[int]], Callable[[Any], float]]:
    """IFEval inputs (chat-templated, 0-shot) and scorer reporting prompt-level strict accuracy."""
    return _build_lighteval_inputs_scorer(
        tokenizer,
        task_spec="ifeval|0",
        task_name="ifeval",
        use_chat_template=True,
        primary_metric="prompt_level_strict_acc",
    )


def get_random_data(batch_size: int, num_tokens: int, vocab_size: int = 16000) -> list[list[int]]:
    """Random token sequences of fixed length, for raw throughput tests."""
    rng = torch.Generator().manual_seed(0)
    return [torch.randint(0, vocab_size, (num_tokens,), generator=rng).tolist() for _ in range(batch_size)]


# Benchmark entries and collection
@dataclass
class BenchmarkEntry:
    """Single CB run: what was fed in, which configs were used, and the resulting metrics."""

    label: str
    num_samples: int
    avg_input_tokens: float
    max_new_tokens: int
    cb_config: dict[str, Any]
    gen_config: dict[str, Any]
    time_seconds: float | None = None
    num_tokens: int | None = None
    throughput_tok_per_sec: float | None = None
    peak_memory_gb: float | None = None
    accuracy: float | None = None
    error: str | None = None


class BenchmarkResults:
    """Holds all CB benchmark runs and the shared model they execute against."""

    def __init__(self, model_id: str, attn_impl: str, tp_size: int = 1, dp_size: int = 1):
        self.model_id = model_id
        self.attn_impl = attn_impl
        self.tp_size = tp_size
        self.dp_size = dp_size
        # For now, TP and DP are mutually exclusive
        if self.tp_size > 1 and self.dp_size > 1:
            raise ValueError("TP and DP cannot be used together")
        # torchrun sets these per worker; the work is independent so no process group is needed.
        self.global_rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        # Pin this worker to its own GPU so all torch.cuda calls (memory stats, cleanup) target it.
        if self.dp_size > 1:
            torch.cuda.set_device(self.local_rank)
        # Entries accumulator
        self.entries: list[BenchmarkEntry] = []

    def cleanup(self) -> None:
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()

    def _get_model(self) -> Any:
        self.cleanup()
        # tp_plan and device_map are mutually exclusive — TP uses its own placement.
        if self.tp_size > 1:
            placement = {"tp_plan": "auto"}
        elif self.dp_size > 1:
            placement = {"device_map": self.local_rank}
        else:
            placement = {"device_map": 0}
        model = AutoModelForCausalLM.from_pretrained(self.model_id, attn_implementation=self.attn_impl, **placement)
        return model.eval()

    def add_benchmark(
        self,
        data: list[list[int]],
        max_new_tokens: int,
        cb_config: ContinuousBatchingConfig,
        gen_config: GenerationConfig | None = None,
        label: str | None = None,
        score_fn: Callable[[Any], float] | None = None,
    ) -> None:
        """Run one CB benchmark and record time, tokens, and peak memory."""

        gen_config = GenerationConfig() if gen_config is None else gen_config
        gen_config.max_new_tokens = max_new_tokens

        avg_input = sum(len(x) for x in data) / max(len(data), 1)
        entry = BenchmarkEntry(
            label=label or f"bench_{len(self.entries)}",
            num_samples=len(data),
            avg_input_tokens=avg_input,
            max_new_tokens=max_new_tokens,
            cb_config=_config_summary(cb_config),
            gen_config=_config_summary(gen_config),
        )
        print(f"\n[{entry.label}] samples={entry.num_samples} avg_in={avg_input:.1f} max_new={max_new_tokens}")

        # In DP, entries are sharded round-robin across ranks: entry i runs on rank i % dp_size.
        if self.dp_size > 1 and len(self.entries) % self.dp_size != self.global_rank:
            entry.error = f"Rank {self.global_rank} is not in charge of this entry"
            print(f"SKIPPED: {entry.error}")
            self.entries.append(entry)
            return None

        model = self._get_model()
        self.cleanup()

        try:
            outputs = model.generate_batch(
                inputs=data,
                generation_config=gen_config,
                continuous_batching_config=cb_config,
                progress_bar=True,
            )
            gen_start = min(out.created_time for out in outputs.values())
            gen_end = max(out.lifespan[1] for out in outputs.values())
            gen_time = gen_end - gen_start
            num_tokens = sum(len(out.generated_tokens) for out in outputs.values())

            entry.time_seconds = gen_time
            entry.num_tokens = num_tokens
            entry.throughput_tok_per_sec = num_tokens / gen_time if gen_time > 0 else 0.0
            entry.peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
            if score_fn is not None:
                entry.accuracy = score_fn(outputs)
            print(
                f"   {gen_time:.2f}s, {num_tokens} tokens, "
                f"{entry.throughput_tok_per_sec:.2f} tok/s, peak {entry.peak_memory_gb:.2f} GB"
                + (f", acc {entry.accuracy:.3f}" if entry.accuracy is not None else "")
            )
        except Exception as e:
            entry.error = str(e)
            print(f"   ERROR: {e}")

        self.entries.append(entry)
        self.cleanup()

    # Persistence
    def save(self, name: str) -> Path:
        """Save all entries to a timestamped JSON file keyed by name."""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        filename = RESULTS_DIR / f"{name}__{int(time.time())}.json"
        payload = {
            "model_id": self.model_id,
            "attn_impl": self.attn_impl,
            "entries": [asdict(e) for e in self.entries],
        }
        with open(filename, "a") as f:
            json.dump(payload, f, indent=2)
        print(f"\nResults saved to {filename}")
        return filename

    @classmethod
    def load_most_recent(cls, name: str) -> "BenchmarkResults":
        """Load the most recent JSON file matching name."""
        candidates = sorted(RESULTS_DIR.glob(f"{name}__*.json"))
        if not candidates:
            raise FileNotFoundError(f"No baseline with name '{name}' in {RESULTS_DIR}")
        data = json.loads(candidates[-1].read_text())
        instance = cls(
            model_id=data.get("model_id"),
            attn_impl=data.get("attn_impl"),
        )
        instance.entries = [BenchmarkEntry(**e) for e in data["entries"]]
        print(f"Loaded baseline from {candidates[-1]}")
        return instance

    # Display
    def print_summary(self) -> None:
        if self.dp_size > 1:
            print("-" * 80, f"RANK {self.global_rank}", "-" * 80, sep="\n")
        rows = [
            {
                "label": e.label,
                "samples": e.num_samples,
                "avg_in": f"{e.avg_input_tokens:.1f}",
                "max_new": e.max_new_tokens,
                "time (s)": _fmt(e.time_seconds, ".2f"),
                "tokens": _fmt(e.num_tokens, "d"),
                "tok/s": _fmt(e.throughput_tok_per_sec, ".2f", "ERROR"),
                "mem (GB)": _fmt(e.peak_memory_gb, ".2f"),
                "acc": _fmt(e.accuracy, ".3f", "-"),
            }
            for e in self.entries
        ]
        print("\n" + tabulate(rows, headers="keys", tablefmt="github"))

    def compare_to(self, baseline: "BenchmarkResults") -> None:
        """Print a side-by-side throughput comparison against a baseline run."""
        base_tps = {e.label: e.throughput_tok_per_sec for e in baseline.entries}

        def diff(cur: float | None, base: float | None) -> str:
            if cur is None or not base:
                return "N/A"
            return f"{(cur - base) / base * 100:+.1f}%"

        rows = [
            {
                "label": e.label,
                "baseline (tok/s)": _fmt(base_tps.get(e.label), ".2f", "N/A"),
                "current (tok/s)": _fmt(e.throughput_tok_per_sec, ".2f", e.error or "N/A"),
                "diff": diff(e.throughput_tok_per_sec, base_tps.get(e.label)),
            }
            for e in self.entries
        ]
        print(f"\nComparison against baseline (model={baseline.model_id}):")
        print(tabulate(rows, headers="keys", tablefmt="github"))


# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None, help="Name of the benchmark run (for saving).")
    parser.add_argument("--compare-to", type=str, default=None, help="Name of a previous run to compare against.")
    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--attn", type=str, default="kernels-community/flash-attn3")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size (1 = no TP).")
    parser.add_argument("--dp-size", type=int, default=1, help="Data parallel size (1 = no DP).")
    parser.add_argument(
        "--rollouts-lengths",
        "-rl",
        type=int,
        nargs="+",
        help="If this is specified, only the rollouts benchmarks run, with the given sizes (in tokens).",
    )
    args = parser.parse_args()

    results = BenchmarkResults(model_id=args.model_id, attn_impl=args.attn, tp_size=args.tp_size, dp_size=args.dp_size)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, padding_side="left")

    if args.rollouts_lengths is not None:
        rollouts_only = True
        rollout_sizes = args.rollouts_lengths
    else:
        rollouts_only = False
        rollout_sizes = [1024, 2048, 4096, 8192, 16384]

    if not rollouts_only:
        # GSM8K benchmarks (256 max new tokens) — gsm8k_platinum dataset, 8-shot, lighteval extractive_match
        gsm8k_data, gsm8k_score_fn = get_tokenized_gsm8k(tokenizer)

        ## No options
        results.add_benchmark(
            data=gsm8k_data,
            max_new_tokens=256,
            cb_config=ContinuousBatchingConfig(),
            gen_config=GenerationConfig(eos_token_id=-1),
            label="gsm8k_default",
            score_fn=gsm8k_score_fn,
        )

        ## With sampling. Recommended chat sampling (T=0.6, top_p=0.9), low enough that math reasoning isn't derailed
        results.add_benchmark(
            data=gsm8k_data,
            max_new_tokens=256,
            cb_config=ContinuousBatchingConfig(),
            gen_config=GenerationConfig(eos_token_id=-1, do_sample=True, temperature=0.6, top_p=0.9),
            label="gsm8k_sampling",
            score_fn=gsm8k_score_fn,
        )

        ## With compile
        results.add_benchmark(
            data=gsm8k_data,
            max_new_tokens=256,
            cb_config=ContinuousBatchingConfig(use_default_compile_configs=True),
            gen_config=GenerationConfig(eos_token_id=-1),
            label="gsm8k_compile",
            score_fn=gsm8k_score_fn,
        )

        ## No decode fast path
        results.add_benchmark(
            data=gsm8k_data,
            max_new_tokens=256,
            cb_config=ContinuousBatchingConfig(max_blocks_per_request=0),
            gen_config=GenerationConfig(eos_token_id=-1),
            label="gsm8k_no_fast_decode",
            score_fn=gsm8k_score_fn,
        )

        ## Bare-bones CB config
        results.add_benchmark(
            data=gsm8k_data,
            max_new_tokens=256,
            cb_config=ContinuousBatchingConfig(
                max_blocks_per_request=0, use_async_batching=False, use_cuda_graph=False
            ),
            gen_config=GenerationConfig(eos_token_id=-1),
            label="gsm8k_bare_bones",
            score_fn=gsm8k_score_fn,
        )

        # IFEval: 0-shot chat prompts; uses real EOS so instruction-following metrics see the model's natural stop.
        ifeval_data, ifeval_score_fn = get_tokenized_ifeval(tokenizer)
        results.add_benchmark(
            data=ifeval_data,
            max_new_tokens=1280,
            cb_config=ContinuousBatchingConfig(),
            label="ifeval_default",
            score_fn=ifeval_score_fn,
        )

        # Raw benchmarks (various options)

        ## Few blocks — tight cache pressure
        results.add_benchmark(
            data=get_random_data(batch_size=20, num_tokens=256),
            max_new_tokens=256,
            cb_config=ContinuousBatchingConfig(num_blocks=16),
            gen_config=GenerationConfig(eos_token_id=-1),
            label="few_blocks",
        )

        ## Multiple return sequences (sampling + parallel decoding)
        results.add_benchmark(
            data=get_random_data(batch_size=50, num_tokens=256),
            max_new_tokens=256,
            cb_config=ContinuousBatchingConfig(),
            gen_config=GenerationConfig(eos_token_id=-1, do_sample=True, num_return_sequences=8),
            label="multi_return_seq",
        )

    ## RL rollouts: small batch, growing generation lengths
    for length in rollout_sizes:
        results.add_benchmark(
            data=get_random_data(batch_size=32, num_tokens=256),
            max_new_tokens=length,
            cb_config=ContinuousBatchingConfig(use_default_compile_configs=True),
            gen_config=GenerationConfig(eos_token_id=-1),
            label=f"rollouts_{length}",
        )

    # Post processing and display. Only on rank 0 in TP runs to avoid duplicate output / file writes.
    is_rank_zero = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    if is_rank_zero:
        results.print_summary()
        if cli_args.compare_to:
            baseline = BenchmarkResults.load_most_recent(cli_args.compare_to)
            results.compare_to(baseline=baseline)
        if cli_args.name:
            results.save(cli_args.name)
