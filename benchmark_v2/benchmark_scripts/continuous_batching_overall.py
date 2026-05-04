"""
Continuous batching overall benchmark suite.

Runs CB in-process across many configurations (GSM8K prompts and synthetic
data) and can compare throughput against a previously-saved run.
"""

import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import datasets
import torch
from tabulate import tabulate

from transformers import AutoModelForCausalLM, AutoTokenizer, ContinuousBatchingConfig, GenerationConfig


# Defaults
RESULTS_DIR = Path(__file__).parent.parent / "benchmark_results/cb_overall/"


# Data helpers
def get_tokenized_gms8k(tokenizer: AutoTokenizer) -> list[list[int]]:
    """Tokenize the GSM8K questions as chat prompts."""
    dataset = datasets.load_dataset("openai/gsm8k", "socratic", split="test")
    batched_inputs = []
    for item in dataset:
        messages = [{"role": "user", "content": item["question"]}]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True)  # type: ignore
        batched_inputs.append(inputs if isinstance(inputs, list) else inputs["input_ids"])
    return batched_inputs


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
    error: str | None = None


def _config_summary(cfg: Any) -> dict[str, Any]:
    """Extract a JSON-friendly summary of a dataclass/config object."""
    raw = cfg.to_dict() if hasattr(cfg, "to_dict") else cfg.__dict__
    return {k: v for k, v in raw.items() if isinstance(v, (int, float, str, bool, type(None)))}


class BenchmarkResults:
    """Holds all CB benchmark runs and the shared model they execute against."""

    def __init__(self, model_id: str, attn_impl: str):
        self.model_id = model_id
        self.attn_impl = attn_impl
        self.entries: list[BenchmarkEntry] = []

    def cleanup(self) -> None:
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()

    def _get_model(self) -> Any:
        model = None
        self.cleanup()
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, attn_implementation=self.attn_impl, device_map="auto"
        )
        model = model.eval()
        return model

    def add_benchmark(
        self,
        data: list[list[int]],
        max_new_tokens: int,
        cb_config: ContinuousBatchingConfig,
        gen_config: GenerationConfig | None = None,
        label: str | None = None,
    ) -> BenchmarkEntry:
        """Run one CB benchmark and record time, tokens, and peak memory."""

        gen_config = GenerationConfig() if gen_config is None else gen_config
        gen_config.max_new_tokens = max_new_tokens
        # Disable EOS so every request runs to max_new_tokens — consistent benchmarking.
        gen_config.eos_token_id = -1

        model = self._get_model()

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

        self.cleanup()

        try:
            outputs = model.generate_batch(
                inputs=data,
                generation_config=gen_config,
                continuous_batching_config=cb_config,
                progress_bar=False,
            )
            gen_start = min(out.created_time for out in outputs.values())
            gen_end = max(out.lifespan[1] for out in outputs.values())
            gen_time = gen_end - gen_start
            num_tokens = sum(len(out.generated_tokens) for out in outputs.values())

            entry.time_seconds = gen_time
            entry.num_tokens = num_tokens
            entry.throughput_tok_per_sec = num_tokens / gen_time if gen_time > 0 else 0.0
            entry.peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
            print(
                f"   {gen_time:.2f}s, {num_tokens} tokens, "
                f"{entry.throughput_tok_per_sec:.2f} tok/s, peak {entry.peak_memory_gb:.2f} GB"
            )
        except Exception as e:
            entry.error = str(e)
            print(f"   ERROR: {e}")

        self.entries.append(entry)
        self.cleanup()
        return entry

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
        filename.write_text(json.dumps(payload, indent=2))
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
        rows = [
            {
                "label": e.label,
                "samples": e.num_samples,
                "avg_in": f"{e.avg_input_tokens:.1f}",
                "max_new": e.max_new_tokens,
                "time (s)": f"{e.time_seconds:.2f}" if e.time_seconds is not None else "X",
                "tokens": e.num_tokens if e.num_tokens is not None else "X",
                "tok/s": f"{e.throughput_tok_per_sec:.2f}" if e.throughput_tok_per_sec is not None else "ERROR",
                "mem (GB)": f"{e.peak_memory_gb:.2f}" if e.peak_memory_gb is not None else "X",
            }
            for e in self.entries
        ]
        print("\n" + tabulate(rows, headers="keys", tablefmt="github"))

    def compare_to(self, baseline: "BenchmarkResults") -> None:
        """Print a side-by-side throughput comparison against a baseline run."""
        baseline_by_label = {e.label: e for e in baseline.entries}
        rows = []
        for e in self.entries:
            base = baseline_by_label.get(e.label)
            base_tp = base.throughput_tok_per_sec if base else None
            cur_tp = e.throughput_tok_per_sec
            if isinstance(base_tp, (int, float)) and isinstance(cur_tp, (int, float)) and base_tp > 0:
                diff_str = f"{(cur_tp - base_tp) / base_tp * 100:+.1f}%"
            else:
                diff_str = "N/A"
            rows.append(
                {
                    "label": e.label,
                    "baseline (tok/s)": f"{base_tp:.2f}" if isinstance(base_tp, (int, float)) else "N/A",
                    "current (tok/s)": (f"{cur_tp:.2f}" if isinstance(cur_tp, (int, float)) else (e.error or "N/A")),
                    "diff": diff_str,
                }
            )
        print(f"\nComparison against baseline (model={baseline.model_id}):")
        print(tabulate(rows, headers="keys", tablefmt="github"))


# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None, help="Name of the benchmark run (for saving).")
    parser.add_argument("--compare-to", type=str, default=None, help="Name of a previous run to compare against.")
    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--attn", type=str, default="kernels-community/flash-attn3")
    cli_args = parser.parse_args()

    results = BenchmarkResults(model_id=cli_args.model_id, attn_impl=cli_args.attn)

    # GSM8K benchmarks (256 max new tokens)

    tokenizer = AutoTokenizer.from_pretrained(cli_args.model_id, padding_side="left")
    gsm8k_data = get_tokenized_gms8k(tokenizer)

    ## No options
    results.add_benchmark(
        data=gsm8k_data,
        max_new_tokens=256,
        cb_config=ContinuousBatchingConfig(),
        label="gsm8k_default",
    )

    ## With sampling
    results.add_benchmark(
        data=gsm8k_data,
        max_new_tokens=256,
        cb_config=ContinuousBatchingConfig(),
        gen_config=GenerationConfig(do_sample=True),
        label="gsm8k_sampling",
    )

    ## With compile
    results.add_benchmark(
        data=gsm8k_data,
        max_new_tokens=256,
        cb_config=ContinuousBatchingConfig(use_default_compile_configs=True),
        label="gsm8k_compile",
    )

    ## No decode fast path
    results.add_benchmark(
        data=gsm8k_data,
        max_new_tokens=256,
        cb_config=ContinuousBatchingConfig(max_blocks_per_request=0),
        label="gsm8k_no_fast_decode",
    )

    # Raw benchmarks (synthetic data, variable max new tokens)

    ## RL rollouts: small batch, growing generation lengths
    for length in [1024, 2048, 4096, 8192, 16384]:
        results.add_benchmark(
            data=get_random_data(batch_size=32, num_tokens=256),
            max_new_tokens=length,
            cb_config=ContinuousBatchingConfig(use_default_compile_configs=True),
            label=f"rollouts_{length}",
        )

    ## Few blocks — tight cache pressure
    results.add_benchmark(
        data=get_random_data(batch_size=20, num_tokens=256),
        max_new_tokens=256,
        cb_config=ContinuousBatchingConfig(num_blocks=16),
        label="few_blocks",
    )

    ## Multiple return sequences (sampling + parallel decoding)
    results.add_benchmark(
        data=get_random_data(batch_size=50, num_tokens=256),
        max_new_tokens=256,
        cb_config=ContinuousBatchingConfig(),
        gen_config=GenerationConfig(do_sample=True, num_return_sequences=8),
        label="multi_return_seq",
    )

    # Post processing and display

    results.print_summary()

    if cli_args.compare_to:
        baseline = BenchmarkResults.load_most_recent(cli_args.compare_to)
        results.compare_to(baseline=baseline)

    if cli_args.name:
        results.save(cli_args.name)
