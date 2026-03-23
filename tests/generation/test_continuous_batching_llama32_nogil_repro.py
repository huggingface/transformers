# Copyright 2026 The HuggingFace Team Inc.
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

import os
import sys
import sysconfig
import threading
import time
import traceback
import unittest
from dataclasses import dataclass

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, ContinuousBatchingConfig, GenerationConfig
from transformers.generation.continuous_batching.requests import GenerationOutput, RequestStatus
from transformers.utils import is_flash_attn_2_available, is_kernels_available


# This repro must be started with no-GIL settings already applied at interpreter startup, for example:
#   PYTHON_GIL=0 /root/vm314t/bin/python -m unittest <test module>
# Setting this env var inside this script is too late for extension imports such as `tokenizers`, which can
# re-enable the GIL during import unless the interpreter itself starts with `PYTHON_GIL=0`.


MODEL_PATH = "/monster/data/model/Llama-3.2-1B-Instruct" # FIXME
DEVICES = ("cuda:0", "cuda:1")
MAX_NEW_TOKENS = 512
REQUEST_TIMEOUT_SECONDS = 600


def _nogil_is_enabled() -> tuple[bool, str]:
    if os.environ.get("PYTHON_GIL") != "0":
        return False, "Launch this test with PYTHON_GIL=0 before Python starts."

    gil_flag = getattr(sys.flags, "gil", None)
    if gil_flag is not None:
        return gil_flag == 0, f"sys.flags.gil={gil_flag}"

    is_gil_enabled = getattr(sys, "_is_gil_enabled", None)
    if callable(is_gil_enabled):
        gil_enabled = is_gil_enabled()
        if gil_enabled:
            return (
                False,
                "sys._is_gil_enabled()=True. On CPython free-threading builds, start the interpreter with "
                "PYTHON_GIL=0 or -Xgil=0 so the GIL stays disabled.",
            )
        return not gil_enabled, f"sys._is_gil_enabled()={gil_enabled}"

    py_gil_disabled = sysconfig.get_config_var("Py_GIL_DISABLED")
    if py_gil_disabled == 1:
        return True, "Detected a free-threaded Python build via Py_GIL_DISABLED=1."

    abi_flags = getattr(sys, "abiflags", "")
    if "t" in abi_flags:
        return True, f"Detected free-threaded ABI flags: {abi_flags!r}"

    return False, "Could not verify that this interpreter has the GIL disabled."


def _wait_for_terminal_output(
    manager, request_id: str, timeout_seconds: float = REQUEST_TIMEOUT_SECONDS
) -> GenerationOutput:
    deadline = time.perf_counter() + timeout_seconds
    last_result = None

    while time.perf_counter() < deadline:
        result = manager.get_result(request_id=request_id, timeout=0.5)
        if result is None:
            if not manager.is_running():
                break
            continue

        last_result = result
        if result.status in (RequestStatus.FINISHED, RequestStatus.FAILED):
            return result

    if last_result is not None:
        raise TimeoutError(
            f"Timed out waiting for terminal output for {request_id}. Last status was {last_result.status.name}."
        )
    raise TimeoutError(f"Timed out waiting for any output for {request_id}.")


@dataclass
class WorkerResult:
    device: str
    request_id: str
    output: GenerationOutput
    decoded_text: str


class ContinuousBatchingLlama32NogilReproTest(unittest.TestCase):
    def test_two_thread_llama32_continuous_batching_cuda_graph_repro(self) -> None:
        """Repro for two concurrent continuous-batching managers under a no-GIL interpreter."""
        nogil_enabled, nogil_reason = _nogil_is_enabled()
        self.assertTrue(
            nogil_enabled,
            f"This repro requires a no-GIL interpreter with GIL disabled. Details: {nogil_reason}",
        )
        self.assertEqual(os.environ.get("PYTHON_GIL"), "0")
        self.assertTrue(torch.cuda.is_available(), "This repro requires CUDA.")
        self.assertGreaterEqual(torch.cuda.device_count(), 2, "This repro requires at least 2 CUDA devices.")
        self.assertTrue(os.path.isdir(MODEL_PATH), f"Model path not found: {MODEL_PATH}")
        self.assertTrue(
            is_flash_attn_2_available() or is_kernels_available(),
            "This repro requires Flash Attention 2 support.",
        )

        barrier = threading.Barrier(len(DEVICES) + 1)
        results: dict[str, WorkerResult] = {}
        failures: dict[str, str] = {}

        def worker(device: str, prompt: str) -> None:
            try:
                torch.cuda.set_device(device)

                tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")
                if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token

                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_PATH,
                    attn_implementation="paged|flash_attention_2",
                    torch_dtype="auto",
                )
                model = model.to(device).eval()
                self.assertEqual(model.config._attn_implementation, "paged|flash_attention_2")

                generation_config = GenerationConfig(
                    do_sample=False,
                    eos_token_id=-1,
                    max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=tokenizer.pad_token_id,
                )
                continuous_batching_config = ContinuousBatchingConfig(
                    block_size=32,
                    num_blocks=64,
                    max_batch_tokens=1024,
                    allow_block_sharing=False,
                    use_async_batching=False,
                    use_cuda_graph=True,
                )

                messages = [{"role": "user", "content": prompt}]
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=False,
                )[0].tolist()

                with model.continuous_batching_context_manager(
                    generation_config=generation_config,
                    continuous_batching_config=continuous_batching_config,
                ) as manager:
                    self.assertEqual(manager.model.config._attn_implementation, "paged|flash_attention_2")

                    barrier.wait()

                    request_id = manager.add_request(
                        input_ids=input_ids,
                        max_new_tokens=MAX_NEW_TOKENS,
                        streaming=False,
                        eos_token_id=-1,
                    )
                    output = _wait_for_terminal_output(manager, request_id)

                decoded_text = tokenizer.decode(output.generated_tokens, skip_special_tokens=True)
                self.assertEqual(output.status, RequestStatus.FINISHED)
                self.assertIsNone(output.error, output.error)
                self.assertEqual(len(output.generated_tokens), MAX_NEW_TOKENS)
                self.assertTrue(decoded_text.strip(), f"{device} produced empty decoded output.")

                results[device] = WorkerResult(
                    device=device,
                    request_id=request_id,
                    output=output,
                    decoded_text=decoded_text,
                )
            except Exception:
                failures[device] = traceback.format_exc()
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        prompts = {
            "cuda:0": "Write a long numbered outline about thread-local CUDA graph capture behavior. Keep going.",
            "cuda:1": "Write a long numbered outline about isolated continuous batching managers on separate GPUs. Keep going.",
        }
        threads = [threading.Thread(target=worker, args=(device, prompts[device]), name=f"repro-{device}") for device in DEVICES]

        for thread in threads:
            thread.start()

        barrier.wait()

        for thread in threads:
            thread.join(timeout=REQUEST_TIMEOUT_SECONDS)
            self.assertFalse(thread.is_alive(), f"Worker thread {thread.name} did not finish.")

        if failures:
            devices = ", ".join(sorted(failures))
            self.fail(f"Repro failed on {devices}:\n\n" + "\n".join(failures[device] for device in sorted(failures)))

        self.assertEqual(set(results), set(DEVICES), f"Expected results for {DEVICES}, got {sorted(results)}")
        for device in DEVICES:
            result = results[device]
            self.assertEqual(result.output.status, RequestStatus.FINISHED)
            self.assertEqual(len(result.output.generated_tokens), MAX_NEW_TOKENS)
            self.assertTrue(result.decoded_text.strip(), f"{device} produced empty decoded output.")
