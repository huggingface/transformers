# Copyright 2026 The HuggingFace Team. All rights reserved.
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
import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from parameterized import parameterized

from transformers.testing_utils import require_torch, require_torch_greater_or_equal
from transformers.utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    is_torch_available,
    is_torch_distributed_available,
)


if is_torch_available():
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from safetensors.torch import load_file
    from safetensors.torch import save_file as safe_save_file

    from transformers import LlamaConfig, LlamaForCausalLM, MixtralConfig, MixtralForCausalLM
    from transformers.core_model_loading import Chunk, WeightConverter
    from transformers.distributed import DistributedConfig
    from transformers.distributed.checkpoint import (
        _load_model_checkpoint_distributed,
        consolidate_distributed_checkpoint,
        load_optimizer_distributed,
        save_optimizer_distributed,
    )

    if is_torch_distributed_available():
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
            get_optimizer_state_dict,
        )


PARALLEL_CONFIGS = {
    "tp": {"tp_size": 4},
    "fsdp": {"fsdp_size": 4},
    "tp_fsdp": {"tp_size": 2, "fsdp_size": 2},
}

# The distributed format remains sharded. The consolidated format uses the regular save_pretrained path.
CHECKPOINT_MODES = {"distributed": True, "consolidated": False}


@contextmanager
def _distributed_context(rank, directory):
    environment = {"RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": "4"}
    with patch.dict(os.environ, environment), patch("torch._C._get_accelerator", return_value=torch.device("cpu")):
        dist.init_process_group(
            "gloo",
            init_method=f"file://{directory}/rendezvous",
            rank=rank,
            world_size=4,
            timeout=timedelta(seconds=60),
        )
        try:
            yield
        finally:
            dist.destroy_process_group()


# Workers stay at module scope so multiprocessing.spawn can pickle them.
def _test_save_and_from_pretrained(rank, directory, source_config, destination_config, distributed_checkpoint):
    with _distributed_context(rank, directory):
        # Both models start from the same seed checkpoint so their weights match on every rank.
        reference = LlamaForCausalLM.from_pretrained(f"{directory}/seed")
        distributed = LlamaForCausalLM.from_pretrained(
            f"{directory}/seed", distributed_config=DistributedConfig(**source_config)
        )

        # Distributed checkpoint: `from_pretrained` detects it and loads it with DCP.
        # Consolidated checkpoint: `from_pretrained` loads it and converts it to a distributed model.
        # Either way, it is resharded when the destination configuration differs from the source one.
        distributed.save_pretrained(f"{directory}/saved", distributed_checkpoint=distributed_checkpoint)
        restored = LlamaForCausalLM.from_pretrained(
            f"{directory}/saved", distributed_config=DistributedConfig(**destination_config)
        )
        full_state_dict = get_model_state_dict(restored, options=StateDictOptions(full_state_dict=True))
        torch.testing.assert_close(full_state_dict, reference.state_dict())


def _test_save_sharded_model_files(rank, directory, source_config, max_shard_size):
    with _distributed_context(rank, directory):
        model = MixtralForCausalLM.from_pretrained(
            f"{directory}/seed", distributed_config=DistributedConfig(**source_config)
        )
        model.save_pretrained(f"{directory}/saved", max_shard_size=max_shard_size)


def _test_save_one_file_per_rank(rank, directory, source_config):
    with _distributed_context(rank, directory):
        model = MixtralForCausalLM.from_pretrained(
            f"{directory}/seed", distributed_config=DistributedConfig(**source_config)
        )
        # The tiny model is below the size where files are capped, so lower that limit.
        with patch("transformers.distributed.checkpoint._MIN_CAPPED_FILE_BYTES", 1):
            model.save_pretrained(f"{directory}/saved")


def _test_save_conversion_across_files(rank, directory):
    with _distributed_context(rank, directory):
        model = LlamaForCausalLM.from_pretrained(f"{directory}/seed", distributed_config=DistributedConfig(tp_size=4))
        # As if q, k and v had been loaded from a fused `qkv_proj`: the save fuses them back.
        model._weight_conversions = [
            WeightConverter(
                source_patterns="self_attn.qkv_proj.weight",
                target_patterns=["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"],
                operations=[Chunk(dim=0)],
            )
        ]
        # Files small enough to split q, k and v if the split were planned before reverting the conversion.
        model.save_pretrained(f"{directory}/saved", max_shard_size="1KB")


def _test_save_for_consolidation(rank, directory):
    with _distributed_context(rank, directory):
        model = MixtralForCausalLM.from_pretrained(
            f"{directory}/seed", distributed_config=DistributedConfig(tp_size=4)
        )
        model.save_pretrained(f"{directory}/checkpoint", distributed_checkpoint=True)


def _test_consolidate_on_every_rank(rank, directory):
    # A separate rendezvous file from the save.
    with _distributed_context(rank, f"{directory}/rendezvous_consolidate"):
        with patch("transformers.distributed.checkpoint.safe_save_file", wraps=safe_save_file) as save_file:
            consolidate_distributed_checkpoint(
                f"{directory}/checkpoint",
                f"{directory}/consolidated",
                max_shard_size="10KB",
                model=MixtralForCausalLM.from_pretrained(f"{directory}/seed"),
            )
        written = [os.path.basename(call.args[1]) for call in save_file.call_args_list]
        Path(f"{directory}/written_{rank}.json").write_text(json.dumps(written), encoding="utf-8")


def _test_save_and_push_to_hub(rank, directory):
    with _distributed_context(rank, directory):
        model = MixtralForCausalLM.from_pretrained(
            f"{directory}/seed", distributed_config=DistributedConfig(tp_size=4)
        )
        save_directory = os.path.join(directory, "checkpoint")
        output_dir = os.path.join(save_directory, "consolidated")
        with (
            patch("transformers.modeling_utils.hf_api") as api,
            patch(
                "transformers.distributed.mixin.consolidate_distributed_checkpoint",
                wraps=consolidate_distributed_checkpoint,
            ) as consolidate,
            patch("transformers.distributed.mixin.create_and_tag_model_card") as model_card,
            patch.object(model, "_upload_modified_files") as upload,
        ):
            api.return_value.create_repo.return_value.repo_id = "test/repo"
            model_card.return_value.save.side_effect = lambda path: Path(path).write_text(
                "# Test checkpoint", encoding="utf-8"
            )
            model.save_pretrained(
                save_directory,
                distributed_checkpoint=True,
                push_to_hub=True,
                max_shard_size="10KB",
                repo_id="test/repo",
                commit_message="Export DCP",
                token="test-token",
                create_pr=True,
                private=True,
            )
            # Every rank writes some of the consolidated files.
            consolidate.assert_called_once_with(save_directory, output_dir, max_shard_size="10KB", model=model)
            if rank == 0:
                api.return_value.create_repo.assert_called_once_with(
                    "test/repo", exist_ok=True, token="test-token", private=True
                )
                model_card.return_value.save.assert_called_once_with(os.path.join(output_dir, "README.md"))
                upload.assert_called_once_with(
                    output_dir,
                    "test/repo",
                    files_timestamps={},
                    commit_message="Export DCP",
                    token="test-token",
                    create_pr=True,
                )
            else:
                api.return_value.create_repo.assert_not_called()
                model_card.assert_not_called()
                upload.assert_not_called()


def _test_load_model_checkpoint_distributed(rank, directory):
    with _distributed_context(rank, directory):
        reference = LlamaForCausalLM.from_pretrained(f"{directory}/seed")
        model = LlamaForCausalLM.from_pretrained(
            f"{directory}/seed", distributed_config=DistributedConfig(tp_size=2, fsdp_size=2)
        )
        model.save_pretrained(f"{directory}/dcp", distributed_checkpoint=True)

        # Zero the weights so the check below only passes if the checkpoint was actually loaded.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        _load_model_checkpoint_distributed(model, f"{directory}/dcp")
        full_state_dict = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True))
        torch.testing.assert_close(full_state_dict, reference.state_dict())


def _test_optimizer_checkpoint(rank, directory):
    with _distributed_context(rank, directory):
        full_state_dict = StateDictOptions(full_state_dict=True)
        model = LlamaForCausalLM.from_pretrained(
            f"{directory}/seed", distributed_config=DistributedConfig(tp_size=2, fsdp_size=2)
        )
        optimizer = torch.optim.AdamW(model.parameters())
        for parameter in model.parameters():
            parameter.grad = parameter.detach().clone()
        optimizer.step()
        expected = get_optimizer_state_dict(model, optimizer, options=full_state_dict)
        save_optimizer_distributed(model, optimizer, f"{directory}/saved")

        if rank == 0:
            torch.save(expected, f"{directory}/expected.pt")

        # Load into the same layout and into a different one.
        checkpoint = f"{directory}/saved"
        for config in (DistributedConfig(tp_size=2, fsdp_size=2), DistributedConfig(tp_size=4)):
            restored = LlamaForCausalLM.from_pretrained(f"{directory}/seed", distributed_config=config)
            restored_optimizer = torch.optim.AdamW(restored.parameters())
            load_optimizer_distributed(restored, restored_optimizer, checkpoint)
            actual = get_optimizer_state_dict(restored, restored_optimizer, options=full_state_dict)
            torch.testing.assert_close(actual["state"], expected["state"])
            assert actual["param_groups"] == expected["param_groups"]


@require_torch
@unittest.skipUnless(
    is_torch_available() and dist.is_available() and dist.is_gloo_available(), "Requires distributed Gloo"
)
class DistributedUtilsTest(unittest.TestCase):
    def setUp(self):
        self.config = LlamaConfig(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
        )
        # Its fused experts are reverted to one tensor per expert on save, and its `lm_head` is tied.
        self.moe_config = MixtralConfig(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_local_experts=4,
            tie_word_embeddings=True,
        )

    @parameterized.expand(
        [
            (f"{source_name}_to_{destination_name}_{checkpoint_mode}", source, destination, distributed_checkpoint)
            for source_name, source in PARALLEL_CONFIGS.items()
            for destination_name, destination in PARALLEL_CONFIGS.items()
            for checkpoint_mode, distributed_checkpoint in CHECKPOINT_MODES.items()
        ]
    )
    def test_save_and_from_pretrained(self, _, source_config, destination_config, distributed_checkpoint):
        with tempfile.TemporaryDirectory() as directory:
            reference = LlamaForCausalLM(self.config)
            reference.save_pretrained(f"{directory}/seed")
            mp.spawn(
                _test_save_and_from_pretrained,
                args=(directory, source_config, destination_config, distributed_checkpoint),
                nprocs=4,
                join=True,
            )

            # Reload without a process group or distributed configuration.
            torch.testing.assert_close(
                LlamaForCausalLM.from_pretrained(f"{directory}/saved").state_dict(), reference.state_dict()
            )

    @parameterized.expand(PARALLEL_CONFIGS.items())
    def test_save_sharded_model_files(self, _, source_config):
        # The fused experts are reverted to one tensor per expert on save, the tied `lm_head` is dropped, and the
        # files outnumber the ranks.
        with tempfile.TemporaryDirectory() as directory:
            reference = MixtralForCausalLM(self.moe_config)
            reference.save_pretrained(f"{directory}/seed", max_shard_size="10KB")
            mp.spawn(_test_save_sharded_model_files, args=(directory, source_config, "10KB"), nprocs=4, join=True)

            # Same weight names and files as a non-distributed save.
            seed_index = json.loads(Path(f"{directory}/seed/{SAFE_WEIGHTS_INDEX_NAME}").read_text(encoding="utf-8"))
            saved_index = json.loads(Path(f"{directory}/saved/{SAFE_WEIGHTS_INDEX_NAME}").read_text(encoding="utf-8"))
            self.assertEqual(set(saved_index["weight_map"]), set(seed_index["weight_map"]))
            saved_files = set(saved_index["weight_map"].values())
            self.assertGreater(len(saved_files), 4)
            self.assertEqual(saved_files, {path.name for path in Path(f"{directory}/saved").glob("*.safetensors")})

            torch.testing.assert_close(
                MixtralForCausalLM.from_pretrained(f"{directory}/saved").state_dict(), reference.state_dict()
            )

    @parameterized.expand(PARALLEL_CONFIGS.items())
    def test_save_single_model_file(self, _, source_config):
        # Fewer files than ranks: all ranks write into the same file.
        with tempfile.TemporaryDirectory() as directory:
            reference = MixtralForCausalLM(self.moe_config)
            reference.save_pretrained(f"{directory}/seed")
            mp.spawn(_test_save_sharded_model_files, args=(directory, source_config, "50GB"), nprocs=4, join=True)

            saved_files = [path.name for path in Path(f"{directory}/saved").glob("*.safetensors")]
            self.assertEqual(saved_files, [SAFE_WEIGHTS_NAME])
            self.assertFalse(Path(f"{directory}/saved/{SAFE_WEIGHTS_INDEX_NAME}").exists())
            torch.testing.assert_close(
                MixtralForCausalLM.from_pretrained(f"{directory}/saved").state_dict(), reference.state_dict()
            )

    def test_save_conversion_across_files(self):
        # The weights reverted together by a conversion are gathered as a group, whatever files they are in.
        with tempfile.TemporaryDirectory() as directory:
            reference = LlamaForCausalLM(self.config)
            reference.save_pretrained(f"{directory}/seed")
            mp.spawn(_test_save_conversion_across_files, args=(directory,), nprocs=4, join=True)

            saved_index = json.loads(Path(f"{directory}/saved/{SAFE_WEIGHTS_INDEX_NAME}").read_text(encoding="utf-8"))
            saved = {}
            for filename in set(saved_index["weight_map"].values()):
                saved.update(load_file(f"{directory}/saved/{filename}"))
            expected = reference.state_dict()
            for layer in range(self.config.num_hidden_layers):
                prefix = f"model.layers.{layer}.self_attn"
                self.assertNotIn(f"{prefix}.q_proj.weight", saved)
                torch.testing.assert_close(
                    saved[f"{prefix}.qkv_proj.weight"],
                    torch.cat([expected[f"{prefix}.{name}_proj.weight"] for name in "qkv"]),
                )

    @parameterized.expand(PARALLEL_CONFIGS.items())
    def test_save_one_file_per_rank(self, _, source_config):
        # With the default `max_shard_size`, the files are capped to a `world_size`-th of the model: one file per rank.
        with tempfile.TemporaryDirectory() as directory:
            reference = MixtralForCausalLM(self.moe_config)
            reference.save_pretrained(f"{directory}/seed")
            mp.spawn(_test_save_one_file_per_rank, args=(directory, source_config), nprocs=4, join=True)

            saved_index = json.loads(Path(f"{directory}/saved/{SAFE_WEIGHTS_INDEX_NAME}").read_text(encoding="utf-8"))
            saved_files = set(saved_index["weight_map"].values())
            self.assertGreaterEqual(len(saved_files), 4)
            self.assertEqual(saved_files, {path.name for path in Path(f"{directory}/saved").glob("*.safetensors")})
            torch.testing.assert_close(
                MixtralForCausalLM.from_pretrained(f"{directory}/saved").state_dict(), reference.state_dict()
            )

    @parameterized.expand([("single_process", False), ("every_rank", True)])
    @require_torch_greater_or_equal("2.9")
    def test_consolidate_distributed_checkpoint_round_trip(self, _, on_every_rank):
        with tempfile.TemporaryDirectory() as directory:
            reference = MixtralForCausalLM(self.moe_config)
            reference.save_pretrained(f"{directory}/seed", max_shard_size="10KB")
            mp.spawn(_test_save_for_consolidation, args=(directory,), nprocs=4, join=True)

            output_dir = os.path.join(directory, "consolidated")
            if on_every_rank:
                os.makedirs(f"{directory}/rendezvous_consolidate")
                mp.spawn(_test_consolidate_on_every_rank, args=(directory,), nprocs=4, join=True)
            else:
                consolidate_distributed_checkpoint(
                    os.path.join(directory, "checkpoint"),
                    output_dir,
                    max_shard_size="10KB",
                    model=MixtralForCausalLM.from_pretrained(f"{directory}/seed"),
                )

            # Same weight names as a non-distributed save (one tensor per expert, no tied `lm_head`), split in files
            # of at most `max_shard_size`.
            seed_index = json.loads(Path(f"{directory}/seed/{SAFE_WEIGHTS_INDEX_NAME}").read_text(encoding="utf-8"))
            index = json.loads(Path(output_dir, SAFE_WEIGHTS_INDEX_NAME).read_text(encoding="utf-8"))
            self.assertEqual(set(index["weight_map"]), set(seed_index["weight_map"]))
            files = set(index["weight_map"].values())
            self.assertGreater(len(files), 4)
            if on_every_rank:
                # Every file is written exactly once, and every rank writes some of them.
                written = [
                    json.loads(Path(f"{directory}/written_{rank}.json").read_text(encoding="utf-8"))
                    for rank in range(4)
                ]
                self.assertTrue(all(written))
                self.assertEqual(sorted(sum(written, [])), sorted(files))
            restored = MixtralForCausalLM.from_pretrained(
                output_dir, config=self.moe_config, generation_config=reference.generation_config
            )
            torch.testing.assert_close(restored.state_dict(), reference.state_dict())

    @require_torch_greater_or_equal("2.9")
    def test_save_and_push_to_hub(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = MixtralForCausalLM(self.moe_config)
            reference.save_pretrained(f"{directory}/seed", max_shard_size="10KB")
            mp.spawn(_test_save_and_push_to_hub, args=(directory,), nprocs=4, join=True)

            checkpoint_dir = Path(directory) / "checkpoint"
            output_dir = checkpoint_dir / "consolidated"
            self.assertTrue(list(checkpoint_dir.glob("shard-*.safetensors")))
            self.assertFalse(list(output_dir.glob("shard-*.safetensors")))
            self.assertTrue((output_dir / "config.json").is_file())
            self.assertTrue((output_dir / "generation_config.json").is_file())
            self.assertTrue((output_dir / "README.md").is_file())

            # The uploaded checkpoint has the original weight names (one tensor per expert, no tied `lm_head`) and is
            # split in files of at most `max_shard_size`, like a non-distributed save.
            seed_index = json.loads(Path(f"{directory}/seed/{SAFE_WEIGHTS_INDEX_NAME}").read_text(encoding="utf-8"))
            saved_index = json.loads((output_dir / SAFE_WEIGHTS_INDEX_NAME).read_text(encoding="utf-8"))
            self.assertEqual(set(saved_index["weight_map"]), set(seed_index["weight_map"]))
            self.assertGreater(len(set(saved_index["weight_map"].values())), 1)

            for path in (checkpoint_dir, output_dir):
                torch.testing.assert_close(
                    MixtralForCausalLM.from_pretrained(path).state_dict(), reference.state_dict()
                )

    def test_load_model_checkpoint_distributed(self):
        with tempfile.TemporaryDirectory() as directory:
            LlamaForCausalLM(self.config).save_pretrained(f"{directory}/seed")
            mp.spawn(_test_load_model_checkpoint_distributed, args=(directory,), nprocs=4, join=True)

    def test_optimizer_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            LlamaForCausalLM(self.config).save_pretrained(f"{directory}/seed")
            mp.spawn(_test_optimizer_checkpoint, args=(directory,), nprocs=4, join=True)

            # Reload without a process group or distributed configuration.
            restored = LlamaForCausalLM.from_pretrained(f"{directory}/seed")
            restored_optimizer = torch.optim.AdamW(restored.parameters())
            checkpoint = f"{directory}/saved"
            load_optimizer_distributed(restored, restored_optimizer, checkpoint)
            actual = get_optimizer_state_dict(restored, restored_optimizer)
            expected = torch.load(f"{directory}/expected.pt")
            torch.testing.assert_close(actual["state"], expected["state"])
            assert actual["param_groups"] == expected["param_groups"]


if __name__ == "__main__":
    unittest.main()
