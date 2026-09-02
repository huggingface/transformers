# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

"""Training overfit tester mixin for model tests."""

import logging
import time
from abc import ABC, abstractmethod

import torch

from transformers import set_seed
from transformers.testing_utils import Colors, build_cpu_memory_monitor, init_test_logger, is_training_test


logger = logging.getLogger("transformers.training_test")


class TrainingTesterMixin(ABC):
    """
    Mixin for training overfit tests. Add to model test classes alongside ModelTesterMixin.

    The model_tester (e.g., CausalLMModelTester) already provides:
      - get_config() -> tiny model config
      - prepare_config_and_inputs_for_common() -> config + input dict
      - causal_lm_class, base_model_class, etc.

    This mixin adds training-specific tests using that infrastructure.
    """

    # ============================================================
    # Training hyperparameters
    # ============================================================
    training_overfit_steps: int = 300
    training_overfit_batch_size: int = 2
    training_overfit_learning_rate: float = 1e-3
    training_overfit_seq_length: int = 64
    training_overfit_log_freq: int = 10

    # Loss reduction and grad norm reduction thresholds for passing the test (i.e 95% reduction)
    training_loss_reduction_threshold: float = 0.9
    training_grad_norm_reduction_threshold: float = 0.9

    @property
    @abstractmethod
    def model_tester(self):
        """The model tester instance (e.g., CausalLMModelTester)."""
        ...

    # ============================================================
    # Modality detection
    # ============================================================
    def _get_model_modality(self) -> str:
        """Detect the modality of the model based on its input signature."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        if "input_ids" in inputs_dict:
            return "text"
        elif "pixel_values" in inputs_dict:
            return "image"
        elif "input_features" in inputs_dict or "input_values" in inputs_dict:
            return "audio"
        else:
            raise ValueError(f"Unknown modality: {inputs_dict}")

    # ============================================================
    # Training data creation for each modality
    # ============================================================
    def _create_text_training_batch(
        self,
        batch_size: int,
        seq_length: int,
        vocab_size: int,
    ) -> dict[str, torch.Tensor]:
        """Create a simple text batch without needing a tokenizer."""
        # Create a deterministic sequence (not random, so model can learn it)
        pattern = list(range(1, min(20, vocab_size)))  # tokens 1-19
        num_repeats = (seq_length // len(pattern)) + 1
        tokens = (pattern * num_repeats)[:seq_length]
        input_ids = torch.tensor([tokens] * batch_size, dtype=torch.long)
        return {"input_ids": input_ids, "labels": input_ids.clone()}

    def _create_image_training_batch(
        self,
        batch_size: int,
        num_channels: int,
        height: int,
        width: int,
    ) -> dict[str, torch.Tensor]:
        """Create fixed batch for image models using a deterministic pattern."""
        pass

    def _create_audio_training_batch(
        self,
        batch_size: int,
        audio_length: int,
        feature_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Create fixed batch for audio models using a deterministic waveform."""
        pass

    def _decode_text_tokens(self, tokens: list[int], max_display: int = 40) -> str:
        """Decode tokens to readable string (maps token IDs to letters: 1->a, 2->b, etc.)."""
        decoded = "".join(chr(ord("a") + (t - 1) % 26) for t in tokens)
        if len(decoded) > max_display:
            return f"'{decoded[:max_display]}...'"
        return f"'{decoded}'"

    def _get_trainable_model_class(self):
        """Get the model class to use for training (prefers *ForCausalLM, *ForSequenceClassification, etc.)."""
        # Prefer model classes with a head (for computing loss)
        if hasattr(self.model_tester, "causal_lm_class") and self.model_tester.causal_lm_class is not None:
            return self.model_tester.causal_lm_class
        if (
            hasattr(self.model_tester, "sequence_classification_class")
            and self.model_tester.sequence_classification_class is not None
        ):
            return self.model_tester.sequence_classification_class
        # Fall back to first model class
        return self.all_model_classes[0]

    @is_training_test
    def test_training_overfit(self):
        """Test that a tiny model can overfit on a fixed batch."""
        # Initialize logging and memory monitoring
        init_test_logger()
        memory_monitor = build_cpu_memory_monitor(logger)

        logger.info("=" * 70)
        logger.info(f"Starting test: {self._testMethodName}")
        logger.info("=" * 70)

        # Skip if model doesn't support training
        if not getattr(self.model_tester, "is_training", True):
            logger.info(f"{Colors.YELLOW}Skipping: Model tester not configured for training tests{Colors.RESET}")
            self.skipTest("Model tester not configured for training tests")

        # Configuration
        logger.info(f"{Colors.BOLD}Job Configuration:{Colors.RESET}")
        logger.info(f"  {Colors.CYAN}total_steps:{Colors.RESET} {self.training_overfit_steps}")
        logger.info(f"  {Colors.CYAN}batch_size:{Colors.RESET} {self.training_overfit_batch_size}")
        logger.info(f"  {Colors.CYAN}learning_rate:{Colors.RESET} {self.training_overfit_learning_rate}")
        logger.info(f"  {Colors.CYAN}seq_length:{Colors.RESET} {self.training_overfit_seq_length}")
        logger.info(f"  {Colors.CYAN}log_freq:{Colors.RESET} {self.training_overfit_log_freq}")
        logger.info(f"  {Colors.CYAN}device:{Colors.RESET} cpu")

        set_seed(42)

        logger.info("-" * 70)
        logger.info(f"{Colors.BOLD}Building model{Colors.RESET}")
        load_start = time.perf_counter()

        # Get tiny config from existing infrastructure
        config = self.model_tester.get_config()

        model_class = self._get_trainable_model_class()
        model = model_class(config)
        model.train()

        load_time = time.perf_counter() - load_start
        logger.info(f"Model loaded in {Colors.GREEN}{load_time:.3f}s{Colors.RESET}")

        # Log model architecture
        # TODO(3outeille): make sure if there is other parameters to log
        logger.info(f"{Colors.BOLD}Model Architecture:{Colors.RESET}")
        logger.info(f"  {Colors.CYAN}model_class:{Colors.RESET} {model_class.__name__}")
        if hasattr(config, "hidden_size"):
            logger.info(f"  {Colors.CYAN}hidden_size:{Colors.RESET} {config.hidden_size}")
        if hasattr(config, "num_hidden_layers"):
            logger.info(f"  {Colors.CYAN}num_hidden_layers:{Colors.RESET} {config.num_hidden_layers}")
        if hasattr(config, "num_attention_heads"):
            logger.info(f"  {Colors.CYAN}num_attention_heads:{Colors.RESET} {config.num_attention_heads}")
        if hasattr(config, "num_key_value_heads"):
            logger.info(f"  {Colors.CYAN}num_key_value_heads:{Colors.RESET} {config.num_key_value_heads}")
        if hasattr(config, "intermediate_size"):
            logger.info(f"  {Colors.CYAN}intermediate_size:{Colors.RESET} {config.intermediate_size}")
        if hasattr(config, "vocab_size"):
            logger.info(f"  {Colors.CYAN}vocab_size:{Colors.RESET} {config.vocab_size}")
        if hasattr(config, "num_experts"):
            logger.info(f"  {Colors.CYAN}num_experts:{Colors.RESET} {config.num_experts}")
        if hasattr(config, "num_experts_per_tok"):
            logger.info(f"  {Colors.CYAN}num_experts_per_tok:{Colors.RESET} {config.num_experts_per_tok}")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"{Colors.CYAN}Model size:{Colors.RESET} {Colors.BRIGHT_GREEN}{total_params:,}{Colors.RESET} total parameters"
        )
        logger.info(
            f"{Colors.CYAN}Trainable parameters:{Colors.RESET} {Colors.BRIGHT_GREEN}{trainable_params:,}{Colors.RESET}"
        )

        # Memory after model load
        mem_stats = memory_monitor.get_stats()
        logger.info(
            f"{Colors.MAGENTA}Memory after model load:{Colors.RESET} {mem_stats.rss_gib:.2f} GiB ({mem_stats.rss_pct:.1f}%)"
        )

        logger.info("-" * 70)
        logger.info(f"{Colors.BOLD}Creating fixed batch{Colors.RESET}")

        modality = self._get_model_modality()
        logger.info(f"{Colors.CYAN}Detected modality:{Colors.RESET} {modality}")
        _, sample_inputs = self.model_tester.prepare_config_and_inputs_for_common()

        if modality == "text":
            # For text models, we need a tokenizer - use a simple one or create fake tokens
            batch = self._create_text_training_batch(
                batch_size=self.training_overfit_batch_size,
                seq_length=self.training_overfit_seq_length,
                vocab_size=config.vocab_size,
            )
            logger.info(f"{Colors.CYAN}Training pattern:{Colors.RESET} Repeating token sequence (1-19)")
        else:
            raise ValueError(f"Modality {modality} not supported yet for training overfit")

        tokens_per_batch = self.training_overfit_batch_size * self.training_overfit_seq_length
        logger.info(f"  {Colors.CYAN}batch_size:{Colors.RESET} {self.training_overfit_batch_size}")
        logger.info(f"  {Colors.CYAN}seq_length:{Colors.RESET} {self.training_overfit_seq_length}")
        logger.info(f"  {Colors.CYAN}tokens_per_batch:{Colors.RESET} {tokens_per_batch:,}")
        logger.info(f"{Colors.DIM}Using same fixed batch every step (deterministic overfitting){Colors.RESET}")

        logger.info("-" * 70)
        logger.info(f"{Colors.BOLD}Building optimizer{Colors.RESET}")

        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.training_overfit_learning_rate, weight_decay=0.0, betas=(0.9, 0.999)
        )
        logger.info(f"{Colors.CYAN}Optimizer:{Colors.RESET} Adam")
        logger.info(f"  {Colors.CYAN}learning_rate:{Colors.RESET} {self.training_overfit_learning_rate}")
        logger.info(f"  {Colors.CYAN}weight_decay:{Colors.RESET} 0.0")
        logger.info(f"  {Colors.CYAN}betas:{Colors.RESET} (0.9, 0.999)")

        # Training Loop
        logger.info("-" * 70)
        logger.info("Training starts at step 1")

        initial_loss = None
        final_loss = None
        initial_grad_norm = None
        final_grad_norm = None
        training_start = time.perf_counter()
        memory_monitor.reset_peak_stats()

        for step in range(1, self.training_overfit_steps + 1):
            step_start = time.perf_counter()

            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss

            if initial_loss is None:
                initial_loss = loss.item()
            final_loss = loss.item()

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if initial_grad_norm is None:
                initial_grad_norm = grad_norm.item()
            final_grad_norm = grad_norm.item()

            optimizer.step()

            step_time = time.perf_counter() - step_start

            # Log at frequency
            if step == 1 or step % self.training_overfit_log_freq == 0 or step == self.training_overfit_steps:
                tokens_per_sec = tokens_per_batch / step_time
                mem_stats = memory_monitor.get_stats()
                logger.info(
                    f"{Colors.CYAN}step:{Colors.RESET} {step}  "
                    f"{Colors.GREEN}loss:{Colors.RESET} {loss.item():7.4f}  "
                    f"{Colors.YELLOW}grad_norm:{Colors.RESET} {grad_norm.item():6.4f}  "
                    f"{Colors.MAGENTA}memory:{Colors.RESET} {mem_stats.rss_gib:.2f}GiB({mem_stats.rss_pct:.1f}%)  "
                    f"{Colors.BLUE}tok/s:{Colors.RESET} {tokens_per_sec:,.0f}  "
                    f"{Colors.DIM}step_time:{Colors.RESET} {step_time:.3f}s"
                )

        training_time = time.perf_counter() - training_start

        # Training Summary
        total_tokens = self.training_overfit_steps * tokens_per_batch
        logger.info("-" * 70)
        logger.info(f"{Colors.BOLD}Training completed{Colors.RESET}")
        logger.info(f"Total training time: {training_time:.2f}s")
        logger.info(f"Total steps: {self.training_overfit_steps}")
        logger.info(f"Total tokens seen: {total_tokens:,}")
        logger.info(f"Average tokens/sec: {total_tokens / training_time:,.0f}")

        # Memory summary
        mem_stats = memory_monitor.get_stats()
        logger.info(f"{Colors.BOLD}Memory usage:{Colors.RESET}")
        logger.info(
            f"  {Colors.CYAN}current_rss:{Colors.RESET} {mem_stats.rss_gib:.2f} GiB ({mem_stats.rss_pct:.1f}%)"
        )
        logger.info(
            f"  {Colors.CYAN}peak_rss:{Colors.RESET} {mem_stats.peak_rss_gib:.2f} GiB ({mem_stats.peak_rss_pct:.1f}%)"
        )
        logger.info(
            f"  {Colors.CYAN}available:{Colors.RESET} {mem_stats.available_gib:.2f} GiB / {mem_stats.total_gib:.2f} GiB"
        )

        # Loss analysis
        loss_reduction = (initial_loss - final_loss) / initial_loss * 100
        logger.info(f"{Colors.BOLD}Loss metrics:{Colors.RESET}")
        logger.info(f"  {Colors.CYAN}initial_loss:{Colors.RESET} {initial_loss:.4f}")
        logger.info(f"  {Colors.CYAN}final_loss:{Colors.RESET} {final_loss:.4f}")
        logger.info(f"  {Colors.CYAN}loss_reduction:{Colors.RESET} {loss_reduction:.1f}%")

        # Grad norm analysis
        grad_norm_reduction = (initial_grad_norm - final_grad_norm) / initial_grad_norm * 100
        logger.info(f"{Colors.BOLD}Grad norm metrics:{Colors.RESET}")
        logger.info(f"  {Colors.CYAN}initial_grad_norm:{Colors.RESET} {initial_grad_norm:.4f}")
        logger.info(f"  {Colors.CYAN}final_grad_norm:{Colors.RESET} {final_grad_norm:.4f}")
        logger.info(f"  {Colors.CYAN}grad_norm_reduction:{Colors.RESET} {grad_norm_reduction:.1f}%")

        # Generation Test (only for text/causal LM models)
        # TODO(3outeille): handle audio and generate
        generation_matches = None
        if modality == "text" and hasattr(model, "generate"):
            logger.info("-" * 70)
            logger.info(f"{Colors.BOLD}Testing generation{Colors.RESET}")

            model.eval()

            # Get the expected token sequence (same pattern used in training)
            expected_tokens = batch["input_ids"][0].tolist()

            # Use first token as prompt
            prompt_ids = torch.tensor([[expected_tokens[0]]], dtype=torch.long)
            num_tokens_to_generate = len(expected_tokens) - 1

            logger.info(f"Prompt: {self._decode_text_tokens([expected_tokens[0]])}")

            model_type = getattr(config, "model_type", "")
            use_cache = model_type == "recurrent_gemma"
            if use_cache:
                logger.info("Only RecurrentGemmaModel is using use_cache=True. Other models run with use_cache=False")

            with torch.no_grad():
                generated_ids = model.generate(
                    prompt_ids,
                    max_new_tokens=num_tokens_to_generate,
                    do_sample=False,
                    pad_token_id=config.pad_token_id if hasattr(config, "pad_token_id") else 0,
                    eos_token_id=0,
                    use_cache=use_cache,
                )

            generated_tokens = generated_ids[0].tolist()

            # Compare generated tokens with expected tokens
            generation_matches = generated_tokens == expected_tokens

            # TODO(3outeille): handle audio and image generation
            if generation_matches:
                logger.info(f"Expected:  {Colors.GREEN}{self._decode_text_tokens(expected_tokens)}{Colors.RESET}")
                logger.info(f"Generated: {Colors.GREEN}{self._decode_text_tokens(generated_tokens)}{Colors.RESET}")
                logger.info(f"{Colors.GREEN}✓ Generation matches training sequence!{Colors.RESET}")
            else:
                logger.info(f"Expected:  {Colors.GREEN}{self._decode_text_tokens(expected_tokens)}{Colors.RESET}")
                logger.info(f"Generated: {Colors.RED}{self._decode_text_tokens(generated_tokens)}{Colors.RESET}")
                # Count matching tokens
                matches = sum(1 for g, e in zip(generated_tokens, expected_tokens) if g == e)
                logger.info(
                    f"{Colors.YELLOW}✗ Generation mismatch: {matches}/{len(expected_tokens)} tokens match{Colors.RESET}"
                )

        # Assertions
        logger.info("-" * 70)
        logger.info(f"{Colors.BOLD}Running assertions{Colors.RESET}")

        # Assert loss decreased significantly
        loss_reduction_ratio = (initial_loss - final_loss) / initial_loss
        self.assertGreater(
            loss_reduction_ratio,
            self.training_loss_reduction_threshold,
            f"Expected loss to decrease by at least {self.training_loss_reduction_threshold * 100:.0f}%, "
            f"got {loss_reduction:.1f}%",
        )
        logger.info(
            f"{Colors.GREEN}✓ Loss decreased by more than {self.training_loss_reduction_threshold * 100:.0f}%{Colors.RESET}"
        )

        # Assert grad_norm decreased significantly
        grad_norm_reduction_ratio = (initial_grad_norm - final_grad_norm) / initial_grad_norm
        self.assertGreater(
            grad_norm_reduction_ratio,
            self.training_grad_norm_reduction_threshold,
            f"Expected grad_norm to decrease by at least {self.training_grad_norm_reduction_threshold * 100:.0f}%, "
            f"got {grad_norm_reduction:.1f}%",
        )
        logger.info(
            f"{Colors.GREEN}✓ Grad norm decreased by more than {self.training_grad_norm_reduction_threshold * 100:.0f}%{Colors.RESET}"
        )

        # Assert generation matches (if applicable)
        if generation_matches is not None:
            self.assertTrue(generation_matches, "Expected model to generate the training sequence after overfitting")
            logger.info(f"{Colors.GREEN}✓ Generated sequence matches training sequence{Colors.RESET}")

        logger.info("=" * 70)
        logger.info(f"Finished test: {self._testMethodName}")
        logger.info("=" * 70)
