import json
import os
from typing import Dict, List, Optional, Sequence

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


class NanoGPTTokenizer(PreTrainedTokenizerFast):
    """Transformers-compatible tokenizer backed by a tokenizer.json produced from the tiktoken encoding."""

    vocab_files_names = {"tokenizer_file": "tokenizer.json"}
    model_input_names = ["input_ids", "attention_mask"]

    _SPECIAL_TOKENS = {
        "bos": "<|bos|>",
        "user_start": "<|user_start|>",
        "user_end": "<|user_end|>",
        "assistant_start": "<|assistant_start|>",
        "assistant_end": "<|assistant_end|>",
        "python_start": "<|python_start|>",
        "python_end": "<|python_end|>",
        "output_start": "<|output_start|>",
        "output_end": "<|output_end|>",
    }

    def __init__(
        self,
        tokenizer_file: str,
        bos_token: str = "<|bos|>",
        eos_token: str = "<|assistant_end|>",
        pad_token: Optional[str] = None,
        chat_template: Optional[str] = None,
        **kwargs,
    ) -> None:
        pad_token = pad_token or eos_token

        additional_special_tokens = [
            token
            for token in self._SPECIAL_TOKENS.values()
            if token not in {bos_token, eos_token, pad_token}
        ]

        super().__init__(
            tokenizer_file=tokenizer_file,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            additional_special_tokens=additional_special_tokens,
            chat_template=chat_template,
            **kwargs,
        )

        self.vocab_file = tokenizer_file
        self.special_token_ids: Dict[str, int] = {
            name: self.convert_tokens_to_ids(token)
            for name, token in self._SPECIAL_TOKENS.items()
        }

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoTokenizer"):
        pass

    @classmethod
    def _load_vocab_file(cls, pretrained_model_name_or_path, revision=None):
        local_tok_path = os.path.join(pretrained_model_name_or_path, "tokenizer.json")
        if os.path.isfile(local_tok_path):
            return local_tok_path
        try:
            tok_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename="tokenizer.json",
                revision=revision,
            )
            return tok_path
        except (HfHubHTTPError, OSError) as e:
            raise ValueError(
                f"Could not load tokenizer.json from {pretrained_model_name_or_path}. "
                f"Make sure the path exists or the repo is accessible on the Hub."
            ) from e

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *init_inputs, **kwargs):
        tokenizer_file = cls._load_vocab_file(pretrained_model_name_or_path, revision=kwargs.get("revision"))
        kwargs["tokenizer_file"] = tokenizer_file
        chat_template = kwargs.get("chat_template")
        if chat_template is None:
            config_path = os.path.join(pretrained_model_name_or_path, "tokenizer_config.json")
            if os.path.isfile(config_path):
                with open(config_path) as f:
                    data = json.load(f)
                chat_template = data.get("chat_template")
                if chat_template is not None:
                    kwargs["chat_template"] = chat_template
        return super().from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)

    def save_vocabulary(  # type: ignore[override]
        self,
        save_directory: str,
        filename_prefix: Optional[str] = None,
    ) -> tuple[str]:
        os.makedirs(save_directory, exist_ok=True)
        filename = "tokenizer.json"
        if filename_prefix is not None:
            filename = f"{filename_prefix}-{filename}"
        save_path = os.path.join(save_directory, filename)
        with open(self.vocab_file, "rb") as src, open(save_path, "wb") as dst:
            dst.write(src.read())
        return (save_path,)

    # ------------------------------------------------------------------
    # Chat helpers (unchanged logic)
    # ------------------------------------------------------------------
    def encode_special(self, token: str) -> int:
        if token in self.special_token_ids:
            return self.special_token_ids[token]
        return self.convert_tokens_to_ids(token)

    def apply_chat_template(  # type: ignore[override]
        self,
        conversation,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        return_tensors: Optional[str] = None,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        **kwargs,
    ):
        messages = conversation["messages"] if isinstance(conversation, dict) and "messages" in conversation else conversation
        token_ids = self._render_conversation_ids(messages)
        if add_generation_prompt:
            token_ids.append(self.special_token_ids["assistant_start"])
        if tokenize:
            # Build output dictionary with input_ids and attention_mask
            output = {
                "input_ids": [token_ids],
                "attention_mask": [[1] * len(token_ids)]
            }
            
            # Convert to the requested tensor format
            if return_tensors == "pt":
                import torch
                from transformers.tokenization_utils_base import BatchEncoding
                return BatchEncoding({
                    "input_ids": torch.tensor(output["input_ids"], dtype=torch.long),
                    "attention_mask": torch.tensor(output["attention_mask"], dtype=torch.long)
                })
            elif return_tensors == "np":
                import numpy as np
                return {
                    "input_ids": np.array(output["input_ids"], dtype=np.int64),
                    "attention_mask": np.array(output["attention_mask"], dtype=np.int64)
                }
            else:
                return output
        return self.decode(token_ids, skip_special_tokens=False)

    def _encode_text(self, text: str) -> List[int]:
        return self.encode(text, add_special_tokens=False)

    def _encode_assistant_content(self, content) -> List[int]:
        if isinstance(content, str):
            return self._encode_text(content)
        if isinstance(content, list):
            tokens: List[int] = []
            for part in content:
                part_type = part.get("type", "text")
                text = part.get("text", "")
                if part_type == "text":
                    tokens.extend(self._encode_text(text))
                elif part_type == "python":
                    tokens.extend(self._encode_block(self.special_token_ids["python_start"], text))
                elif part_type == "python_output":
                    tokens.extend(self._encode_block(self.special_token_ids["output_start"], text))
                else:
                    raise ValueError(f"Unknown assistant content part: {part_type}")
            return tokens
        raise ValueError(f"Unsupported assistant content type: {type(content)}")

    def _encode_block(self, start_token_id: int, content: str) -> List[int]:
        tokens = [start_token_id]
        tokens.extend(self._encode_text(content))
        closing = {
            self.special_token_ids["python_start"]: self.special_token_ids["python_end"],
            self.special_token_ids["output_start"]: self.special_token_ids["output_end"],
        }.get(start_token_id, None)
        if closing is None:
            raise ValueError("Unknown block start token id")
        tokens.append(closing)
        return tokens

    def _render_conversation_ids(self, conversation: Sequence[Dict[str, object]]) -> List[int]:
        if not conversation:
            raise ValueError("Conversation must contain at least one message")
        messages = list(conversation)
        if messages[0].get("role") == "system":
            if len(messages) < 2 or messages[1].get("role") != "user":
                raise ValueError("System message must be followed by a user message")
            merged = dict(messages[1])
            merged["content"] = f"{messages[0]['content']}\n\n{messages[1]['content']}"
            messages = [merged] + messages[2:]

        ids: List[int] = [self.special_token_ids["bos"]]
        for idx, message in enumerate(messages):
            expected_role = "user" if idx % 2 == 0 else "assistant"
            role = message.get("role")
            if role != expected_role:
                raise ValueError(f"Expected role {expected_role}, received {role} at index {idx}")
            content = message.get("content")
            if expected_role == "user":
                if not isinstance(content, str):
                    raise ValueError("User messages must contain string content")
                ids.append(self.special_token_ids["user_start"])
                ids.extend(self._encode_text(content))
                ids.append(self.special_token_ids["user_end"])
            else:
                ids.append(self.special_token_ids["assistant_start"])
                ids.extend(self._encode_assistant_content(content))
                ids.append(self.special_token_ids["assistant_end"])
        return ids



