import functools
import os
import copy
import json
import base64
from datetime import datetime
from typing import Union, Dict, Iterable, List, Optional, Tuple
import requests
from io import BytesIO
import numpy as np
from jinja2 import Template, Environment
from PIL import Image
from typing import List, Union
import torch
from ...feature_extraction_utils import BatchFeature
from ...image_processing_utils import select_best_resolution
from ...image_utils import ImageInput, get_image_size, to_numpy_array
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack, _validate_images_text_input_order
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import logging
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union


CHAT_FORMATS = {
    "chat-command-turn_tokens-v3": '{% if preamble != "" %}<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{{ preamble }}<|END_OF_TURN_TOKEN|>{% endif %}{% for turn in turns %}{% if turn.message and turn.message != "" %}<|START_OF_TURN_TOKEN|>{{ turn.role | replace("User", "<|USER_TOKEN|>") | replace("Chatbot", "<|CHATBOT_TOKEN|><|START_RESPONSE|>") | replace("System", "<|SYSTEM_TOKEN|>") }}{{ turn.message }}{% if turn.role == "Chatbot" %}<|END_RESPONSE|>{% endif %}<|END_OF_TURN_TOKEN|>{% endif %}{% endfor %}<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>',
}

STD_PREAMBLE = """
# System Preamble
You are in contextual safety mode. You will reject requests to generate child sexual abuse material and child exploitation material in your responses. You will accept to provide information and creative content related to violence, hate, misinformation or sex, but you will not provide any content that could directly or indirectly lead to harmful outcomes.

Your information cutoff date is June 2024.

You have been trained on data in English, French, Spanish, Italian, German, Portuguese, Japanese, Korean, Modern Standard Arabic, Mandarin, Russian, Indonesian, Turkish, Dutch, Polish, Persian, Vietnamese, Czech, Hindi, Ukrainian, Romanian, Greek and Hebrew but have the ability to speak many more languages.

# Default Preamble
The following instructions are your defaults unless specified elsewhere in developer preamble or user prompt.
- Your name is Aya.
- You are a large language model built by Cohere.
- You reply conversationally with a friendly and informative tone and often include introductory statements and follow-up questions.
- If the input is ambiguous, ask clarifying follow-up questions.
- Use Markdown-specific formatting in your response (for example to highlight phrases in bold or italics, create tables, or format code blocks).
- Use LaTeX to generate mathematical notation for complex equations.
- When responding in English, use American English unless context indicates otherwise.
- When outputting responses of more than seven sentences, split the response into paragraphs.
- Prefer the active voice.
- Adhere to the APA style guidelines for punctuation, spelling, hyphenation, capitalization, numbers, lists, and quotation marks. Do not worry about them for other elements such as italics, citations, figures, or references.
- Use gender-neutral pronouns for unspecified persons.
- Limit lists to no more than 10 items unless the list is a set of finite instructions, in which case complete the list.
- Use the third person when asked to write a summary.
- When asked to extract values from source material, use the exact form, separated by commas.
- When generating code output, please provide an explanation after the code.
- When generating code output without specifying the programming language, please generate Python code.
- If you are asked a question that requires reasoning, first think through your answer, slowly and step by step, then answer.
"""

DEFAULT_ROLE_MAP = {
    "system": "System",
    "user": "User",
    "chatbot": "Chatbot",
}

START_OF_IMG = "<|START_OF_IMG|>"
END_OF_IMG = "<|END_OF_IMG|>"
IMG_PATCH = "<|IMG_PATCH|>"
IMG_LINE_BREAK = "<|IMG_LINE_BREAK|>"

TILE = "TILE"
TILE_GLOBAL = "TILE_GLOBAL"

class AyaVisionProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": "longest",
            "padding_side": "left",
            "truncation": True,
        },
        "images_kwargs": {
            "do_pad": True,
        },
    }


class AyaVisionProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = [
        "chat_template",
        "patch_size",
        "vision_feature_select_strategy",
        "image_token",
        "chat_preamble", # saurabhdash: should this be moved to tokenizer config?
        "max_splits_per_img",
        "size",
        "downsample_factor",
    ]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        patch_size=None,
        downsample_factor=1,
        vision_feature_select_strategy=None,
        chat_template=None,
        **kwargs: Unpack[AyaVisionProcessorKwargs],
    ):
        self.patch_size = patch_size * downsample_factor
        self.img_size = kwargs.get("size", 364)
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.chat_preamble = kwargs.get("chat_preamble", STD_PREAMBLE)
        self.max_splits_per_img = kwargs.get("max_splits_per_img", 1)
        self.role_map = DEFAULT_ROLE_MAP
        self.image_mean = kwargs.get("image_mean", [0.5, 0.5, 0.5])
        self.image_std = kwargs.get("image_std", [0.5, 0.5, 0.5])
        self.image_token = kwargs.get("image_token", "<|IMG_PATCH|>")
        
        # Initialize the parent class
        super().__init__(
            image_processor,
            tokenizer,
            chat_template=CHAT_FORMATS["chat-command-turn_tokens-v3"]
            if chat_template is None
            else chat_template,
        )
        
        # Configure the image processor with our parameters
        if self.image_processor is not None:
            self.image_processor.img_size = self.img_size
            self.image_processor.max_splits_per_image = self.max_splits_per_img
            self.image_processor.image_mean = self.image_mean
            self.image_processor.image_std = self.image_std

    def __call__(self, batch_of_conversations, **kwargs) -> BatchFeature:
        """
        Process a *batch* of conversations, each conversation can be multimodal or text-only.
        
        batch_of_conversations: List[List[Dict]]
            A list of conversations, where each conversation is itself a list of turns (dicts).
        """

        if isinstance(self.chat_template, str):
            # Create a Jinja2 environment with proper configuration
            env = Environment(autoescape=False)
            self.chat_template = env.from_string(self.chat_template)
        
        # Merge user-supplied kwargs with default kwargs
        output_kwargs = self._merge_kwargs(
            AyaVisionProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        
        text_kwargs = output_kwargs["text_kwargs"]
        # Get return_tensors from kwargs or default to None
        return_tensors = kwargs.get("return_tensors", None)

        if not isinstance(batch_of_conversations, list):
            raise ValueError("batch_of_conversations must be a list of conversations.")
        if len(batch_of_conversations) == 0:
            raise ValueError("No conversations provided (empty list).")

        if isinstance(batch_of_conversations[0], dict):
            # Single conversation -> wrap in list
            batch_of_conversations = [batch_of_conversations]

        all_prompts = []
        all_image_patches = []
        image_num_patches = []
        for conversation in batch_of_conversations:
            prompt, image_patches, _ = self.encode_turns(conversation)
            all_prompts.append(prompt)
            all_image_patches.append(image_patches)
            image_num_patches.append(len(image_patches))

        text_inputs = self.tokenizer(
            all_prompts,
            **text_kwargs
        )

        # Zero-pad the image patches across the batch
        max_num_patches = max(image_num_patches) if image_num_patches else 0
        padded_image_patches = []
        padding_image = np.zeros((self.img_size, self.img_size, 3))
        for i, patch_seq in enumerate(all_image_patches):
            # Normalize the patch images
            patch_seq = self.image_processor.normalize_image_patches(patch_seq)
            padded = patch_seq + [
                padding_image for _ in range(max_num_patches - len(patch_seq))
            ]
            padded_image_patches.append(padded)

        # If there are *any* images in the batch, convert to np.array
        if any(len(p) > 0 for p in padded_image_patches):
            # shape = (batch_size, num_patches, H, W, C)
            padded_image_patches = np.array(padded_image_patches)
            # We want (batch_size, num_patches, channels, height, width)
            padded_image_patches = padded_image_patches.transpose(0, 1, 4, 2, 3)
            padded_image_patches = padded_image_patches.astype(np.float16)

            # Return the final BatchFeature with text + image data
            return BatchFeature(
                data={
                    **text_inputs,
                    "pixel_values": padded_image_patches,
                    "image_num_patches": np.array(image_num_patches),
                },
                tensor_type=return_tensors,
            )
        else:
            # No images in the entire batch (text-only). Return text
            return BatchFeature(
                data={**text_inputs},
                tensor_type=return_tensors,
            )

    def encode_turns(self, turns: List[dict]) -> Union[List[int], Tuple[str, List[np.ndarray], List]]:
        prompt = self.prompt_from_turns(turns)
        return prompt
    
    def prompt_from_turns(
        self, turns: List[dict], overwrite_date: Optional[datetime] = None
    ) -> Union[str, Tuple[str, List[np.ndarray], List]]:
        """
        Example code that returns the 3-tuple: (prompt_str, [img_patches], [img_sizes]).
        """
        turns, img_patches, img_sizes = self._turns_as_text_messages(turns)
        turns = [self.validate_turn(turn) for turn in turns]
        preamble = self.render_preamble(overwrite_date)
        if turns and turns[0]["role"] == "System" and isinstance(turns[0]["message"], str):
            # NOTE: system preamble override
            preamble = turns[0]["message"]
            turns = turns[1:]
        prompt = self.chat_template.render(preamble=preamble, turns=turns)
        return prompt, img_patches, img_sizes

    def _turns_as_text_messages(self, turns: List[List[dict]]) -> Tuple[List[dict], List[np.ndarray]]:
        message_turns = []
        img_patches = []
        img_size_array = []
        for turn in turns:
            is_message_content_array = isinstance(turn["message"], list) and \
                len(turn["message"]) > 0 and \
                isinstance(turn["message"][0], dict)
            if is_message_content_array:
                message, turn_img_patches, img_size = self.content_as_text(turn["message"])
            else:
                message, turn_img_patches, img_size = turn["message"], [], None
            message_turns.append({"role": turn["role"], "message": message})
            img_patches.extend(turn_img_patches)
            if img_size is not None:
                img_size_array.append(img_size)
        return message_turns, img_patches, img_size_array

    def content_as_text(self, content: List[dict]) -> Tuple[str, List[np.ndarray]]:
        message = ""
        img_patches = []
        img_size = None
        for content_row in content:
            text, img_patch_list, row_img_size = self.content_row_as_text(content_row)
            message += text
            if img_patch_list is not None:
                img_patches.extend(img_patch_list)
            if row_img_size is not None:
                img_size = row_img_size

        return message, img_patches, img_size

    def validate_turn(self, turn):
        """Validate if a turn has correct role and message.

        Args:
            turn (dict): A chat turn with role.

        Returns:
            dict: A chat turn with role capitalized and message.
        """
        turn = copy.copy(turn)
        original_role = None
        if "message" not in turn:
            raise ValueError(
                f"Chat turn {turn!r} is missing a 'message' field. Note that the convention for platform and datatools is different from command, which uses 'content'."
            )
        normalized_role = turn.get("role", "").capitalize()
        if normalized_role not in ["User", "Chatbot", "System"]:
            raise ValueError(
                f"Chat turn {turn!r} has invalid or missing role. Normalized role was {normalized_role!r}, but must be one of 'User', 'Chatbot', 'System'"
            )
        turn["role"] = normalized_role if original_role is None else original_role
        return turn

    def content_row_as_text(self, content: dict) -> Tuple[str, List[np.ndarray], Optional[Tuple[int, int]]]:
        img_pil = None
        if content.get("url"):
            img_pil = load_img_from_url(content["url"])
        elif content.get("img"):
            img_pil = content["img"]

        if img_pil:
            assert not content.get("text"), "Cannot have both text and url in the same content dict."
            img_pil = img_pil.convert("RGB")
            # Make sure image processor has the correct parameters
            self.image_processor.img_size = self.img_size
            self.image_processor.max_splits_per_image = self.max_splits_per_img
            
            img_pil = self.image_processor.scale_to_optimal_aspect_ratio(img_pil)
            text = self.img_tokens_from_size(*img_pil.size)
            img_splits = self.image_processor.make_img_splits(img_pil)
            # If there are more than 1 splits, also include a thumbnail image
            if len(img_splits) > 1: 
                img_splits += [img_pil.resize((self.img_size, self.img_size))]
            img_splits = [np.array(img) for img in img_splits]
            return text, img_splits, img_pil.size

        return content["text"], [], None

    def img_tokens_from_size(self, width: int, height: int) -> str:
        w_patch = width / self.patch_size
        h_patch = height / self.patch_size

        # Number of crops/ tiles after resizing to optimal aspect ratio
        w_tiles = width // self.img_size
        h_tiles = height // self.img_size

        assert w_patch % 1 == 0 and h_patch % 1 == 0, "height and width doesn't match the patch size"
        return self.create_image_str(w_tiles, h_tiles)

    def create_image_str(self, w_tiles, h_tiles):
        idx = 1
        img_patches_per_tile = (self.img_size // self.patch_size) ** 2

        img_string = f"{START_OF_IMG}"
        if h_tiles * w_tiles > 1:
            for h_tile in range(h_tiles):
                for w_tile in range(w_tiles):
                    img_string += f"{TILE}_{idx}" + f"{IMG_PATCH}" * img_patches_per_tile
                    idx += 1

        img_string += f"{TILE_GLOBAL}" + f"{IMG_PATCH}" * img_patches_per_tile
        img_string += f"{END_OF_IMG}"
        return img_string

    def render_preamble(self, overwrite_date: Optional[datetime] = None) -> str:
        if "{date}" in self.chat_preamble:
            date = overwrite_date if overwrite_date is not None else datetime.now()
            return self.chat_preamble.format(date=date.strftime("%A, %B %d, %Y"))
        return self.chat_preamble


def load_img_from_url(url: str) -> Image:
    if url.startswith("data:"):
        data_prefix = "data:image/jpeg;base64,"
        if not url.startswith(data_prefix):
            raise NotImplementedError(f"Currently only support format: {data_prefix}")
        data = url[len(data_prefix):]
        image_data = base64.b64decode(data)
    elif url.startswith("http"):
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'})
        image_data = response.content
    else:
        raise NotImplementedError(f"Unsupported url value {url=}")
    return Image.open(BytesIO(image_data))
