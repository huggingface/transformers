import re

from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding
from ..clip.image_processing_clip import CLIPImageProcessor


media_token = ({"image": ("<image>", 65)},)


class MplugOwlProcessor(ProcessorMixin):
    r"""
    Constructs a mPLUG-Owl processor which wraps a mPLUG-Owl image processor and an mPLUG-Owl tokenizer into a single
    processor.

    [`MplugOwlProcessor`] offers all the functionalities of [`MplugOwlImageProcessor`] and [`AutoTokenizer`]. See the
    docstring of [`~MplugOwlProcessor.__call__`] and [`~MplugOwlProcessor.decode`] for more information.

    Args:
        image_processor (`MplugOwlImageProcessor`):
            An instance of [`MplugOwlImageProcessor`]. The image processor is a required input.
        tokenizer (`AutoTokenizer`):
            An instance of ['MplugOwlTokenizer`]. The tokenizer is a required input.
    """
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "MplugOwlImageProcessor"
    tokenizer_class = "MplugOwlTokenizer"

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        super().__init__(image_processor, tokenizer)
        self.tokens_to_generate = 0
        self.add_BOS = True

    def __call__(self, text=None, images=None, return_tensors=None, **kwargs):
        if text is None and images is None:
            raise ValueError("You have to specify either text or images. Both cannot be none.")

        if text is not None:
            encoding = tokenize_prompts(
                prompts=text,
                tokens_to_generate=self.tokens_to_generate,
                add_BOS=self.add_BOS,
                tokenizer=self.tokenizer,
                ignore_dist=True,
                **kwargs,
            )
            # encoding = self.tokenizer(text, return_tensors=return_tensors, **kwargs)

        if images is not None:
            image_features = self.image_processor(images, return_tensors=return_tensors, **kwargs)

        if text is not None and images is not None:
            encoding["pixel_values"] = image_features.pixel_values
            return BatchEncoding(data=encoding)
        elif text is not None:
            return BatchEncoding(data=encoding)
        else:
            return BatchEncoding(data=dict(**image_features), tensor_type=return_tensors)

    def batch_decode(self, skip_special_tokens=True, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, skip_special_tokens=skip_special_tokens, **kwargs)

    def decode(self, skip_special_tokens=True, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, skip_special_tokens=skip_special_tokens, **kwargs)


class MplugOwlImageProcessor(CLIPImageProcessor):
    r"""
    A warp of CLIPImageProcessor
    """
    pass


def detokenize_generations(tokens_gpu_tensor, lengths_gpu_tensor, return_segments, tokenizer):
    """Detokenize the generated tokens."""

    prompts_plus_generations = []
    if return_segments:
        prompts_plus_generations_segments = []

    tokens = tokens_gpu_tensor.cpu().numpy().tolist()
    lengths = lengths_gpu_tensor.cpu().numpy().tolist()
    for sequence_tokens, length in zip(tokens, lengths):
        sequence_tokens = sequence_tokens[:length]
        prompts_plus_generations.append(tokenizer.detokenize(sequence_tokens))
        if return_segments:
            from tokenizers.decoders import Metaspace

            if hasattr(tokenizer, "tokenizer"):
                if isinstance(tokenizer.tokenizer.decoder, Metaspace):
                    words = tokenizer.tokenizer.decode(sequence_tokens)
                else:
                    words = []
                    for token in sequence_tokens:
                        word = tokenizer.tokenizer.decoder[token]
                        word = bytearray([tokenizer.tokenizer.byte_decoder[c] for c in word]).decode(
                            "utf-8", errors="replace"
                        )
                        words.append(word)
                prompts_plus_generations_segments.append(words)
            else:
                words = tokenizer.detokenize(sequence_tokens)
                # else:
                #     words = []
                #     for token in sequence_tokens:
                #         word = tokenizer.tokenizer.decoder[token]
                #         word = bytearray(
                #             [tokenizer.tokenizer.byte_decoder[c] for c in word]).decode(
                #                 'utf-8', errors='replace')
                #         words.append(word)
                prompts_plus_generations_segments.append(words)

    if return_segments:
        return tokens, prompts_plus_generations, prompts_plus_generations_segments

    return tokens, prompts_plus_generations


def tokenize_prompts(prompts=None, tokens_to_generate=None, add_BOS=None, rank=0, tokenizer=None, **kwargs):
    """Tokenize prompts and make them avaiable on all ranks."""

    # On all ranks set to None so we can pass them to functions
    prompts_tokens_cuda_long_tensor = None
    prompts_length_cuda_long_tensor = None

    # On the specified rank, build the above.
    attention_mask = None

    prompts_tokens_cuda_long_tensor, prompts_length_cuda_long_tensor, attention_mask = _tokenize_prompts_and_batch(
        prompts, tokens_to_generate, add_BOS, tokenizer, **kwargs
    )

    return {
        "input_ids": prompts_tokens_cuda_long_tensor,
        "attention_mask": attention_mask,
        # "prompt_length": prompts_length_cuda_long_tensor,
    }


def _tokenize_prompts_and_batch(prompts, tokens_to_generate, add_BOS, tokenizer, **kwargs):
    import torch

    """Given a set of prompts and number of tokens to generate:
    - tokenize prompts
    - set the sequence length to be the max of length of prompts plus the number of tokens we would like to generate
    - pad all the sequences to this length so we can convert them into a 2D tensor.
    """

    # Tokenize all the prompts.
    # if add_BOS:
    #     prompts_tokens = [[tokenizer.bos] + tokenizer.tokenize(prompt)
    #                       for prompt in prompts]
    # else:
    #     prompts_tokens = [tokenizer.tokenize(prompt) for prompt in prompts]

    prompts_tokens = [_tokenize_prompt(prompt, tokenizer, add_BOS, **kwargs) for prompt in prompts]

    # Now we have a list of list of tokens which each list has a different
    # size. We want to extend this list to:
    #   - incorporate the tokens that need to be generated
    #   - make all the sequences equal length.
    # Get the prompts length.
    prompts_length = [len(prompt_tokens) for prompt_tokens in prompts_tokens]
    # Get the max prompts length.
    max_prompt_len = max(prompts_length)
    # Number of tokens in the each sample of the batch.
    samples_length = max_prompt_len + tokens_to_generate
    # Now update the list of list to be of the same size: samples_length.
    for prompt_tokens, prompt_length in zip(prompts_tokens, prompts_length):
        padding_size = samples_length - prompt_length
        prompt_tokens.extend([tokenizer.eos_token_id] * padding_size)

    # Now we are in a structured format, we can convert to tensors.
    prompts_tokens_tensor = torch.LongTensor(prompts_tokens)
    prompts_length_tensor = torch.LongTensor(prompts_length)
    attention_mask = torch.zeros(prompts_tokens_tensor.shape[:2])
    for i, l in enumerate(prompts_length_tensor):
        attention_mask[i, :l] = 1
    return prompts_tokens_tensor, prompts_length_tensor, attention_mask


def _tokenize_prompt(prompt, tokenizer, add_BOS=False, media_info={"<image>": 65}, **kwargs):
    media_tokens = {k: -int(i + 1) for i, k in enumerate(media_info.keys())}
    media_lengths = media_info.copy()

    if add_BOS:
        prompt_chunk = [tokenizer.bos_token_id]
    else:
        prompt_chunk = []

    # Pure Text
    if all([media_token not in prompt for media_token in media_tokens.keys()]):
        enc_chunk = prompt_chunk + tokenizer(prompt, add_special_tokens=False, **kwargs)["input_ids"]

    # Multi-Modal Text
    else:
        enc_chunk = prompt_chunk
        pattern = "|".join(map(re.escape, list(media_tokens.keys())))
        chunk_strs = re.split(f"({pattern})", prompt)
        chunk_strs = [x for x in chunk_strs if len(x) > 0]
        for idx, chunk_str in enumerate(chunk_strs):
            if chunk_str in media_tokens:
                enc_chunk += [media_tokens[chunk_str]] * media_lengths[chunk_str]
            else:
                tmp_chunk = tokenizer(chunk_str, add_special_tokens=False)["input_ids"]
                enc_chunk += tmp_chunk
    return enc_chunk
