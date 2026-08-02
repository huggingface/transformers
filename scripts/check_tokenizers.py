from collections import Counter

import datasets

import transformers
from transformers.convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import logging


logging.set_verbosity_info()


# Some tokenizers (e.g. RobertaTokenizerFast) have a broken tokenizer.json on the Hub
# that is missing the model type and pre_tokenizer configuration. We need to patch
# them after loading.
def patch_fast_tokenizer(fast, checkpoint):
    """Patch fast tokenizers that have broken tokenizer.json files on the Hub."""
    import json

    from huggingface_hub import hf_hub_download
    from tokenizers import decoders, pre_tokenizers
    from tokenizers.models import BPE

    # Check if the fast tokenizer is broken (missing pre_tokenizer or merges)
    needs_patching = False
    if fast.backend_tokenizer.pre_tokenizer is None:
        needs_patching = True
    elif hasattr(fast.backend_tokenizer.model, "merges") and len(fast.backend_tokenizer.model.merges) == 0:
        needs_patching = True

    if not needs_patching:
        return fast

    try:
        # Load the tokenizer.json to get the correct configuration
        tokenizer_json_path = hf_hub_download(checkpoint, "tokenizer.json")
        with open(tokenizer_json_path) as f:
            tokenizer_json = json.load(f)

        # Fix pre_tokenizer
        pre_tokenizer_config = tokenizer_json.get("pre_tokenizer")
        if pre_tokenizer_config and pre_tokenizer_config.get("type") == "ByteLevel":
            fast.backend_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
                add_prefix_space=pre_tokenizer_config.get("add_prefix_space", False),
                trim_offsets=pre_tokenizer_config.get("trim_offsets", True),
            )

        # Fix decoder
        decoder_config = tokenizer_json.get("decoder")
        if decoder_config and decoder_config.get("type") == "ByteLevel":
            fast.backend_tokenizer.decoder = decoders.ByteLevel(
                add_prefix_space=decoder_config.get("add_prefix_space", True),
                trim_offsets=decoder_config.get("trim_offsets", True),
            )

        # Fix BPE model (vocab and merges)
        model_config = tokenizer_json.get("model", {})
        vocab = model_config.get("vocab", {})
        merges = model_config.get("merges", [])
        if vocab and merges:
            merges = [tuple(m.split(" ")) if isinstance(m, str) else tuple(m) for m in merges]
            fast.backend_tokenizer.model = BPE(vocab=vocab, merges=merges, fuse_unk=True)

    except Exception as e:
        print(f"Warning: could not patch fast tokenizer for {checkpoint}: {e}")

    return fast


TOKENIZER_CLASSES = {
    name: (getattr(transformers, name), getattr(transformers, name + "Fast")) for name in SLOW_TO_FAST_CONVERTERS
}

dataset = datasets.load_dataset("facebook/xnli", "all_languages", split="test+validation")  # no-script

total = 0
perfect = 0
imperfect = 0
wrong = 0


def check_diff(
    spm_diff: list[int], tok_diff: list[int], slow: PreTrainedTokenizerBase, fast: PreTrainedTokenizerBase
) -> bool:
    if spm_diff == list(reversed(tok_diff)):
        # AAA -> AA+A vs A+AA case.
        return True
    elif len(spm_diff) == len(tok_diff) and fast.decode(spm_diff) == fast.decode(tok_diff):
        # Second order OK
        # Barrich -> Barr + ich vs Bar + rich
        return True
    spm_reencoded = slow.encode(slow.decode(spm_diff))
    tok_reencoded = fast.encode(fast.decode(spm_diff))
    if spm_reencoded != spm_diff and spm_reencoded == tok_reencoded:
        # Type 3 error.
        # Snehagatha ->
        #       Sne, h, aga, th, a
        #       Sne, ha, gat, ha
        # Encoding the wrong with sp does not even recover what spm gave us
        # It fits tokenizer however...
        return True
    return False


def check_LTR_mark(line: str, idx: int, fast: PreTrainedTokenizerBase) -> bool:
    # Use encode_plus if available, otherwise use encode with return_offsets_mapping
    if hasattr(fast, "encode_plus"):
        enc = fast.encode_plus(line)[0]
        offsets = enc.offsets
    else:
        # For newer tokenizers that don't have encode_plus
        enc = fast(line, return_offsets_mapping=True, return_tensors=None)
        offsets = enc["offset_mapping"]
    curr, prev = offsets[idx], offsets[idx - 1]
    if curr is not None and line[curr[0] : curr[1]] == "\u200f":
        return True
    if prev is not None and line[prev[0] : prev[1]] == "\u200f":
        return True
    return False


def check_details(
    line: str, spm_ids: list[int], tok_ids: list[int], slow: PreTrainedTokenizerBase, fast: PreTrainedTokenizerBase
) -> bool:
    # Encoding can be the same with same result AAA -> A + AA vs AA + A
    # We can check that we use at least exactly the same number of tokens.
    for i, (spm_id, tok_id) in enumerate(zip(spm_ids, tok_ids)):
        if spm_id != tok_id:
            break
    first = i
    for i, (spm_id, tok_id) in enumerate(zip(reversed(spm_ids), reversed(tok_ids))):
        if spm_id != tok_id:
            break
    last = len(spm_ids) - i

    spm_diff = spm_ids[first:last]
    tok_diff = tok_ids[first:last]

    if check_diff(spm_diff, tok_diff, slow, fast):
        return True

    if check_LTR_mark(line, first, fast):
        return True

    if last - first > 5:
        # We might have twice a single problem, attempt to subdivide the disjointed tokens into smaller problems
        spms = Counter(spm_ids[first:last])
        toks = Counter(tok_ids[first:last])

        removable_tokens = {spm_ for (spm_, si) in spms.items() if toks.get(spm_, 0) == si}
        min_width = 3
        for i in range(last - first - min_width):
            if all(spm_ids[first + i + j] in removable_tokens for j in range(min_width)):
                possible_matches = [
                    k
                    for k in range(last - first - min_width)
                    if tok_ids[first + k : first + k + min_width] == spm_ids[first + i : first + i + min_width]
                ]
                for j in possible_matches:
                    if check_diff(
                        spm_ids[first : first + i], tok_ids[first : first + j], slow, fast
                    ) and check_details(
                        line,
                        spm_ids[first + i : last],
                        tok_ids[first + j : last],
                        slow,
                        fast,
                    ):
                        return True

    print(f"Spm: {[fast.decode([spm_ids[i]]) for i in range(first, last)]}")
    try:
        print(f"Tok: {[fast.decode([tok_ids[i]]) for i in range(first, last)]}")
    except Exception as e:
        print(f"Could not decode tok_ids: {e}")

    fast.decode(spm_ids[:first])
    fast.decode(spm_ids[last:])
    wrong = fast.decode(spm_ids[first:last])
    print()
    print(wrong)
    return False


def test_string(slow: PreTrainedTokenizerBase, fast: PreTrainedTokenizerBase, text: str) -> None:
    global perfect
    global imperfect
    global wrong
    global total

    slow_ids = slow.encode(text)
    fast_ids = fast.encode(text)

    skip_assert = False
    total += 1

    if slow_ids != fast_ids:
        if check_details(text, slow_ids, fast_ids, slow, fast):
            skip_assert = True
            imperfect += 1
        else:
            wrong += 1
    else:
        perfect += 1

    if total % 10000 == 0:
        print(f"({perfect} / {imperfect} / {wrong} ----- {perfect + imperfect + wrong})")

    if skip_assert:
        return

    assert slow_ids == fast_ids, (
        f"line {text} : \n\n{slow_ids}\n{fast_ids}\n\n{slow.tokenize(text)}\n{fast.tokenize(text)}"
    )


def test_tokenizer(slow: PreTrainedTokenizerBase, fast: PreTrainedTokenizerBase) -> None:
    global batch_total
    for i in range(len(dataset)):
        # premise, all languages
        for text in dataset[i]["premise"].values():
            test_string(slow, fast, text)

        # hypothesis, all languages
        for text in dataset[i]["hypothesis"]["translation"]:
            test_string(slow, fast, text)


if __name__ == "__main__":
    # Hardcoded checkpoints for common tokenizers (since max_model_input_sizes was removed)
    CHECKPOINTS = {
        "AlbertTokenizer": ["albert-base-v2", "albert-large-v2"],
        "BartTokenizer": ["facebook/bart-base", "facebook/bart-large"],
        "BertTokenizer": ["bert-base-uncased", "bert-base-cased", "bert-large-uncased"],
        "BigBirdTokenizer": ["google/bigbird-roberta-base"],
        "BlenderbotTokenizer": ["facebook/blenderbot-400M-distill"],
        "CamembertTokenizer": ["camembert-base"],
        "CLIPTokenizer": ["openai/clip-vit-base-patch32"],
        "CodeGenTokenizer": ["Salesforce/codegen-350M-mono"],
        "ConvBertTokenizer": ["YituTech/conv-bert-base"],
        "DebertaTokenizer": ["microsoft/deberta-base"],
        "DebertaV2Tokenizer": ["microsoft/deberta-v2-xlarge"],
        "DistilBertTokenizer": ["distilbert-base-uncased", "distilbert-base-cased"],
        "ElectraTokenizer": ["google/electra-small-discriminator", "google/electra-base-discriminator"],
        "FNetTokenizer": ["google/fnet-base"],
        "FunnelTokenizer": ["funnel-transformer/small"],
        "GPT2Tokenizer": ["gpt2", "gpt2-medium"],
        "HerbertTokenizer": ["allegro/herbert-base-cased"],
        "LayoutLMTokenizer": ["microsoft/layoutlm-base-uncased"],
        "LayoutLMv2Tokenizer": ["microsoft/layoutlmv2-base-uncased"],
        "LayoutLMv3Tokenizer": ["microsoft/layoutlmv3-base"],
        "LEDTokenizer": ["allenai/led-base-16384"],
        "LongformerTokenizer": ["allenai/longformer-base-4096"],
        "LxmertTokenizer": ["unc-nlp/lxmert-base-uncased"],
        "MarkupLMTokenizer": ["microsoft/markuplm-base"],
        "MBartTokenizer": ["facebook/mbart-large-cc25"],
        "MBart50Tokenizer": ["facebook/mbart-large-50"],
        "MobileBertTokenizer": ["google/mobilebert-uncased"],
        "MPNetTokenizer": ["microsoft/mpnet-base"],
        "MvpTokenizer": ["RUCAIBox/mvp"],
        "NllbTokenizer": ["facebook/nllb-200-distilled-600M"],
        "OpenAIGPTTokenizer": ["openai-gpt"],
        "PegasusTokenizer": ["google/pegasus-xsum"],
        "Qwen2Tokenizer": ["Qwen/Qwen2-0.5B"],
        "ReformerTokenizer": ["google/reformer-crime-and-punishment"],
        "RemBertTokenizer": ["google/rembert"],
        "RobertaTokenizer": ["roberta-base", "roberta-large"],
        "RoFormerTokenizer": ["junnyu/roformer_chinese_base"],
        "SeamlessM4TTokenizer": ["facebook/hf-seamless-m4t-medium"],
        "SqueezeBertTokenizer": ["squeezebert/squeezebert-uncased"],
        "T5Tokenizer": ["t5-small", "t5-base"],
        "UdopTokenizer": ["microsoft/udop-large"],
        "WhisperTokenizer": ["openai/whisper-tiny"],
        "XGLMTokenizer": ["facebook/xglm-564M"],
        "XLMRobertaTokenizer": ["xlm-roberta-base", "xlm-roberta-large"],
        "XLNetTokenizer": ["xlnet-base-cased", "xlnet-large-cased"],
    }

    for name, (slow_class, fast_class) in TOKENIZER_CLASSES.items():
        checkpoint_names = CHECKPOINTS.get(name, [])

        if not checkpoint_names:
            print(f"Skipping {name}: no hardcoded checkpoints (add to CHECKPOINTS dict to test)")
            continue

        for checkpoint in checkpoint_names:
            imperfect = 0
            perfect = 0
            wrong = 0
            total = 0

            print(f"========================== Checking {name}: {checkpoint} ==========================")
            try:
                slow = slow_class.from_pretrained(checkpoint, force_download=True)
                fast = fast_class.from_pretrained(checkpoint, force_download=True)
                # Patch broken fast tokenizers (e.g. RobertaTokenizerFast)
                fast = patch_fast_tokenizer(fast, checkpoint)
                test_tokenizer(slow, fast)
                print(f"Accuracy {perfect * 100 / total:.2f}")
            except Exception as e:
                print(f"Error testing {name} with {checkpoint}: {e}")
