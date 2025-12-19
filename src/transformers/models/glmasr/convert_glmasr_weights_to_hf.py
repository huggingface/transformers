# python test_file/convert_asr.py \
#   --src zai-org/GLM-ASR-Nano-2512/model.safetensors \
#   --dst GLM-ASR-Nano-2512-HF/model.safetensors

import argparse
import os

from safetensors.torch import load_file, save_file


AUDIO_BOS_EOS_TOKEN_KEY = "audio_encoder.audio_bos_eos_token.weight"
EMBED_TOKENS_KEY = "model.embed_tokens.weight"

AUDIO_BOS_IDX = 59261
AUDIO_EOS_IDX = 59262


def rename_key(k: str):
    if k == AUDIO_BOS_EOS_TOKEN_KEY:
        return None

    if k.startswith("audio_encoder.proj."):
        return None

    if k == "model.norm.weighth":
        return "language_model.model.norm.weight"

    if k.startswith("model."):
        return "language_model.model." + k[len("model.") :]

    if k.startswith("lm_head."):
        return "language_model.lm_head." + k[len("lm_head.") :]

    if k.startswith("audio_encoder.adapting.0."):
        return "multi_modal_projector.linear_1." + k.split(".")[-1]

    if k.startswith("audio_encoder.adapting.2."):
        return "multi_modal_projector.linear_2." + k.split(".")[-1]

    if k.startswith("audio_encoder.whisper."):
        return "audio_tower." + k[len("audio_encoder.whisper.") :]

    if k.startswith("audio_encoder.layer_norm."):
        return "audio_tower.layer_norm." + k.split(".")[-1]

    return k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.dst), exist_ok=True)

    state = load_file(args.src, device="cpu")
    new_state = {}
    collisions = []

    audio_bos_eos_weight = None
    if AUDIO_BOS_EOS_TOKEN_KEY in state:
        audio_bos_eos_weight = state[AUDIO_BOS_EOS_TOKEN_KEY]
        print(f"[INFO] Found {AUDIO_BOS_EOS_TOKEN_KEY}, shape: {audio_bos_eos_weight.shape}")

    for old_k, v in state.items():
        new_k = rename_key(old_k)
        if new_k is None:
            if old_k == AUDIO_BOS_EOS_TOKEN_KEY:
                print(f"[MERGE] {old_k} -> will merge to embed_tokens")
            else:
                print(f"[DROP]  {old_k}")
            continue

        tag = "KEEP" if old_k == new_k else "RENAME"
        print(f"[{tag}]  {old_k} -> {new_k}")

        if new_k in new_state:
            collisions.append((old_k, new_k))
        new_state[new_k] = v

    if collisions:
        for o, n in collisions[:20]:
            print(f"COLLISION {o} -> {n}")
        raise RuntimeError(f"key collision: {len(collisions)}")

    embed_tokens_new_key = "language_model.model.embed_tokens.weight"
    if audio_bos_eos_weight is not None and embed_tokens_new_key in new_state:
        embed_weight = new_state[embed_tokens_new_key]
        print(f"[INFO] embed_tokens shape: {embed_weight.shape}")
        print(f"[INFO] audio_bos_eos shape: {audio_bos_eos_weight.shape}")

        embed_weight = embed_weight.clone()
        embed_weight[AUDIO_BOS_IDX] = audio_bos_eos_weight[0]
        embed_weight[AUDIO_EOS_IDX] = audio_bos_eos_weight[1]

        new_state[embed_tokens_new_key] = embed_weight
        print(f"[MERGE] audio_bos_eos_token[0] -> embed_tokens[{AUDIO_BOS_IDX}]")
        print(f"[MERGE] audio_bos_eos_token[1] -> embed_tokens[{AUDIO_EOS_IDX}]")

    save_file(new_state, args.dst)

    print(f"\n Save: {args.dst}")
    print(f"Weight Count: {len(state)} -> {len(new_state)}")


if __name__ == "__main__":
    main()
