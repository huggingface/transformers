"""
Helper to load vocab from SentencePiece model for RemBert.
This avoids the sentencepiece dependency.
"""

import json
import os


def load_vocab_from_file(vocab_file):
    """
    Load vocabulary from a SentencePiece model file or JSON file.
    Returns a dict mapping tokens to their IDs.
    """
    if vocab_file.endswith(".json"):
        with open(vocab_file, "r", encoding="utf-8") as f:
            return json.load(f)

    if vocab_file.endswith(".model"):
        # Try to load from cached JSON
        json_vocab_file = vocab_file.replace(".model", "_vocab.json")
        if os.path.exists(json_vocab_file):
            with open(json_vocab_file, "r", encoding="utf-8") as f:
                return json.load(f)

        # Extract from protobuf
        try:
            from ...convert_slow_tokenizer import import_protobuf

            model_pb2 = import_protobuf()

            m = model_pb2.ModelProto()
            with open(vocab_file, "rb") as f:
                m.ParseFromString(f.read())

            vocab = {piece.piece: i for i, piece in enumerate(m.pieces)}

            # Cache it for next time
            try:
                with open(json_vocab_file, "w", encoding="utf-8") as f:
                    json.dump(vocab, f, ensure_ascii=False, indent=2)
            except Exception:
                pass  # If we can't cache, that's ok

            return vocab
        except Exception as e:
            raise ValueError(f"Could not load vocab from {vocab_file}. Error: {e}")

    raise ValueError(f"Unsupported vocab file format: {vocab_file}")
