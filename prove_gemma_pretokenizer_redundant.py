from transformers import AutoTokenizer

def get_backend(tok):
    # TokenizersBackend wrappers sometimes expose either backend_tokenizer or _tokenizer
    if hasattr(tok, "backend_tokenizer"):
        return tok.backend_tokenizer
    if hasattr(tok, "_tokenizer"):
        return tok._tokenizer
    raise RuntimeError(f"Cannot find backend tokenizer on {type(tok)}")

def main():
    tok = AutoTokenizer.from_pretrained("hf-internal-testing/dummy-gemma", use_fast=True)
    bt = get_backend(tok)

    print("Tokenizer class:", type(tok))
    print("Normalizer:", bt.normalizer)
    print("PreTokenizer:", bt.pre_tokenizer)

    texts = [
        "Hello   this\tis  a test",
        "Hello this is a test",
        " Hello this is a test",
        "Hello\tthis is a test",
        "Hello\nthis is a test",
        "Hello\u00A0this is a test",   # non-breaking space
        "a  b   c",
        "a\tb c",
        "a b\tc",
    ]

    # baseline
    baseline_tok = [tok.tokenize(t) for t in texts]
    baseline_ids = [tok.encode(t) for t in texts]

    # disable pretokenizer
    orig = bt.pre_tokenizer
    bt.pre_tokenizer = None

    no_pre_tok = [tok.tokenize(t) for t in texts]
    no_pre_ids = [tok.encode(t) for t in texts]

    # restore
    bt.pre_tokenizer = orig

    print("\n== Results ==")
    for i, t in enumerate(texts):
        same_ids = baseline_ids[i] == no_pre_ids[i]
        same_tok = baseline_tok[i] == no_pre_tok[i]
        if not (same_ids and same_tok):
            print("\nDIFF for:", repr(t))
            print("tokenize baseline:", baseline_tok[i])
            print("tokenize no_pre :", no_pre_tok[i])
            print("ids baseline:", baseline_ids[i])
            print("ids no_pre :", no_pre_ids[i])
            break
    else:
        print("✅ No differences found: disabling pre_tokenizer does not change tokenize() or encode() for all test strings.")

if __name__ == "__main__":
    main()