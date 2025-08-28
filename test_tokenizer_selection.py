from transformers.utils.tokenizer_selection import TokenizerSelector

# Simple test data
test_texts = [
    ["Hello world, this is a test.", "Machine learning is fascinating."],
    ["Natural language processing helps computers understand text.", "Tokenization is important."],
    ["BPE and WordPiece are popular algorithms.", "SentencePiece works well too."]
]

print("Testing corpus analysis...")
stats = TokenizerSelector.analyze_corpus(iter(test_texts))
print(f"Vocab size: {stats.vocab_size}")
print(f"Avg word length: {stats.avg_word_length:.2f}")
print(f"Character diversity: {stats.char_diversity}")

print("\nTesting tokenizer recommendation...")
recommendation = TokenizerSelector.recommend_tokenizer(stats)
print(f"Recommended: {recommendation['type']}")
print(f"Rationale: {recommendation['rationale']}")
print(f"Vocab size suggestion: {recommendation['config']['vocab_size']}")

print("\nAll tests passed!")
