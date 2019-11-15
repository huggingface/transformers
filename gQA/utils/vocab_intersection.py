import os
import json
import argparse

from tqdm import tqdm


def merge_vocabs(args):

	combined_words = dict()

	with open(args.hotpot_vocab_file) as open_file:
		hotpot_vocab = json.loads(open_file.read())

	with open(args.sum_vocab_file) as open_file:
		sum_vocab = json.loads(open_file.read())

	for hotpot_word, hotpot_freq in tqdm(hotpot_vocab):
		combined_words[hotpot_word] = hotpot_freq

	for sum_word, sum_freq in tqdm(sum_vocab):
		if sum_word in combined_words:
			combined_words[sum_word] = sum_freq + combined_words[sum_word]
		else:
			combined_words[sum_word] = sum_freq

	combined_words = sorted(combined_words.items(), key=lambda x: x[1], reverse=True)
	
	output_vocabs = list()
	prev_freq = 1e15
	for word, freq in combined_words:
		if freq > prev_freq:
			print("word")
		prev_freq = freq
		output_vocabs.append([word, freq])
	print(output_vocabs[:10])
	print(len(output_vocabs))
	output_file = os.path.join(args.output_dir, "gsumgqa_vocab.json")
	# print(output_vocabs[:100])
	with open(output_file, 'w', encoding='utf-8') as open_file:
		open_file.write(json.dumps(output_vocabs))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hotpot_vocab_file", default=None, type=str, help="data file")
    parser.add_argument("--sum_vocab_file", default=None, type=str, help="date file")
    parser.add_argument("--output_dir", default=None, type=str, help="directory to output")
    args = parser.parse_args()

    merge_vocabs(args)

if __name__ == "__main__":
    main()
