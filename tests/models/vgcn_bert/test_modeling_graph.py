import os
import tempfile
import unittest

from transformers.models.vgcn_bert.modeling_graph import WordGraph


class WordGraphTest(unittest.TestCase):
    def test_word_graph(self):
        # Create a WordGraph
        word_graph = WordGraph()
        self.assertTrue(isinstance(word_graph, WordGraph))

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Save it to disk
            word_graph.save_pretrained(tmpdirname)
            # Load it from disk
            word_graph = WordGraph.from_pretrained(tmpdirname)
            self.assertTrue(isinstance(word_graph, WordGraph))

    def test_word_graph_from_pretrained(self):
        # Download model and configuration from S3 and cache.
        model = WordGraph.from_pretrained("bert-base-uncased")
        self.assertIsNotNone(model)

    # os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # model_path = "/tmp/local-huggingface-models/hf-maintainers_distilbert-base-uncased"

    # # DistilBertTokenizerFast
    # tokenizer = tfr.AutoTokenizer.from_pretrained(model_path)

    # words_relations = [
    #     ("I", "you", 0.3),
    #     ("here", "there", 0.7),
    #     ("city", "montreal", 0.8),
    #     ("comeabc", "gobefbef", 0.2),
    # ]
    # wgraph = WordGraph(words_relations, tokenizer)
    # # vocab_adj, vocab, vocab_indices = wgraph.adjacency_matrix, wgraph.vocab, wgraph.vocab_indices
    # print(len(wgraph.vocab))
    # # print(wgraph.tokenizer_id_to_wgraph_id_array)
    # print_matrix(wgraph.adjacency_matrix.todense())

    # # texts = [" I am here", "He is here", "here i am, gobefbef"]
    # texts = [" I am here!", "He is here", "You are also here, gobefbef!", "What is interpribility"]
    # wgraph = WordGraph(texts, tokenizer, window_size=4)
    # # vocab_adj, vocab, vocab_indices = wgraph.adjacency_matrix, wgraph.vocab, wgraph.vocab_indices

    # print(len(wgraph.vocab))
    # # print(wgraph.tokenizer_id_to_wgraph_id_array)
    # print_matrix(wgraph.adjacency_matrix.todense())
    # print()
    # norm_adj = _normalize_adj(wgraph.adjacency_matrix)
    # print_matrix(norm_adj.todense())

    # # print(vocab_indices[vocab[3]])
    # print("---end---")
