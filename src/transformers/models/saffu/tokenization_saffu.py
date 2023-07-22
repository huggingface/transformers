from abc import ABC
import json, os, re, torch
from abc import abstractmethod
from transformers.utils import logging
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, List, Optional, Tuple
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "subtoken_reference_file": "subtoken_reference.json",
    "vocab_file": "vocab.json",
    "merges_file": "merges.json",
    "chars_file": "chars.txt"
}

PRETRAINED_VOCAB_FILES_MAP = {
    "subtoken_reference_file": {
        "https://huggingface.co/saffu-BBLM10M/resolve/main/subtoken_reference.json",
        "https://huggingface.co/saffu-BBLM100M/resolve/main/subtoken_reference.json",
    },
    "chars_file": {
        "https://huggingface.co/saffu-BBLM10M/resolve/main/chars.json",
        "https://huggingface.co/saffu-BBLM100M/resolve/main/chars.json",
    },
    "vocab_file": {
        "saffu-BBLM10M": "https://huggingface.co/saffu-BBLM10M/resolve/main/vocab.json",
        "saffu-BBLM100M": "https://huggingface.co/saffu-BBLM100M/resolve/main/vocab.json",
    },
    "merges_file": {
        "saffu-BBLM10M": "https://huggingface.co/saffu-BBLM10M/resolve/main/merges.json",
        "saffu-BBLM100M": "https://huggingface.co/saffu-BBLM100M/resolve/main/merges.json",
    },
}

class SAFFUTokenizer(PreTrainedTokenizer):
    """
    Construct a SAFFU tokenizer. Based on rule-based pre-tokenization followed by Byte-Pair sub-word chunking.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    model_input_names = ["input_ids"]

    def __init__(
        self,
        vocab_file,
        subtoken_reference_file,
        merges_file,
        chars_file,
        r = 2,
        block_size = 100,
        heads = 2,
        space = False,
        pad = "<pad>", 
        oov = "<oov>", 
        sod = "<sod>", 
        eod = "<eod>", 
        frg = "<frg>",
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
    
        self._r = r
        self._space = space
        self._heads = heads
        self._block_size = block_size
        self._raw_td = json.load(open(merges_file))
        self._td = load_td(path = merges_file)
        self._wordchars = re.sub(" ", "", open(chars_file).read().strip())
        self._vocabulary = json.loads(open(vocab_file).read())
        self._index = {self._vocabulary[t]: t for t in self._vocabulary}
        self._subtoken_reference = json.loads(open(subtoken_reference_file).read())
        self._pad = pad
        self._oov = oov
        self._sod = sod
        self._eod = eod
        self._frg = frg
        self._padding = [self._vocabulary[self._pad]]*self._r
        self._masking = [self._vocabulary[self._pad]]*self._block_size
        self._heads_padding = [self._vocabulary[self._pad]]*self._heads

    def save_vocabulary(self, save_directory: str, 
                        filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + 
            VOCAB_FILES_NAMES["vocab_file"]
        )
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") +
            VOCAB_FILES_NAMES["merges_file"]
        )
        subtoken_reference_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + 
            VOCAB_FILES_NAMES["subtoken_reference_file"]
        )
        chars_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") +
            VOCAB_FILES_NAMES["chars_file"]
        )
        
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self._vocabulary, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        with open(merge_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self._raw_td, indent=2, sort_keys=True, ensure_ascii=False))
        with open(subtoken_reference_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self._subtoken_reference, indent=2, sort_keys=True, ensure_ascii=False))
        with open(chars_file, "w", encoding="utf-8") as f:
            f.write(self._wordchars)

        return vocab_file, merge_file, subtoken_reference_file, chars_file

    @property
    def vocab_size(self):
        return len(self._vocabulary)

    @staticmethod
    def word_tokenize(text, wordchars = "a-zA-Z0-9-'"): 
        return [token for token in re.split("(["+wordchars+"'-]+)", text) if token]

    @staticmethod
    def stick_spaces(stream):
        tokens = []
        for wi, w in enumerate(stream):
            if not tokens:
                tokens.append(w)
            elif w == ' ':
                if (tokens[-1][-1] != ' ') and (wi != len(stream)-1):
                    tokens.append(w)
                else:
                    tokens[-1] = tokens[-1] + w
            else:
                if tokens[-1][-1] == ' ':
                    if tokens[-1] == ' ':
                        tokens[-1] = tokens[-1] + w
                    else:
                        tokens[-1] = tokens[-1][:-1]#  + w
                        tokens.append(' ' + w)
                else:
                    tokens.append(w)
        return tokens

    @staticmethod
    def sentence_tokenize(text, wordchars = "a-zA-Z0-9-'", puncts = ".?!;:\n|"):
        sentences = []
        for sentence in re.split("(\s+(?<=["+puncts+"][^"+wordchars+"'-])\s*)", text):
            if not sentence: continue
            if not re.search("["+wordchars+"'-]", sentence):
                if len(sentences):
                    if sentence[-1] == " ":
                        if len(sentence) > 1:
                            sentences[-1] = sentences[-1] + sentence[:-1]
                            sentences.append(sentence[-1])
                        else:
                            sentences.append(sentence)
                    else:
                        sentences[-1] = sentences[-1] + sentence
                else:
                    sentences.append(sentence)
            else:
                if len(sentences):
                    if len(sentences[-1]) == 1 and sentences[-1] == " ": 
                        sentences[-1] = sentences[-1] + sentence
                    else:
                        sentences.append(sentence)
                else:
                    sentences.append(sentence)
        return sentences

    def bpe_tokenize(self, text): # , td = {}, reference = {}, space = False
        stream = self._subtoken_reference.get(text, list_tokenize(text, td = self._td))
        return (list(stream if self._space else self.stick_spaces(stream)))
    
    def _tokenize(self, text):
        """Tokenize a string."""
        return [sub for s in self.sentence_tokenize(text, wordchars = self._wordchars)
                for t in (self.word_tokenize(s, wordchars = self._wordchars) if self._space else 
                          self.stick_spaces(self.word_tokenize(s, wordchars = self._wordchars)))
                for sub in self.bpe_tokenize(t)]
        return document

    def preprocess(self, input_ids = []):
        document  = input_ids + [self._vocabulary[self._eod], self._vocabulary[self._pad]]
        blocks = []; docsize  = len(document)
        for bi in range(int(docsize/self._block_size) + 1):
            start = bi*self._block_size
            if start > docsize: continue
            end = min([(bi+1)*self._block_size, docsize])
            data = [self._vocabulary[self._frg if bi else  self._sod]] + document[start:end]
            block = (lambda x: x[:3] + [x[3:]])(
                list(map(list,
            zip(*[(t, self._padding[:self._r - m] + data[:m] if m - self._r < 0 else data[m - self._r:m],
                   data[:m] + self._masking[:self._block_size - m] if m < self._block_size else data[:self._block_size],
                   *(self._heads_padding[:self._heads - m] + data[:m] if m - self._heads < 0 else 
                     data[m - self._heads:m])[::-1]) # last heads tokens
                  for m, t in enumerate(data)])
            )) )
            blocks.append(block)
        return blocks

    def _convert_token_to_id(self, t):
        """Converts a token (str) in an id using the vocab."""
        return self._vocabulary.get(t, self._vocabulary.get(self._oov))

    def _convert_id_to_token(self, i):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self._index.get(i, self._oov)

def load_td(data = None, path = ''):
    if data is None and path:
        data = json.load(open(path))
    td = {}
    # from Tokenizer
    td['tok2ind'] = data['tok2ind']
    td['ind2tok'] = {v: k for k, v in td['tok2ind'].items()}
    td['action_trace'] = [{'pair': tuple(a[0]), 'type': 'merge' if a[1] else 'split', 
                           'count': a[2], 'score': a[3]} for a in data['action_trace']]
    td['tok2acts'] = defaultdict(list)
    td['pair2merge'] = dict()
    td['tok2splits'] = defaultdict(list)
    for aix, a in enumerate(td['action_trace']):
        if a['type'] =='split':
            td['tok2acts']["".join(a['pair'])].append(aix)
            td['tok2splits']["".join(a['pair'])].append(aix)
        else:
            td['pair2merge'][tuple(a['pair'])] = aix
            td['tok2acts'][a['pair'][0]].append(aix)
            td['tok2acts'][a['pair'][1]].append(aix)
    td['maxtoklen'] = max([len(t) for t in td['tok2ind']])
    # from BPE
    if 'unigraph' in data:
        td['unigraph'] = Counter(data['unigraph'])
        td['digraph'] = Counter({(l, r): v for l, r, v in data['digraph']})
        td['doc_unigraph'] = defaultdict(Counter)
        for k, v in data['doc_unigraph'].items():
            td['doc_unigraph'][k] = Counter(v)
    td['init_method'] = data['init_method']
    # from HRBPE
    if 'param_method' in data:
        td['param_method'] = data['param_method']
        td['reg_model'] = data['reg_model']
        td['early_stop'] = data['early_stop']
    return td

def list_tokenize(text, td = {}):
    assert td['action_trace'], "Can't tokenize, no trained model!"
    mock = BPE()
    mock.init([text], method=td['init_method'], apply=True) # method='char'
    prev_aix = -1; available_action_indices = []; observed = set(); tokenizing = True
    while tokenizing:
        available_action_indices = sorted(list(filter(lambda next_aix: next_aix > prev_aix, available_action_indices)) + 
                                          [next_aix for next_aix in [aix for tok in mock._unigraph 
                                                                     for aix in td['tok2splits'][tok] if tok not in observed] + 
                                           [td['pair2merge'][pair] for pair in mock._digraph
                                            if pair not in observed and pair in td['pair2merge']]
                                           if next_aix > prev_aix])
        observed = observed.union(set(mock._unigraph.keys()).union(set(mock._digraph.keys())))
        if available_action_indices:
            aix = available_action_indices[0]
        else:
            tokenizing = False
            break
        prev_aix = aix
        action = td['action_trace'][aix]
        if action['type'] == 'merge':
            mock.merge(action['pair'])
        else:
            mock.split(action['pair'])
    tks = []
    for t, idxs in mock._tok_idx.items():
        for ix in idxs:
            tks.append((t, ix))
    tks.sort(key=lambda ti: ti[1])
    tks, _ = zip(*tks)
    return tks

# purpose: cast a merge or split action object
# arguments: see __init__()
# prereqs: none
# use methods: none
# - __init__: 
# use attributes:
# - pair: tuple, form of the action
# - type: str, 'merge' or 'split' describing the type of action
# - count: int, indicating the number of times an action has been observed (for ranking)
class Action:
    # purpose: initialize a merge or split action object
    # arguments:
    # - pair: tuple, form of the action
    # - type: str, 'merge' or 'split' describing the type of action
    # - count: int, indicating the number of times an action has been observed (for ranking)
    # prereqs: NA
    # output: an initialized Action object for merge-split augmented BPE
    def __init__(self, pair, type='merge', count=-1):
        self.pair = pair
        self.type = type
        self.count = int(count)

# purpose: cast a tokenizer
# arguments: see __init__()
# prereqs:
# use methods: 
# - __init__: 
# - save: save a model for later use
# - load: load a saved model
# - init: initialize a model
# - fit: fit a model
# - tokenize: apply the model's trained tokenizer to a given text
# - display: plot the rank-frequency distribution of the current segmentation of the system's ingested data, as well as a given frequency model
# properties: the tokenizer's length is the size of its index
class Tokenizer(ABC):
    # purpose: initialize a tokenizer
    # arguments:
    # - tok2ind: (optional) dict, used by .load to set the index
    # prereqs: NA
    # output: a model shell ready for training or trained parameters
    def __init__(self, tok2ind=None):
        if tok2ind is None:
            self._tok2ind = {}
        else:
            self._tok2ind = tok2ind
        self._ind2tok = {v: k for k, v in self._tok2ind.items()}
        self._action_trace = []
        
    # the tokenizer's length is the size of its index
    def __len__(self):
        return len(self._tok2ind)
    
    # purpose: add a token to the index
    # arguments: tok: str, the token to add to the index
    def add_type(self, tok):
        if tok not in self._tok2ind:
            self._tok2ind[tok] = len(self._tok2ind)
            self._ind2tok[self._tok2ind[tok]] = tok
            
    # purpose: delete a type from the index
    # arguments: tok: str, the token to delete from the index
    def del_type(self, tok):
        if tok in self._tok2ind:
            idx = self._tok2ind[tok]
            # delete the token from the index and the lookup
            del self._ind2tok[idx]
            del self._tok2ind[tok]
            # shifting down each type that's a larger index
            # than the one just removed
            i = idx + 1
            while i in self._ind2tok:
                t = self._ind2tok[i]
                self._tok2ind[t] = i - 1
                self._ind2tok[i - 1] = t
                del self._ind2tok[i]
                
    # purpose: save a model for later use
    # arguments:
    # - path: str, directory location where models are to be saved
    # - data: dict, with fields 'tok2ind', and 'action_trace', which key a dictionary index mapping the vocabulary, and a list of ranked actions to apply as the tokenizers parameters.
    # output: saved model parameters in the location defined by path
    def save(self, path, data=None):
        if data is None:
            data = {}
        data['tok2ind'] = self._tok2ind
        data['action_trace'] = [[a.pair, 1 if a.type == 'merge' else 0, a.count, a.score if hasattr(a, "score") else None]
                                for a in self._action_trace]
        json.dump(data, open(path, 'w+'))
        
    # purpose: load a saved model
    # arguments: path: str, directory location from which the model will be loaded
    # prereqs: a saved model
    # output: loaded model parameters for operation of a trained tokenizer
    def load(self, path):
        data = json.load(open(path))
        self._tok2ind = data['tok2ind']
        self._ind2tok = {v: k for k, v in self._tok2ind.items()}
        self._action_trace = [ScoredAction(tuple(a[0]), count=a[2], score=a[3], 
                                           type='merge' if a[1] else 'split') 
                              for a in data['action_trace']]
        self._tok2acts = defaultdict(list)
        self._pair2merge = dict()
        self._tok2splits = defaultdict(list)
        for aix, a in enumerate(self._action_trace):
            if a.type =='split':
                self._tok2acts["".join(a.pair)].append(aix)
                self._tok2splits["".join(a.pair)].append(aix)
            else:
                self._pair2merge[tuple(a.pair)] = aix
                self._tok2acts[a.pair[0]].append(aix)
                self._tok2acts[a.pair[1]].append(aix)
        self._maxtoklen = max([len(t) for t in self._tok2ind])
        
        return data

    # purpose: initialize a model
    # arguments:
    # - docs: list (corpus) of strs (documents) to be used for training
    # prereqs: a selected BPE variant, which is what will inherit Tokenizer
    @abstractmethod
    def init(self, docs, seed=None):
        raise NotImplementedError

    # purpose: fit a model
    # arguments:
    # - num_batches: int, indicating the number of batches over which to rank and apply actions
    # - batch_size: int, indicating the number of actions to rank for each batch
    # - seed: int, indicating the seed of randomization
    # prereqs: a selected BPE variant, which is what will inherit Tokenizer
    # output: a trained model
    @abstractmethod
    def fit(self, num_batches, batch_size=1, seed=None):
        raise NotImplementedError
        
    # purpose: tokenize a text and map its tokens to vocabulary indices
    # arguments: text: str, to be tokenized
    # prereqs: a trained tokenizer
    # output: a list of token indices, mapped according to the model's vocabulary
    def encode(self, text):
        return self.tokens_to_indices(self.tokenize(text))

    # purpose: apply the model's trained tokenizer to a given text
    # arguments:
    # - text: str, target text to be tokenized
    # - start: 
    # prereqs: a trained tokenizer with an _action_trace, recording the ranked list of tokenization actions
    # output: a tuple of strs (tokens), which if joined together form the text
    def tokenize(self, text, start=-1):
        assert self._action_trace, "Can't tokenize, no trained model!"
        return self.apply_action_trace(text)
        
    # purpose: update the index of actions according to those which are applicable to the data
    # arguments:
    # - available_action_indices: list (actions list) of ints (indices) indicating those that are currently available for non-null operation
    # - model: a trained model object 
    # - prev_aix: int, indicating the previous action index that was applied
    # - observed: set, indicating the set of observed tokens within the current tokenization of the data
    def update_action_indices(self, available_action_indices, model, prev_aix = -1, observed = set()):
        available_action_indices = sorted(list(filter(lambda next_aix: next_aix > prev_aix, available_action_indices)) + 
                                          [next_aix for next_aix in [aix for tok in model._unigraph 
                                                                     for aix in self._tok2splits[tok] if tok not in observed] + 
                                                                    [self._pair2merge[pair] for pair in model._digraph
                                                                     if pair not in observed and pair in self._pair2merge]
                                           if next_aix > prev_aix])
        observed = observed.union(set(model._unigraph.keys()).union(set(model._digraph.keys())))
        return available_action_indices, observed

    # purpose: apply a trained models list of actions to a given text (operate BPE)
    # arguments:
    # - text: str, document to be tokenized
    # prereqs: a trained model
    # output: tuple (document) of strings (tokens)
    def apply_action_trace(self, text):
        mock = BPE()
        mock.init([text], method=self._init_method, apply=True) # method='char'
        prev_aix = -1; available_action_indices = []; observed = set(); tokenizing = True
        while tokenizing:
            available_action_indices, observed = self.update_action_indices(available_action_indices, mock, 
                                                                            prev_aix = prev_aix, observed = observed)
            if available_action_indices:
                aix = available_action_indices[0]
            else:
                tokenizing = False
                break
            prev_aix = aix
            action = self._action_trace[aix]
            if action.type == 'merge':
                mock.merge(action.pair)
            else:
                mock.split(action.pair)
        tks = []
        for t, idxs in mock._tok_idx.items():
            for ix in idxs:
                tks.append((t, ix))
        tks.sort(key=lambda ti: ti[1])
        tks, _ = zip(*tks)
        return tks
    
    def return_tokenization(self):
        tks = []
        for t, idxs in self._tok_idx.items():
            for ix in idxs:
                tks.append((t, ix))
        tks.sort(key=lambda ti: ti[1])
        tks, _ = zip(*tks)
        return tks

    # purpose: convert a list of token indices to a str object
    # arguments: indices: list (tokenized document) of ints (indices) to be mapped to strings and joined
    # prereqs: a trained model
    # output: str, representing the underlying form of the list of tokens
    def decode(self, indices):
        return ''.join(self.indices_to_tokens(indices))

    # purpose: convert str tokens to a list of int token indices
    # arguments:
    # - toks: list (document) of strs (tokens) to be mapped theought the model's index to indices
    # prereqs: a trained model
    # output: a list of ints (indices) representing the model's tokenized representation of the document
    def tokens_to_indices(self, toks):
        return [self._tok2ind[t] for t in toks]

    # purpose: conver a list of token indices to a list of str objects
    # arguments: indices: list (tokenized document) of ints (indices) to be mapped to strings and joined
    # prereqs: a trained model
    # output: list of strs, representing the tokenized form of the document's tokens
    def indices_to_tokens(self, indices):
        return [self._ind2tok[i] for i in indices]

# purpose: cast a BPE-based tokenizer object
# arguments: see __init__()
# use methods: 
# - __init__: intialize a BPE-based tokenizer object
# - save: (see Tokenizer)
# - init: (see Tokenizer)
# - load: (see Tokenizer)
# use attributes: none
class BPE(Tokenizer):
    # purpose: initialize a BPE-based tokenizer object
    # arguments:
    # - tok2ind: dict (see Tokenizer)
    # - covering_vocab: set, indicating a collection of strs that the tokenizer should consider as bounds for the result of all possible actions
    def __init__(self, tok2ind=None, covering_vocab = set()):
        # defining a covering vocabulary restricts merge/split-able pathways
        self._covering_vocab = covering_vocab
        self._covered = {}
        self._covering = {}
        if self._covering_vocab:
            if tok2ind:
                tok2ind = {t: i for i, t in enumerate(set(list(tok2ind.keys())+list(self._covering_vocab)))}
            else:
                tok2ind = {t: i for i, t in enumerate(self._covering_vocab)}
        # initialize token index, now that the cover is included
        super().__init__(tok2ind=tok2ind)
        # starting and ending points for each token (as a set for constant lookup)
        self._lefts = {}
        self._rights = {}
        # frequency-based information
        self._unigraph = Counter()
        self._doc_unigraph = defaultdict(Counter)
        self._digraph = Counter()
        # mapping to and from indices
        self._tok_idx = defaultdict(set)
        self._pair_idx = defaultdict(set)
        self._char2docidx = {}        

    # purpose: save a model
    # arguments:
    # - path: (see Tokenizer)
    # - data: (see Tokenizer)
    def save(self, path, data=None):
        if data is None:
            data = {}
        data['unigraph'] = dict(self._unigraph)
        data['digraph'] = [[k[0], k[1], v] for k, v in self._digraph.items()]
        data['doc_unigraph'] = {k: dict(v) for k, v in self._doc_unigraph.items()}
        data['init_method'] = self._init_method
        super(BPE, self).save(path, data=data)
#         json.dump({'docs': self._training_data, 'covering': self._training_covering}, 
#                   open(re.sub('.json', '-docs.json', path), 'w+'))

    # purpose: load a model
    # arguments:
    # - path: (see Tokenizer)
    def load(self, path):
        data = super(BPE, self).load(path)
        self._unigraph = Counter(data['unigraph'])
        self._digraph = Counter({(l, r): v for l, r, v in data['digraph']})
        self._doc_unigraph = defaultdict(Counter)
        for k, v in data['doc_unigraph'].items():
            self._doc_unigraph[k] = Counter(v)
        self._init_method = data['init_method']
        return data

    # purpose: intialize a BPE-based model
    # arguments:
    # - docs: list (corpus) of strs (document), containing the data on which the model will be trained
    # - seed: int, indicating the seed of randomization
    # - method: str, one from: 'char' (start from characters), 'warm' (start from a space-based segmentation), or 'rand' (start from a random segmentation)
    # - apply: whether or not this will be an application of the action trace post-training, which just shuts off a progress bar
    # - covering: list (corpus) of lists (documents) of strs (tokens), representing a collection of token boundaries that must be observed during learning, i.e., restricting the learnable rules.
    # - action_protect: list of strs, indicating regular expressions of that cannot be included in actions, protecting the model from, e.g., learning to merge known unwanted tokens
    # prereqs: a corpus of document to either tokenize or initialize for training
    # output: none, data are ingested and structured for learning or application of a model
    def init(self, docs, seed=None, method='char', apply=False, covering = [], action_protect = ''):
        # self._training_data = docs; self._training_covering = covering
        ##
        self._doc_counts = Counter(); ds = []; cs = []; doc_index = {}
        for di, doc in enumerate(docs): # tqdm(list(), desc = "Indexing and counting multiplicity of documents"):
            if doc not in doc_index:
                doc_index[doc] = len(ds)
                ds.append(doc)
                if covering:
                    cs.append(covering[di])
            self._doc_counts[doc_index[doc]] += docs[doc] if type(docs) == Counter else 1
        docs = ds; covering = cs
        ##
        self._init_method = method
        self._action_protect = action_protect
        ## guarentees a covering
        self._covering = {}
        self._hascover = bool(covering)
        ix = 0
        if covering:
            # assert(len(docs) == len(covering))
            for doc, segmentation in zip(docs, covering):
                for s_ix, s in enumerate(segmentation):
                    for ch in s:
                        self._covering[ix] = s_ix
                        ix += 1
                ix += 1
        d_ix = 0
        s_ix = max(self._covering.values()) if self._covering else -1
        for doc in docs:
            if d_ix + len(doc) > ix:
                s_ix += 1
            for ch in doc:
                if d_ix > ix:
                    self._covering[d_ix] = s_ix
                d_ix += 1
            d_ix += 1
        if seed:
            np.random.seed(seed=seed)
        offset = 0
        for doc_idx, doc in enumerate(docs if apply else tqdm(docs, desc=f'Initializing')):
            stream = self._init_doc(doc, method=method)
            assert (sum(map(len, stream)) == len(doc))
            for ix, tok in enumerate(stream):
                self._unigraph[tok] += self._doc_counts[doc_idx] # 1
                self._tok_idx[tok].add(offset)
                for char_idx in range(offset, offset + len(tok)):
                    self._char2docidx[char_idx] = doc_idx #####################################
                self._doc_unigraph[doc_idx][tok] += self._doc_counts[doc_idx] # 1
                tok_pair = (stream[ix - 1], tok) if ix else ('', tok)
                self._lefts[(offset - len(stream[ix - 1])) if ix else (offset - 1)] = tok_pair
                self._rights[offset] = tok_pair
                if ix:
                    self._digraph[tok_pair] += self._doc_counts[doc_idx] # 1
                    self._pair_idx[tok_pair].add(offset - len(stream[ix - 1]))
                offset += len(tok)
            tok_pair = (tok, '')
            self._lefts[offset - len(tok)] = tok_pair
            self._rights[offset] = tok_pair
            offset += 1

    # purpose: initialize a document for training or tokenization
    # arguments:
    # - d: str, representing the underlying data to be used for training, or which will be tokenized
    # - method: str, indicating method for initialization. one of: 'char' (start from characters), 'warm' (start from a space-based segmentation), or 'rand' (start from a random segmentation)
    # output: a list of strings, which will serve as starting points for tokenization by operation of actions
    @staticmethod
    def _init_doc(d, method='char'):
        if method == 'char':
            return d
        elif method == 'warm':
            return [token for token in re.split("([a-zA-Z0-9-']+)", d) if token] # tokenize(d)
        elif method == 'rand':
            topidx = sorted(set(
                [0] + sorted(np.random.choice(np.arange(1, len(d)), size=int(len(d) / 2), replace=False)) + [len(d)]))
            return [d[topidx[idx - 1]:topidx[idx]] for idx in range(1, len(topidx))]
        else:
            raise ValueError(f'Unrecognized document pre-processing method: {method}')

    # purpose: determine if a given pair can be merged, based on whether the join of its constituents is a substring of the cover
    # arguments:
    # - pair: tuple of strs, indicating two tokens adjacent to one another
    # output: boolean, indicating whether or not the pair's join would compatible with the cover
    def under_cover(self, pair): # for span-level covering
        newtok = "".join(pair)
        skip_next = False
        for i in sorted(list(self._pair_idx[pair])):
            if skip_next:  # handle odd numbers of repeated tokens
                skip_next = False
                continue
            skip_next = True if pair[0] == pair[1] and pair[1] == self._lefts[i + len(pair[0])][1] else False
            if (i in self._covering) and (i+len(newtok)-1 in self._covering):
                if self._covering[i] != self._covering[i+len(newtok)-1]:
                    return False
            elif (i in self._covering) or ((i+len(newtok)-1) in self._covering):
                return False                
        else:
            return True

    # purpose: determine if a given pair can be split, based on whether breaking its consitituents violates the cover
    # arguments:
    # - wpair: tuple of strs whose interaction with the cover will be evaluated 
    # output: bool, indicating whether or not the split degrades the efficiency of the cover
    def split_under_cover(self, wpair): # for span-level covering
        oldtok = "".join(wpair)
        locations = list(self._tok_idx[oldtok])
        for i in sorted(locations):
            if (self._covering[i] != self._covering[i+len(wpair[0])-1] or 
                self._covering[i+len(wpair[0])] != self._covering[i+len(wpair[0])+len(wpair[1])-1]):
                return False
        else:
            return True

    # purpose: determines if a given str is a token within the covering vocabulary
    # arguments: newtok: str, to be evaluated for substring status within at least one token contained in the covering vocabulary
    # output: bool, indicating if the token is a substring one from the covering vocabulary
    def is_covered(self, newtok): # for vocab-level covering
        if newtok in self._covered: 
            return self._covered[newtok]
        else:
            for cover_token in self._covering_vocab:
                if newtok in cover_token:
                    self._covered[newtok] = True
                    return self._covered[newtok]
            else:
                self._covered[newtok] = False
                return self._covered[newtok]

    # purpose: determines if token covers a token within a covering vocabulary
    # arguments: newtok: str, to be evaluated for superstring status over at least one token contained in the covering vocabulary
    # output: bool, indicating if the token is a superstring of one from the covering vocabulary
    def is_covering(self, newtok): # for vocab-level covering
        if newtok in self._covering: 
            return self._covering[newtok]
        else:
            for cover_token in self._covering_vocab:
                if cover_token in newtok:
                    self._covering[newtok] = True
                    return self._covering[newtok]
            else:
                self._covering[newtok] = False
                return self._covering[newtok]

    # purpose: trains an hr-bpe model over the system's current state of ingested data
    # arguments:
    # - num_batches: int, indicating the number of iterations of action ranking into test batches that will be operated
    # - batch_size: int, indicating the number of potentially-optimizing actions to rank per test batch (merge and split, each)
    # - actions_per_batch: int, indicating the number of optimizing actions to sample and test for inclusion as learned rules, per test batch
    # - seed: int, for control of randomization sampling
    # output: NA, method modifies the state of model parameters by learning optimizing actions
    def fit(self, num_batches, batch_size=1, actions_per_batch=None, seed=None):
        if seed:
            np.random.seed(seed=seed)

        if actions_per_batch is None:
            actions_per_batch = batch_size
        elif actions_per_batch > batch_size:
            actions_per_batch = batch_size

        pbar = tqdm(total=self._early_stop, desc = 'Fitting')
        for batch in range(num_batches): # tqdm(range(num_batches), desc='Fitting'):
            actions = self.rank_actions(self.get_actions(batch_size, actions_per_batch)) # [:batch_size]
            for action in actions:
                vsize = len(self._unigraph)
                if action.type == 'merge':
                    ## Different criteria for avoiding merges
                    newtok = "".join(action.pair)
                    if self._action_protect:
                        if re.search("("+"|".join(self._action_protect)+")", newtok): continue
                    if self._hascover:
                        if not self.under_cover(action.pair):
                            continue
                    if self._covering_vocab:
                        if (not self.is_covered(newtok)) and (not self.is_covering(newtok)):
                            continue
                    self.merge(action.pair)
                else:
                    ## Different criteria for avoiding splits
                    if self._action_protect:
                        if (re.search("("+"|".join(self._action_protect)+")", action.pair[0]) or
                            re.search("("+"|".join(self._action_protect)+")", action.pair[1])): 
                            continue
                    if self._hascover:
                        if not self.split_under_cover(action.pair):
                            continue
                    if self._covering_vocab:
                        if (((not self.is_covered(action.pair[0])) and (not self.is_covering(action.pair[0]))) or
                            ((not self.is_covered(action.pair[1])) and (not self.is_covering(action.pair[1])))):
                            continue
                    self.split(action.pair)
                    
                self._action_trace.append(action)
                pbar.update(len(self._unigraph) - vsize)
                if self.do_break_early() or not actions:
                    break
            if self.do_break_early() or not actions:
                break

        # for k in self._unigraph.keys():
        for k, v in sorted(self._unigraph.items(), key=lambda kv: kv[1], reverse=True):
            self.add_type(k)

        self._tok2acts = defaultdict(list)
        self._pair2merge = dict()
        self._tok2splits = defaultdict(list)
        for aix, a in enumerate(self._action_trace):
            if a.type =='split':
                self._tok2acts["".join(a.pair)].append(aix)
                self._tok2splits["".join(a.pair)].append(aix)
            else:
                self._pair2merge[tuple(a.pair)] = aix
                self._tok2acts[a.pair[0]].append(aix)
                self._tok2acts[a.pair[1]].append(aix)
        self._maxtoklen = max([len(t) for t in self._tok2ind])

        print(f'Built a vocabulary of {len(self)} types')

    # purpose: concatenate (merge) all adjacent pairs of a given type
    # arguments:
    # - pair: tuple of strs, indicating the types of two adjacent tokens
    # output: an updated representation of the ingested documents, with all pairs of the given type merged
    def merge(self, pair):
        newtok = "".join(pair)

        skip_next = False
        locations = list(self._pair_idx[pair])
        for i in sorted(locations):
            if skip_next:  # handle odd numbers of repeated tokens
                skip_next = False
                continue

            # gather the instance's neighbors
            lneighbor = self._rights[i][0]
            rneighbor = self._lefts[i + len(pair[0])][1]
            skip_next = True if pair[0] == pair[1] and pair[1] == rneighbor else False

            # delete the entries for this pair in both indices
            del (self._lefts[i])
            del (self._rights[i + len(pair[0])])

            # gather the old left and right adjacent pairs
            lpair = (lneighbor, pair[0])
            rpair = (pair[1], rneighbor)

            # construct new left and right adjacent pairs
            newlpair = (lneighbor, newtok)
            newrpair = (newtok, rneighbor)

            # delete the old left and right pair from both left/right indexings
            del (self._lefts[i - len(lneighbor) if lneighbor else i - 1])  # lpair
            del (self._rights[i])  # lpair
            del (self._lefts[i + len(pair[0])])  # rpair
            del (self._rights[i + len(newtok)])  # rpair

            # update both left and right indexings with the new left and right pairs
            self._lefts[i - len(lneighbor) if lneighbor else i - 1] = newlpair
            self._rights[i] = newlpair
            self._lefts[i] = newrpair
            self._rights[i + len(newtok)] = newrpair

            texti = self._char2docidx[i]
            # weight = self._doc_counts[texti]
            # only update left co-occurrences if lneighbor is non-empty
            if lneighbor:  # including deleting the lpair instance from codata
                self._digraph[newlpair] += self._doc_counts[texti] # 1
                self._digraph[lpair] -= self._doc_counts[texti] # 1
                self._pair_idx[newlpair].add(i - len(lneighbor))
                self._pair_idx[lpair].remove(i - len(lneighbor))
                if not self._digraph[lpair]:
                    del (self._digraph[lpair])
                if not self._pair_idx[lpair]:
                    del (self._pair_idx[lpair])

            # only update right co-occurrences if rneighbor is non-empty
            if rneighbor:  # including deleting rpair the instance from codata
                self._digraph[newrpair] += self._doc_counts[texti] # 1
                self._digraph[rpair] -= self._doc_counts[texti] # 1
                self._pair_idx[newrpair].add(i)
                self._pair_idx[rpair].remove(i + len(pair[0]))
                if not self._digraph[rpair]:
                    del (self._digraph[rpair])
                if not self._pair_idx[rpair]:
                    del (self._pair_idx[rpair])

            # update unigram frequencies
            self._unigraph[newtok] += self._doc_counts[texti] # 1
            self._unigraph[pair[0]] -= self._doc_counts[texti] # 1
            self._unigraph[pair[1]] -= self._doc_counts[texti] # 1
            if not self._unigraph[pair[0]]:
                del (self._unigraph[pair[0]])
            if not self._unigraph[pair[1]]:
                del (self._unigraph[pair[1]])

            # texti = self._char2docidx[i] ###############################################
            self._doc_unigraph[texti][newtok] += self._doc_counts[texti] # 1
            self._doc_unigraph[texti][pair[0]] -= self._doc_counts[texti] # 1
            if not self._doc_unigraph[texti][pair[0]]:
                del (self._doc_unigraph[texti][pair[0]])
            self._doc_unigraph[texti][pair[1]] -= self._doc_counts[texti] # 1
            if not self._doc_unigraph[texti][pair[1]]:
                del (self._doc_unigraph[texti][pair[1]])

            # update the token locations
            self._tok_idx[newtok].add(i)
            self._tok_idx[pair[0]].remove(i)
            self._tok_idx[pair[1]].remove(i + len(pair[0]))
            if not self._tok_idx[pair[0]]:
                del (self._tok_idx[pair[0]])
            if not self._tok_idx[pair[1]]:
                del (self._tok_idx[pair[1]])

            # delete the pair from the co-occurrence data record
            self._digraph[pair] -= self._doc_counts[texti] # 1
            self._pair_idx[pair].remove(i)
            if not self._pair_idx[pair]:
                del (self._pair_idx[pair])
            if not self._digraph[pair]:
                del (self._digraph[pair])

    # purpose: divide (split) all tokens of a given type into a defined pair of types (wpair)
    # arguments:
    # - wpair: tuple of strs, indicating the two adjacent types inwo which the given token is split 
    # output: an updated representation of the ingested documents, with all instances of the given type split
    def split(self, wpair):
        oldtok = "".join(wpair)
        locations = list(self._tok_idx[oldtok])
        for i in sorted(locations):
            # update the left/right and consequential digraph indices
            # wpair[0] updates
            lneighbor = self._rights[i][0]
            rneighbor = self._lefts[i][1]
            lpair = (lneighbor, oldtok)
            rpair = (oldtok, rneighbor)
            newlpair = (lneighbor, wpair[0])
            newcpair = wpair
            newrpair = (wpair[1], rneighbor)

            texti = self._char2docidx[i]
            # weight = self._doc_counts[texti]
            # cpair
            self._digraph[newcpair] += self._doc_counts[texti] # 1
            self._pair_idx[newcpair].add(i)
            self._lefts[i] = wpair
            self._rights[i + len(wpair[0])] = wpair

            # lpairs
            del (self._rights[i])
            self._rights[i] = newlpair
            del (self._lefts[i - len(lneighbor) if lneighbor else i - 1])
            self._lefts[i - len(lneighbor) if lneighbor else i - 1] = newlpair
            if lneighbor:
                self._digraph[newlpair] += self._doc_counts[texti] # 1
                self._digraph[lpair] -= self._doc_counts[texti] # 1
                self._pair_idx[newlpair].add(i - len(lneighbor))
                self._pair_idx[lpair].remove(i - len(lneighbor))
                if not self._digraph[lpair]:
                    del self._digraph[lpair]
                if not self._pair_idx[lpair]:
                    del (self._pair_idx[lpair])

            # rpairs
            # del(left_indexed_pairs[i]) # technically, this was just overwritten w/wpair and doesn't need deletion
            self._lefts[i + len(wpair[0])] = newrpair
            # del(right_indexed_pairs[i+len(oldtok)])
            self._rights[i + len(oldtok)] = newrpair
            if rneighbor:
                self._digraph[newrpair] += self._doc_counts[texti] # 1
                self._digraph[rpair] -= self._doc_counts[texti] # 1
                self._pair_idx[newrpair].add(i + len(wpair[0]))
                self._pair_idx[rpair].remove(i)
                if not self._digraph[rpair]:
                    del (self._digraph[rpair])
                if not self._pair_idx[rpair]:
                    del (self._pair_idx[rpair])

            # update unigram frequencies
            self._unigraph[oldtok] -= self._doc_counts[texti] # 1
            self._unigraph[wpair[0]] += self._doc_counts[texti] # 1
            self._unigraph[wpair[1]] += self._doc_counts[texti] # 1
            if not self._unigraph[oldtok]:
                del self._unigraph[oldtok]

            # update the token locations
            self._tok_idx[oldtok].remove(i)
            self._tok_idx[wpair[0]].add(i)
            self._tok_idx[wpair[1]].add(i + len(wpair[0]))
            if not self._tok_idx[oldtok]:
                del (self._tok_idx[oldtok])

            # texti = self._char2docidx[i]#########################################
            self._doc_unigraph[texti][oldtok] -= self._doc_counts[texti] # 1
            if not self._doc_unigraph[texti][oldtok]:
                del (self._doc_unigraph[texti][oldtok])
            self._doc_unigraph[texti][wpair[0]] += self._doc_counts[texti] # 1
            self._doc_unigraph[texti][wpair[1]] += self._doc_counts[texti] # 1

    # purpose: return a list of actions, ranked according to the given system settings (see: HRBPE from regularized, GreedyBPE from greedy)
    # arguments:
    # - batch_size: int, indicating the number of potentially-optimizing actions to rank per test batch (merge and split, each)
    # - actions_per_batch: int, indicating the number of optimizing actions to sample and test for inclusion as learned rules, per test batch
    # output: list of Action objects
    def get_actions(self, batch_size, actions_per_batch):
        raise NotImplementedError

    # purpose: rank a list of actions according to the system's given settings (see: HRBPE from regularized, GreedyBPE from greedy)
    # arguments:
    # - actions: list of Action objects
    # output: list of Action objects, ordered by decreasing sorting value, such as count 
    def rank_actions(self, actions):
        raise NotImplementedError

    # purpose: halt the given training process according to the system settings (see: HRBPE from regularized, GreedyBPE from greedy)
    # arguments: NA
    # output: boolean, indicating whether or not a stopping criterion has been reached
    def do_break_early(self):
        return False

# purpose: instantiate a standard bpe model that greedily accepts merges of highest co-frequency
# arguments: (see __init__ from base.BPE)
# prereqs: (see base.BPE)
# use methods: (see base.BPE)
# use attributes: (see base.BPE)
class GreedyBPE(BPE):
    
    # - tok2ind: (optional) dict, used by .load to set the index
    # - covering_vocab: set, indicating a collection of strs that the tokenizer should consider as bounds for the result of all possible actions
    # - early_stop: bool, with True indicating the model should stop early, i.e., once no actions are predicted to optimize the negative log likelihood
    def __init__(self, tok2ind=None, covering_vocab = set(), early_stop=1_000_000_000):
        super().__init__(tok2ind=tok2ind, covering_vocab = covering_vocab)
        self._early_stop = early_stop
    
    # purpose: return a list of actions, ranked according to the current count value for each action's pair of tokens
    # arguments:
    # - batch_size: int, indicating the number of potentially-optimizing actions to rank per test batch (merge and split, each)
    # output: list of Action objects
    def get_actions(self, batch_size, _):
        return [Action(pair, type='merge', count=cnt) for pair, cnt in self._digraph.most_common(batch_size)]

    # purpose: rank a list of actions according to the system's current count value for each action's pair of tokens
    # arguments:
    # - actions: list of Action objects
    # output: list of Action objects, ordered by decreasing sorting value, such as count 
    def rank_actions(self, actions):
        return sorted(actions, reverse=True, key=lambda a: a.count)

    # purpose: halt the given training process when the vocabulary is the desired size (equal to self._early_stop)
    # arguments: NA
    # output: boolean, indicating whether or not a stopping criterion has been reached
    def do_break_early(self):
        ## the vocabulary size exceeds the self._early_stop (limit) or largest co-frequency pair has count 1
        return((len(self._unigraph) >= self._early_stop and self._early_stop) or 
               self.get_actions(1, 1)[0].count == 1)