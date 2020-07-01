Tokenizer summary
-----------------

In this page, we will have a closer look at tokenization. As we saw in
:doc:`the preprocessing tutorial <preprocessing>`, tokenizing a text is splitting it into words or subwords, which then
are converted to ids. The second part is pretty straightforward, here we will focus on the first part. More
specifically, we will look at the three main different kinds of tokenizers used in 🤗 Transformers: Byte Pair
Encoding (BPE), WordPiece and SentencePiece, and provide examples of models using each of those.

Note that on each model page, you can look at the documentation of the associated tokenizer to know which of those
algorithms the pretrained model used. For instance, if we look at :class:`~transformers.BertTokenizer`, we can see it's
using WordPiece.

Introduction to tokenization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Splitting a text in smaller chunks is a task that's harder than it looks, and there are multiple ways of doing it. For
instance, let's look at the sentence "Don't you love 🤗 Transformers? We sure do." A first simple way of tokenizing
this text is just to split it by spaces, which would give:

::

    ["Don't", "you", "love", "🤗", "Transformers?", "We", "sure", "do."]

This is a nice first step, but if we look at the tokens "Transformers?" or "do.", we can see we can do better. Those
will be different than the tokens "Transformers" and "do" for our model, so we should probably take the punctuation
into account. This would give:

::

    ["Don", "'", "t", "you", "love", "🤗", "Transformers", "?", "We", "sure", "do", "."]

which is better already. One thing that is annoying though is how it dealt with "Don't". "Don't" stands for do not, so
it should probably be better tokenized as "Do", "n't". This is where it can start to be more complicated, and you'll
start to understand why each kind of model as its own tokenizer. Depending on the rules we apply to split our texts
into tokens, we'll get different tokenized versions of the same text. And of course, a given pretrained model won't
perform properly if you don't use the exact same rules as the persons who pretrained it.

`spaCy <https://spacy.io/>`__ and `Moses <http://www.statmt.org/moses/?n=Development.GetStarted>`__ are two popular
rule-based tokenizers. On the text above, they'd output something like:

::

    ["Do", "n't", "you", "love", "🤗", "Transformers", "?", "We", "sure", "do", "."]

Space/punctuation-tokenization and rule-based tokenization are both examples of word tokenization, which is splitting a
sentence into words. While it's the most intuitive way to separate texts in smaller chunks, it can have a problem when
you have a huge corpus: it usually yields a very big vocabulary. :doc:`Transformer XL <model_doc/transformerxl>` for
instance uses space/punctuation-tokenization, and has a vocabulary size of 267,735!

A huge vocabulary size means a huge embedding matrix, which will cause memory problem. TransformerXL deals with it by
using a special kind of embeddings called adaptive embeddings, but in general, transformers model rarely have a
vocabulary size greater than 50,000, especially if they are trained on one language.

So if tokenizing on words is unsatisfactory, we could go on the opposite direction and simply tokenize on characters.
While it's very simple and would save a lot of memory, this doesn't allow the model to learn representations as
meaningful as with words, leading to a loss of performance. So to get the best of both world, all tokenizer models use
a hybrid between word-level and character-level tokenization, using subword tokenization.

Subword tokenization
^^^^^^^^^^^^^^^^^^^^

Subword tokenization algorithms rely on the principle that most common words should be left as is, but less frequent
words should be decomposed in subword units. For instance "annoyingly" might be considered a rare word and decomposed
as "annoying" and "ly". This is especially useful in agglutinative languages such as Turkish, where you can form
(almost) arbitrarily long complex words by stringing together some subwords.

This allows the model to keep a reasonable vocabulary while still learning useful representations for common words or
subwords. This also give the ability to the model to process words it has never seen before, by decomposing them into
subwords it knows. For instance, the base :class:`~transformers.BertTokenizer` will tokenize "I have a new GPU!" like
this:

::

    >>> from transformers import BertTokenizer
    >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    >>> tokenizer.tokenize("I have a new GPU!")
    ['i', 'have', 'a', 'new', 'gp', '##u', '!']

Since we are considering the uncased model, the sentence was lowercased first. Then all the words were present in the
vocabulary of the tokenizer, except for "gpu", so the tokenizer split it in subwords it knows: "gp" and "##u". The "##"
means that the rest of the token should be attached to the previous one, without space (for when we need to decode
predictions and reverse the tokenization).

Another example is when we use the base :class:`~transformers.XLNetTokenizer` to tokenize our previous text:

::

    >>> from transformers import XLNetTokenizer
    >>> tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    >>> tokenizer.tokenize("Don't you love 🤗 Transformers? We sure do.")
    ['▁Don', "'", 't', '▁you', '▁love', '▁', '🤗', '▁', 'Transform', 'ers', '?', '▁We', '▁sure', '▁do', '.']

We'll get back to the meaning of those '▁' when we look at SentencePiece but you can see Transformers has been split
into "Transform" and "ers".

Let's now look at how the different subword tokenization algorithms work.

.. _byte-pair-encoding:

Byte-Pair Encoding
~~~~~~~~~~~~~~~~~~

Byte-Pair Encoding was introduced in `this paper <https://arxiv.org/abs/1508.07909>`__ It relies on a pretokenizer
splitting the training data into words, which can be a simple space tokenization
(:doc:`GPT-2 <model_doc/gpt2>` and :doc:`Roberta <model_doc/roberta>` uses this for instance) or a rule-based tokenizer
(:doc:`XLM <model_doc/xlm>` uses Moses for most languages, :doc:`GPT <model_doc/gpt>` uses Spacy and ftfy) and,
beginning from the list of all characters, will learn merge rules until it has learned a vocabulary of the desired
size (this is a hyperparameter to pick).

Let's say that after the pre-tokenization we have the following words (the number indicating the frequency of each
word):

::

    ('hug', 10), ('pug', 5), ('pun', 12), ('bun', 4), ('hugs', 5)

Then the base vocabulary is ['b', 'g', 'h', 'n', 'p', 's', 'u'] and all our words are first split by character:

::

    ('h' 'u' 'g', 10), ('p' 'u' 'g', 5), ('p' 'u' 'n', 12), ('b' 'u' 'n', 4), ('h' 'u' 'g' 's', 5)

We then take each pair of symbols and look at the most frequent. For instance 'hu' is present `10 + 5 = 15` times (10
times in the 10 occurrences of 'hug', 5 times in the 5 occurrences of 'hugs'). The most frequent here is 'ug', present
`10 + 5 + 2 + 5 = 22` times in total. So the first merge rule the tokenizer learns is to group all 'u' and 'g' together
then it adds 'ug' to the vocabulary. Our corpus then becomes

::

    ('h' 'ug', 10), ('p' 'ug', 5), ('p' 'u' 'n', 12), ('b' 'u' 'n', 4), ('h' 'ug' 's', 5)

and we continue by looking at the next most common pair of symbols. It's 'un', present 16 times, so we merge those two
and add 'un' to the vocabulary. Then it's 'hug' (as 'h' + 'ug'), present 15 times, so we merge those two and add 'hug'
to the vocabulary.

At this stage, the vocabulary is ['b', 'g', 'h', 'n', 'p', 's', 'u', 'ug', 'un', 'hug'] and our corpus is represented
as

::

    ('hug', 10), ('p' 'ug', 5), ('p' 'un', 12), ('b' 'un', 4), ('hug' 's', 5)

If we stop there, the tokenizer can apply the rules it learned to new words (as long as they don't have characters that
are in the base vocabulary). For instance 'bug' would be tokenized as ['b', 'ug'] but mug would be tokenized as
['<unk>', 'ug'] since the 'm' is not in the base vocabulary. This doesn't happen to letters in general, but to special
characters like emoticons.

As we said before, the vocabulary size (which is the base vocabulary size + the number of merges) is a hyperparameter
to choose. For instance :doc:`GPT <model_doc/gpt>` has a vocabulary size of 40,478 since they have 478 base characters
and chose to stop the training of the tokenizer at 40,000 merges.

Byte-level BPE
^^^^^^^^^^^^^^

To deal with the fact the base vocabulary needs to get all base characters, which can be quite big if one allows for
all unicode characters, the
`GPT-2 paper <https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>`__
introduces a clever trick, which is to use bytes as the base vocabulary (which gives a size of 256). With some
additional rules to deal with punctuation, this manages to be able to tokenize every text without needing an unknown
token. For instance, the :doc:`GPT-2 model <model_doc/gpt>` has a vocabulary size of 50,257, which corresponds to the
256 bytes base tokens, a special end-of-text token and the symbols learned with 50,000 merges.

.. _wordpiece:

WordPiece
=========

WordPiece is the subword tokenization algorithm used for :doc:`BERT <model_doc/bert>` (as well as
:doc:`DistilBERT <model_doc/distilbert>` and :doc:`Electra <model_doc/electra>`) and was outlined in
`this paper <https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/37842.pdf>`__. It relies
on the same base as BPE, which is to initialize the vocabulary to every character present in the corpus and
progressively learn a given number of merge rules, the difference is that it doesn't choose the pair that is the most
frequent but the one that maximizes the likelihood on the corpus once merged.

That means, in the previous example, we would only merge 'u' and 'g' if the probability of having 'ug' minus the
probability of having 'u' then 'g' is greater than for any other pair of symbols.