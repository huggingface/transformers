from typing import Union

from colbert import Searcher
from colbert.data import Queries
from colbert.infra.config import ColBERTConfig


TextQueries = Union[str, 'list[str]', 'dict[int, str]', Queries]


class HopSearcher(Searcher):
    def __init__(self, *args, config=None, interaction='flipr', **kw_args):
        defaults = ColBERTConfig(query_maxlen=64, interaction=interaction)
        config = ColBERTConfig.from_existing(defaults, config)

        super().__init__(*args, config=config, **kw_args)

    def encode(self, text: TextQueries, context: TextQueries):
        queries = text if type(text) is list else [text]
        context = context if context is None or type(context) is list else [context]
        bsize = 128 if len(queries) > 128 else None

        self.checkpoint.query_tokenizer.query_maxlen = self.config.query_maxlen
        Q = self.checkpoint.queryFromText(queries, context=context, bsize=bsize, to_cpu=True)

        return Q

    def search(self, text: str, context: str, k=10):
        return self.dense_search(self.encode(text, context), k)

    def search_all(self, queries: TextQueries, context: TextQueries, k=10):
        queries = Queries.cast(queries)
        context = Queries.cast(context) if context is not None else context

        queries_ = list(queries.values())
        context_ = list(context.values()) if context is not None else context

        Q = self.encode(queries_, context_)

        return self._search_all_Q(queries, Q, k)
