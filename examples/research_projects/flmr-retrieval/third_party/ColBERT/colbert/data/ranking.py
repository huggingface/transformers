import os
import tqdm
import ujson
from colbert.infra.provenance import Provenance

from colbert.infra.run import Run
from colbert.utils.utils import print_message, groupby_first_item
from utility.utils.save_metadata import get_metadata_only


def numericize(v):
    if '.' in v:
        return float(v)

    return int(v)


def load_ranking(path):  # works with annotated and un-annotated ranked lists
    print_message("#> Loading the ranked lists from", path)

    with open(path) as f:
        return [list(map(numericize, line.strip().split('\t'))) for line in f]


class Ranking:
    def __init__(self, path=None, data=None, metrics=None, provenance=None):
        self.__provenance = provenance or path or Provenance()
        self.data = self._prepare_data(data or self._load_file(path))

    def provenance(self):
        return self.__provenance
    
    def toDict(self):
        return {'provenance': self.provenance()}

    def _prepare_data(self, data):
        # TODO: Handle list of lists???
        if isinstance(data, dict):
            self.flat_ranking = [(qid, *rest) for qid, subranking in data.items() for rest in subranking]
            return data

        self.flat_ranking = data
        return groupby_first_item(tqdm.tqdm(self.flat_ranking))

    def _load_file(self, path):
        return load_ranking(path)

    def todict(self):
        return dict(self.data)

    def tolist(self):
        return list(self.flat_ranking)

    def items(self):
        return self.data.items()

    def _load_tsv(self, path):
        raise NotImplementedError

    def _load_jsonl(self, path):
        raise NotImplementedError

    def save(self, new_path):
        assert 'tsv' in new_path.strip('/').split('/')[-1].split('.'), "TODO: Support .json[l] too."

        with Run().open(new_path, 'w') as f:
            for items in self.flat_ranking:
                line = '\t'.join(map(lambda x: str(int(x) if type(x) is bool else x), items)) + '\n'
                f.write(line)

            output_path = f.name
            print_message(f"#> Saved ranking of {len(self.data)} queries and {len(self.flat_ranking)} lines to {f.name}")
        
        with Run().open(f'{new_path}.meta', 'w') as f:
            d = {}
            d['metadata'] = get_metadata_only()
            d['provenance'] = self.provenance()
            line = ujson.dumps(d, indent=4)
            f.write(line)
        
        return output_path

    @classmethod
    def cast(cls, obj):
        if type(obj) is str:
            return cls(path=obj)

        if isinstance(obj, dict) or isinstance(obj, list):
            return cls(data=obj)

        if type(obj) is cls:
            return obj

        assert False, f"obj has type {type(obj)} which is not compatible with cast()"
