import os
import torch
import ujson
import dataclasses

from typing import Any
from collections import defaultdict
from dataclasses import dataclass, fields
from colbert.utils.utils import timestamp, torch_load_dnn

from utility.utils.save_metadata import get_metadata_only


@dataclass
class DefaultVal:
    val: Any


@dataclass
class CoreConfig:
    def __post_init__(self):
        """
        Source: https://stackoverflow.com/a/58081120/1493011
        """

        self.assigned = {}

        for field in fields(self):
            field_val = getattr(self, field.name)

            if isinstance(field_val, DefaultVal) or field_val is None:
                setattr(self, field.name, field.default.val)

            if not isinstance(field_val, DefaultVal):
                self.assigned[field.name] = True
    
    def assign_defaults(self):
        for field in fields(self):
            setattr(self, field.name, field.default.val)
            self.assigned[field.name] = True

    def configure(self, ignore_unrecognized=True, **kw_args):
        ignored = set()

        for key, value in kw_args.items():
            self.set(key, value, ignore_unrecognized) or ignored.update({key})

        return ignored

        """
        # TODO: Take a config object, not kw_args.

        for key in config.assigned:
            value = getattr(config, key)
        """

    def set(self, key, value, ignore_unrecognized=False):
        if hasattr(self, key):
            setattr(self, key, value)
            self.assigned[key] = True
            return True

        if not ignore_unrecognized:
            raise Exception(f"Unrecognized key `{key}` for {type(self)}")

    def help(self):
        print(ujson.dumps(dataclasses.asdict(self), indent=4))

    def __export_value(self, v):
        v = v.provenance() if hasattr(v, 'provenance') else v

        if isinstance(v, list) and len(v) > 100:
            v = v[:3] #(f"list with {len(v)} elements starting with...", v[:3])

        if isinstance(v, dict) and len(v) > 100:
            v = list(v.keys())[:3] #(f"dict with {len(v)} keys starting with...", list(v.keys())[:3])

        return v

    def export(self):
        d = dataclasses.asdict(self)

        for k, v in d.items():
            d[k] = self.__export_value(v)

        return d
