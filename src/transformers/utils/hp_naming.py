# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import re


class TrialShortNamer:
    PREFIX = "hp"
    DEFAULTS = {}
    NAMING_INFO = None

    @classmethod
    def set_defaults(cls, prefix, defaults):
        cls.PREFIX = prefix
        cls.DEFAULTS = defaults
        cls.build_naming_info()

    @staticmethod
    def shortname_for_word(info, word):
        if len(word) == 0:
            return ""
        short_word = None
        if any(char.isdigit() for char in word):
            raise Exception(f"Parameters should not contain numbers: '{word}' contains a number")
        if word in info["short_word"]:
            return info["short_word"][word]
        for prefix_len in range(1, len(word) + 1):
            prefix = word[:prefix_len]
            if prefix in info["reverse_short_word"]:
                continue
            else:
                short_word = prefix
                break

        if short_word is None:
            # Paranoid fallback
            def int_to_alphabetic(integer):
                s = ""
                while integer != 0:
                    s = chr(ord("A") + integer % 10) + s
                    integer //= 10
                return s

            i = 0
            while True:
                sword = word + "#" + int_to_alphabetic(i)
                if sword in info["reverse_short_word"]:
                    continue
                else:
                    short_word = sword
                    break

        info["short_word"][word] = short_word
        info["reverse_short_word"][short_word] = word
        return short_word

    @staticmethod
    def shortname_for_key(info, param_name):
        words = param_name.split("_")

        shortname_parts = [TrialShortNamer.shortname_for_word(info, word) for word in words]

        # We try to create a separatorless short name, but if there is a collision we have to fallback
        # to a separated short name
        separators = ["", "_"]

        for separator in separators:
            shortname = separator.join(shortname_parts)
            if shortname not in info["reverse_short_param"]:
                info["short_param"][param_name] = shortname
                info["reverse_short_param"][shortname] = param_name
                return shortname

        return param_name

    @staticmethod
    def add_new_param_name(info, param_name):
        short_name = TrialShortNamer.shortname_for_key(info, param_name)
        info["short_param"][param_name] = short_name
        info["reverse_short_param"][short_name] = param_name

    @classmethod
    def build_naming_info(cls):
        if cls.NAMING_INFO is not None:
            return

        info = dict(
            short_word={},
            reverse_short_word={},
            short_param={},
            reverse_short_param={},
        )

        field_keys = list(cls.DEFAULTS.keys())

        for k in field_keys:
            cls.add_new_param_name(info, k)

        cls.NAMING_INFO = info

    @classmethod
    def shortname(cls, params):
        cls.build_naming_info()
        assert cls.PREFIX is not None
        name = [copy.copy(cls.PREFIX)]

        for k, v in params.items():
            if k not in cls.DEFAULTS:
                raise Exception(f"You should provide a default value for the param name {k} with value {v}")
            if v == cls.DEFAULTS[k]:
                # The default value is not added to the name
                continue

            key = cls.NAMING_INFO["short_param"][k]

            if isinstance(v, bool):
                v = 1 if v else 0

            sep = "" if isinstance(v, (int, float)) else "-"
            e = f"{key}{sep}{v}"
            name.append(e)

        return "_".join(name)

    @classmethod
    def parse_repr(cls, repr):
        repr = repr[len(cls.PREFIX) + 1 :]
        if repr == "":
            values = []
        else:
            values = repr.split("_")

        parameters = {}

        for value in values:
            if "-" in value:
                p_k, p_v = value.split("-")
            else:
                p_k = re.sub("[0-9.]", "", value)
                p_v = float(re.sub("[^0-9.]", "", value))

            key = cls.NAMING_INFO["reverse_short_param"][p_k]

            parameters[key] = p_v

        for k in cls.DEFAULTS:
            if k not in parameters:
                parameters[k] = cls.DEFAULTS[k]

        return parameters
