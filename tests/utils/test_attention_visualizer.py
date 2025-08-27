# Copyright 2025 The HuggingFace Inc. team.
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

import builtins
import io
import re
import unittest

from transformers.testing_utils import require_read_token, require_torch
from transformers.utils.attention_visualizer import AttentionMaskVisualizer


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _normalize(s: str) -> str:
    # drop ANSI (colors may be disabled on CI), normalize line endings,
    # and strip trailing spaces without touching alignment inside lines
    s = ANSI_RE.sub("", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in s.split("\n")).strip()


@require_torch
class AttentionMaskVisualizerTester(unittest.TestCase):
    """Test suite for AttentionMaskVisualizer"""

    @require_read_token
    def test_paligemma_multimodal_visualization(self):
        """Test AttentionMaskVisualizer with PaliGemma multimodal model"""
        model_name = "hf-internal-testing/namespace_google_repo_name_paligemma-3b-pt-224"
        input_text = "<img> What is in this image?"

        buf = io.StringIO()
        orig_print = builtins.print

        def _print(*args, **kwargs):
            kwargs.setdefault("file", buf)
            orig_print(*args, **kwargs)

        try:
            builtins.print = _print
            visualizer = AttentionMaskVisualizer(model_name)
            visualizer(input_text)
        finally:
            builtins.print = orig_print
        output = buf.getvalue()

        expected_output = """
##########################################################################################################################################################################################################################################
##                                                      Attention visualization for \033[1mpaligemma:hf-internal-testing/namespace_google_repo_name_paligemma-3b-pt-224\033[0m PaliGemmaModel                                                         ##
##########################################################################################################################################################################################################################################
 \033[92m■\033[0m: i == j (diagonal)   \033[93m■\033[0m: token_type_ids
              Attention Matrix  


\033[93m'<image>'\033[0m:  0 \033[93m■\033[0m ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚    |    
\033[93m'<image>'\033[0m:  1 \033[93m■\033[0m \033[93m■\033[0m \033[93m■\033[0m \033[93m■\033[0m \033[93m■\033[0m ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚    |    
\033[93m'<image>'\033[0m:  2 \033[93m■\033[0m \033[93m■\033[0m \033[93m■\033[0m \033[93m■\033[0m \033[93m■\033[0m ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚    |    
\033[93m'<image>'\033[0m:  3 \033[93m■\033[0m \033[93m■\033[0m \033[93m■\033[0m \033[93m■\033[0m \033[93m■\033[0m ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚    |    
\033[93m'<image>'\033[0m:  4 \033[93m■\033[0m \033[93m■\033[0m \033[93m■\033[0m \033[93m■\033[0m \033[93m■\033[0m ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚    |    
'<bos>'  :  5 ■ ■ ■ ■ ■ \033[92m■\033[0m ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚    |    
'▁What'  :  6 ■ ■ ■ ■ ■ ■ \033[92m■\033[0m ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚    |    
'▁is'    :  7 ■ ■ ■ ■ ■ ■ ■ \033[92m■\033[0m ⬚ ⬚ ⬚ ⬚ ⬚ ⬚    |    
'▁in'    :  8 ■ ■ ■ ■ ■ ■ ■ ■ \033[92m■\033[0m ⬚ ⬚ ⬚ ⬚ ⬚    |    
'▁this'  :  9 ■ ■ ■ ■ ■ ■ ■ ■ ■ \033[92m■\033[0m ⬚ ⬚ ⬚ ⬚    |    
'▁image' : 10 ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ \033[92m■\033[0m ⬚ ⬚ ⬚    |    
'?'      : 11 ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ \033[92m■\033[0m ⬚ ⬚    |    
'\\n'     : 12 ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ \033[92m■\033[0m ⬚    |    
'<eos>'  : 13 ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ \033[92m■\033[0m    |    
##########################################################################################################################################################################################################################################
"""  # noqa

        self.assertEqual(_normalize(output), _normalize(expected_output))

    @require_read_token
    def test_llama_text_only_visualization(self):
        """Test AttentionMaskVisualizer with Llama text-only model"""
        model_name = "hf-internal-testing/namespace_meta-llama_repo_name_Llama-2-7b-hf"
        input_text = "Plants create energy through a process known as"

        buf = io.StringIO()
        orig_print = builtins.print

        def _print(*args, **kwargs):
            kwargs.setdefault("file", buf)
            orig_print(*args, **kwargs)

        try:
            builtins.print = _print
            visualizer = AttentionMaskVisualizer(model_name)
            visualizer(input_text)
        finally:
            builtins.print = orig_print
        output = buf.getvalue()

        expected_output = """
##########################################################################################################################################################################################################
##                                           Attention visualization for \033[1mllama:hf-internal-testing/namespace_meta-llama_repo_name_Llama-2-7b-hf\033[0m LlamaModel                                              ##
##########################################################################################################################################################################################################
 \033[92m■\033[0m: i == j (diagonal)   \033[93m■\033[0m: token_type_ids
               Attention Matrix

'▁Pl'     :  0 \033[92m■\033[0m ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚    |    
'ants'    :  1 ■ \033[92m■\033[0m ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚    |    
'▁create' :  2 ■ ■ \033[92m■\033[0m ⬚ ⬚ ⬚ ⬚ ⬚ ⬚    |    
'▁energy' :  3 ■ ■ ■ \033[92m■\033[0m ⬚ ⬚ ⬚ ⬚ ⬚    |    
'▁through':  4 ■ ■ ■ ■ \033[92m■\033[0m ⬚ ⬚ ⬚ ⬚    |    
'▁a'      :  5 ■ ■ ■ ■ ■ \033[92m■\033[0m ⬚ ⬚ ⬚    |    
'▁process':  6 ■ ■ ■ ■ ■ ■ \033[92m■\033[0m ⬚ ⬚    |    
'▁known'  :  7 ■ ■ ■ ■ ■ ■ ■ \033[92m■\033[0m ⬚    |    
'▁as'     :  8 ■ ■ ■ ■ ■ ■ ■ ■ \033[92m■\033[0m    |    
##########################################################################################################################################################################################################
"""  # noqa

        self.assertEqual(_normalize(output), _normalize(expected_output))
