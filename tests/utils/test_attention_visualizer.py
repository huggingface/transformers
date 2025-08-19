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

import io
import sys
import unittest

from transformers.testing_utils import require_read_token, require_torch
from transformers.utils.attention_visualizer import AttentionMaskVisualizer


@require_torch
@require_read_token
class AttentionMaskVisualizerTester(unittest.TestCase):
    """Test suite for AttentionMaskVisualizer"""

    def test_paligemma_multimodal_visualization(self):
        """Test AttentionMaskVisualizer with PaliGemma multimodal model"""
        model_name = "google/paligemma2-3b-mix-224"
        input_text = "<img> What is in this image?"

        # capture the output, can be messy
        captured_output = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured_output

        visualizer = AttentionMaskVisualizer(model_name)
        visualizer(input_text)

        sys.stdout = original_stdout
        output = captured_output.getvalue()

        expected_output = """
##########################################################################################################################################################################################################################################
##                                                                         Attention visualization for \033[1mpaligemma:google/paligemma2-3b-mix-224\033[0m PaliGemmaModel                                                                            ##
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

        self.assertEqual(output.strip(), expected_output.strip())

    def test_llama_text_only_visualization(self):
        """Test AttentionMaskVisualizer with Llama text-only model"""
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
        input_text = "Plants create energy through a process known as"

        captured_output = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured_output

        visualizer = AttentionMaskVisualizer(model_name)
        visualizer(input_text)

        sys.stdout = original_stdout
        output = captured_output.getvalue()

        expected_output = """
##########################################################################################################################################################################################################
##                                                           Attention visualization for \033[1mllama:meta-llama/Llama-3.2-1B-Instruct\033[0m LlamaModel                                                              ##
##########################################################################################################################################################################################################
 \033[92m■\033[0m: i == j (diagonal)   \033[93m■\033[0m: token_type_ids
               Attention Matrix

'Pl'      :  0 \033[92m■\033[0m ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚    |    
'ants'    :  1 ■ \033[92m■\033[0m ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚    |    
'Ġcreate' :  2 ■ ■ \033[92m■\033[0m ⬚ ⬚ ⬚ ⬚ ⬚ ⬚    |    
'Ġenergy' :  3 ■ ■ ■ \033[92m■\033[0m ⬚ ⬚ ⬚ ⬚ ⬚    |    
'Ġthrough':  4 ■ ■ ■ ■ \033[92m■\033[0m ⬚ ⬚ ⬚ ⬚    |    
'Ġa'      :  5 ■ ■ ■ ■ ■ \033[92m■\033[0m ⬚ ⬚ ⬚    |    
'Ġprocess':  6 ■ ■ ■ ■ ■ ■ \033[92m■\033[0m ⬚ ⬚    |    
'Ġknown'  :  7 ■ ■ ■ ■ ■ ■ ■ \033[92m■\033[0m ⬚    |    
'Ġas'     :  8 ■ ■ ■ ■ ■ ■ ■ ■ \033[92m■\033[0m    |    
##########################################################################################################################################################################################################
"""  # noqa

        self.assertEqual(output.strip(), expected_output.strip())
