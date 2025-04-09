# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import gc
import tempfile
import unittest
# import sys
# sys.path.insert(0,"/home/wenhuach/transformers/src")

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.testing_utils import (
    backend_empty_cache,
    # require_accelerate,
    # require_intel_extension_for_pytorch,
    # require_torch_accelerator,
    # require_torch_gpu,
    # require_torch_multi_gpu,
    # slow,
    torch_device,
)
from transformers.utils import is_accelerate_available, is_torch_available


if is_torch_available():
    import torch

if is_accelerate_available():
    from accelerate import init_empty_weights



class AutoRoundTest(unittest.TestCase):
    # called only once for all test in this class
    @classmethod
    def setUpClass(cls):
        """
        Setup quantized model
        """
        pass

    def tearDown(self):
        gc.collect()
        backend_empty_cache(torch_device)
        gc.collect()

    def test_inference(self): ##TODO use quantized model directly
        model_name = "facebook/opt-125m"
        from auto_round import AutoRound

        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        quantizer = AutoRound(model=model,tokenizer=tokenizer,iters=1)
        output_dir = "./tmp_autoround"
        quantizer.quantize_and_save(output_dir)

        q_model = AutoModelForCausalLM.from_pretrained(output_dir,
                                                     device_map="auto")
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(q_model.device)
        print(tokenizer.decode(q_model.generate(**inputs, max_new_tokens=50)[0]))
        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)

    def test_infer_gptq(self):
        model_name = "ybelkada/opt-125m-gptq-4bit"
        from transformers import AutoRoundConfig
        quantization_config = AutoRoundConfig()

        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     device_map="auto",quantization_config=quantization_config)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        print(tokenizer.decode(model.generate(**inputs, max_new_tokens=5)[0]))



if __name__ == "__main__":
    unittest.main()




