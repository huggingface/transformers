# Copyright 2024 The HuggingFace Team and the AllenNLP authors. All rights reserved.
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
"""
Benchmark for different cache types used in model generation
"""
import json
import os
from typing import Optional

import torch
import torch._dynamo.config
import torch._inductor.config
from benchmark_utils_generic import BenchMark, SpeedBenchMark

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, StaticCache, set_seed


os.environ["TOKENIZERS_PARALLELISM"] = "False"


# Does multinomial sampling without a cuda synchronization
def multinomial_sample_one_no_sync(probs_sort):
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[:, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
# Experimental feature to reduce compilation times, will be on by default in future
# torch._inductor.config.fx_graph_cache = True

torch.set_printoptions(linewidth=200)  # you can better see how the mask is shaped


class CacheBenchMark(BenchMark):
    def __init__(
        self,
        repo_id,
        prefill_num_iter=3,
        num_iter=100,
        device="cuda",
        attn_implementation="sdpa",
        dtype="bfloat16",
    ):
        super().__init__()

        self._buffer["init_kwargs"]["repo_id"] = repo_id
        self._buffer["init_kwargs"]["prefill_num_iter"] = prefill_num_iter
        self._buffer["init_kwargs"]["num_iter"] = num_iter
        self._buffer["init_kwargs"]["device"] = device
        self._buffer["init_kwargs"]["attn_implementation"] = attn_implementation
        self._buffer["init_kwargs"]["dtype"] = dtype

        self.repo_id = repo_id
        self.prefill_num_iter = prefill_num_iter
        self.num_iter = num_iter
        self.device = device
        self.dtype = dtype

        dtype = getattr(torch, dtype)

        token = os.getenv("HF_HUB_READ_TOKEN", None)
        self.tokenizer = AutoTokenizer.from_pretrained(repo_id, padding_side="left", pad_token="<s>", token=token)
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                repo_id, torch_dtype=dtype, attn_implementation=attn_implementation, token=token
            )
            .to(device, dtype)
            .eval()
        )

        # FRANCE_ARTICLE
        self.input_text = """Marseille, France (CNN)The French prosecutor leading an investigation into the crash of Germanwings Flight 9525 insisted Wednesday that he was not aware of any video footage from on board the plane. Marseille prosecutor Brice Robin told CNN that "so far no videos were used in the crash investigation." He added, "A person who has such a video needs to immediately give it to the investigators." Robin\'s comments follow claims by two magazines, German daily Bild and French Paris Match, of a cell phone video showing the harrowing final seconds from on board Germanwings Flight 9525 as it crashed into the French Alps. All 150 on board were killed. Paris Match and Bild reported that the video was recovered from a phone at the wreckage site. The two publications described the supposed video, but did not post it on their websites. The publications said that they watched the video, which was found by a source close to the investigation. \"One can hear cries of 'My God' in several languages,\" Paris Match reported. "Metallic banging can also be heard more than three times, perhaps of the pilot trying to open the cockpit door with a heavy object.  Towards the end, after a heavy shake, stronger than the others, the screaming intensifies. Then nothing." "It is a very disturbing scene," said Julian Reichelt, editor-in-chief of Bild online. An official with France's accident investigation agency, the BEA, said the agency is not aware of any such video. Lt. Col. Jean-Marc Menichini, a French Gendarmerie spokesman in charge of communications on rescue efforts around the Germanwings crash site, told CNN that the reports were "completely wrong" and "unwarranted." Cell phones have been collected at the site, he said, but that they "hadn\'t been exploited yet." Menichini said he believed the cell phones would need to be sent to the Criminal Research Institute in Rosny sous-Bois, near Paris, in order to be analyzed by specialized technicians working hand-in-hand with investigators. But none of the cell phones found so far have been sent to the institute, Menichini said. Asked whether staff involved in the search could have leaked a memory card to the media, Menichini answered with a categorical "no." Reichelt told "Erin Burnett: Outfront" that he had watched the video and stood by the report, saying Bild and Paris Match are "very confident" that the clip is real. He noted that investigators only revealed they\'d recovered cell phones from the crash site after Bild and Paris Match published their reports. "That is something we did not know before. ... Overall we can say many things of the investigation weren't revealed by the investigation at the beginning," he said. What was mental state of Germanwings co-pilot? German airline Lufthansa confirmed Tuesday that co-pilot Andreas Lubitz had battled depression years before he took the controls of Germanwings Flight 9525, which he's accused of deliberately crashing last week in the French Alps. Lubitz told his Lufthansa flight training school in 2009 that he had a "previous episode of severe depression," the airline said Tuesday. Email correspondence between Lubitz and the school discovered in an internal investigation, Lufthansa said, included medical documents he submitted in connection with resuming his flight training. The announcement indicates that Lufthansa, the parent company of Germanwings, knew of Lubitz's battle with depression, allowed him to continue training and ultimately put him in the cockpit. Lufthansa, whose CEO Carsten Spohr previously said Lubitz was 100% fit to fly, described its statement Tuesday as a "swift and seamless clarification" and said it was sharing the information and documents -- including training and medical records -- with public prosecutors. Spohr traveled to the crash site Wednesday, where recovery teams have been working for the past week to recover human remains and plane debris scattered across a steep mountainside. He saw the crisis center set up in Seyne-les-Alpes, laid a wreath in the village of Le Vernet, closer to the crash site, where grieving families have left flowers at a simple stone memorial. Menichini told CNN late Tuesday that no visible human remains were left at the site but recovery teams would keep searching. French President Francois Hollande, speaking Tuesday, said that it should be possible to identify all the victims using DNA analysis by the end of the week, sooner than authorities had previously suggested. In the meantime, the recovery of the victims' personal belongings will start Wednesday, Menichini said. Among those personal belongings could be more cell phones belonging to the 144 passengers and six crew on board. Check out the latest from our correspondents . The details about Lubitz's correspondence with the flight school during his training were among several developments as investigators continued to delve into what caused the crash and Lubitz's possible motive for downing the jet. A Lufthansa spokesperson told CNN on Tuesday that Lubitz had a valid medical certificate, had passed all his examinations and "held all the licenses required." Earlier, a spokesman for the prosecutor\'s office in Dusseldorf, Christoph Kumpa, said medical records reveal Lubitz suffered from suicidal tendencies at some point before his aviation career and underwent psychotherapy before he got his pilot's license. Kumpa emphasized there's no evidence suggesting Lubitz was suicidal or acting aggressively before the crash. Investigators are looking into whether Lubitz feared his medical condition would cause him to lose his pilot's license, a European government official briefed on the investigation told CNN on Tuesday. While flying was "a big part of his life," the source said, it\'s only one theory being considered. Another source, a law enforcement official briefed on the investigation, also told CNN that authorities believe the primary motive for Lubitz to bring down the plane was that he feared he would not be allowed to fly because of his medical problems. Lubitz's girlfriend told investigators he had seen an eye doctor and a neuropsychologist, both of whom deemed him unfit to work recently and concluded he had psychological issues, the European government official said. But no matter what details emerge about his previous mental health struggles, there's more to the story, said Brian Russell, a forensic psychologist. "Psychology can explain why somebody would turn rage inward on themselves about the fact that maybe they weren't going to keep doing their job and they're upset about that and so they're suicidal," he said. "But there is no mental illness that explains why somebody then feels entitled to also take that rage and turn it outward on 149 other people who had nothing to do with the person's problems." Germanwings crash compensation: What we know . Who was the captain of Germanwings Flight 9525? CNN's Margot Haddad reported from Marseille and Pamela Brown from Dusseldorf, while Laura Smith-Spark wrote from London. CNN's Frederik Pleitgen, Pamela Boykoff, Antonia Mortensen, Sandrine Amiel and Anna-Maja Rappard contributed to this report."""

    def _target(self, batch_size, max_cache_length, seq_length, cache_type, mode):
        self._run_buffer["batch_size"] = batch_size
        self._run_buffer["max_cache_length"] = max_cache_length
        self._run_buffer["seq_length"] = seq_length
        self._run_buffer["cache_type"] = cache_type
        self._run_buffer["mode"] = mode

        self.model.generation_config.max_length = max_cache_length
        input_ids = (
            self.tokenizer([self.input_text] * batch_size, return_tensors="pt")
            .to(self.device)
            .input_ids[:batch_size, :seq_length]
        )

        self.model._reset_cache()

        if cache_type == "static":
            self.model._setup_cache(StaticCache, batch_size, max_cache_len=max_cache_length)

        generated_ids = torch.zeros(batch_size, self.num_iter)

        model = self.model
        past_key_values = None

        set_seed(0)

        if cache_type == "dynamic":
            past_key_values = DynamicCache()

        if mode == "compiled":
            torch.compiler.reset()
            with torch.no_grad():
                model = torch.compile(self.model, mode="reduce-overhead", fullgraph=True)

        with torch.no_grad():
            outputs = model(input_ids, past_key_values=past_key_values, return_dict=False, use_cache=True)
            input_ids = sample(outputs[0], temperature=0.6, top_k=5)[0]
            generated_ids[:, 0] = input_ids[:, 0]

        _input_ids = input_ids

        def target():
            input_ids = _input_ids

            torch.cuda.synchronize()

            with torch.no_grad():
                for i in range(1, self.num_iter):
                    outputs = model(input_ids, past_key_values=past_key_values, return_dict=False, use_cache=True)
                    input_ids = sample(outputs[0], temperature=0.6, top_k=5)[0]
                    generated_ids[:, i] = input_ids[:, 0]

            torch.cuda.synchronize()

        return target


class CacheSpeedBenchMark(SpeedBenchMark, CacheBenchMark):
    pass


if __name__ == "__main__":
    repo_id = "meta-llama/Llama-2-7b-hf"
    prefill_num_iter = 3
    num_iter = 16

    benchmark = CacheSpeedBenchMark(repo_id=repo_id, prefill_num_iter=prefill_num_iter, num_iter=num_iter)

    # all_batch_size = [1, 2, 4]
    # all_max_cache_length = [4096, 2048, 1024, 512]
    # all_seq_length = [512, 1, 1024, 2048]

    all_batch_size = [1, 2]
    all_max_cache_length = [16]
    all_seq_length = [4]

    results = []
    for batch_size in all_batch_size:
        for max_cache_length in all_max_cache_length:
            for seq_length in all_seq_length:
                if seq_length >= max_cache_length:
                    break

                for cache_type in ["dynamic", "static"]:
                    for compile in ["eager", "compiled"]:
                        if cache_type == "dynamic" and compile:
                            continue

                        print(f"{batch_size}, {seq_length}, {max_cache_length}")

                        run_kwargs = {
                            "measure_kwargs": {"number": 2, "repeat": 3},
                            "target_kwargs": {
                                "batch_size": batch_size,
                                "max_cache_length": max_cache_length,
                                "seq_length": seq_length,
                                "cache_type": cache_type,
                                "mode": compile,
                            },
                            "inputs_kwargs": {},
                            "report_kwargs": {"output_path": "benchmark_report.json"},
                        }

                        result = benchmark.run(**run_kwargs)
                        results.append(result)

    print(json.dumps(results, indent=4))
