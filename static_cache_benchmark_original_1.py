FRANCE_ARTICLE = (  # @noqa
    """Marseille, France (CNN)The French prosecutor leading an investigation into the crash of Germanwings Flight 9525 insisted Wednesday that he was not aware of any video footage from on board the plane. Marseille prosecutor Brice Robin told CNN that "so far no videos were used in the crash investigation." He added, "A person who has such a video needs to immediately give it to the investigators." Robin\'s comments follow claims by two magazines, German daily Bild and French Paris Match, of a cell phone video showing the harrowing final seconds from on board Germanwings Flight 9525 as it crashed into the French Alps. All 150 on board were killed. Paris Match and Bild reported that the video was recovered from a phone at the wreckage site. The two publications described the supposed video, but did not post it on their websites. The publications said that they watched the video, which was found by a source close to the investigation. \"One can hear cries of 'My God' in several languages,\" Paris Match reported. "Metallic banging can also be heard more than three times, perhaps of the pilot trying to open the cockpit door with a heavy object.  Towards the end, after a heavy shake, stronger than the others, the screaming intensifies. Then nothing." "It is a very disturbing scene," said Julian Reichelt, editor-in-chief of Bild online. An official with France's accident investigation agency, the BEA, said the agency is not aware of any such video. Lt. Col. Jean-Marc Menichini, a French Gendarmerie spokesman in charge of communications on rescue efforts around the Germanwings crash site, told CNN that the reports were "completely wrong" and "unwarranted." Cell phones have been collected at the site, he said, but that they "hadn\'t been exploited yet." Menichini said he believed the cell phones would need to be sent to the Criminal Research Institute in Rosny sous-Bois, near Paris, in order to be analyzed by specialized technicians working hand-in-hand with investigators. But none of the cell phones found so far have been sent to the institute, Menichini said. Asked whether staff involved in the search could have leaked a memory card to the media, Menichini answered with a categorical "no." Reichelt told "Erin Burnett: Outfront" that he had watched the video and stood by the report, saying Bild and Paris Match are "very confident" that the clip is real. He noted that investigators only revealed they\'d recovered cell phones from the crash site after Bild and Paris Match published their reports. "That is something we did not know before. ... Overall we can say many things of the investigation weren't revealed by the investigation at the beginning," he said. What was mental state of Germanwings co-pilot? German airline Lufthansa confirmed Tuesday that co-pilot Andreas Lubitz had battled depression years before he took the controls of Germanwings Flight 9525, which he's accused of deliberately crashing last week in the French Alps. Lubitz told his Lufthansa flight training school in 2009 that he had a "previous episode of severe depression," the airline said Tuesday. Email correspondence between Lubitz and the school discovered in an internal investigation, Lufthansa said, included medical documents he submitted in connection with resuming his flight training. The announcement indicates that Lufthansa, the parent company of Germanwings, knew of Lubitz's battle with depression, allowed him to continue training and ultimately put him in the cockpit. Lufthansa, whose CEO Carsten Spohr previously said Lubitz was 100% fit to fly, described its statement Tuesday as a "swift and seamless clarification" and said it was sharing the information and documents -- including training and medical records -- with public prosecutors. Spohr traveled to the crash site Wednesday, where recovery teams have been working for the past week to recover human remains and plane debris scattered across a steep mountainside. He saw the crisis center set up in Seyne-les-Alpes, laid a wreath in the village of Le Vernet, closer to the crash site, where grieving families have left flowers at a simple stone memorial. Menichini told CNN late Tuesday that no visible human remains were left at the site but recovery teams would keep searching. French President Francois Hollande, speaking Tuesday, said that it should be possible to identify all the victims using DNA analysis by the end of the week, sooner than authorities had previously suggested. In the meantime, the recovery of the victims' personal belongings will start Wednesday, Menichini said. Among those personal belongings could be more cell phones belonging to the 144 passengers and six crew on board. Check out the latest from our correspondents . The details about Lubitz's correspondence with the flight school during his training were among several developments as investigators continued to delve into what caused the crash and Lubitz's possible motive for downing the jet. A Lufthansa spokesperson told CNN on Tuesday that Lubitz had a valid medical certificate, had passed all his examinations and "held all the licenses required." Earlier, a spokesman for the prosecutor\'s office in Dusseldorf, Christoph Kumpa, said medical records reveal Lubitz suffered from suicidal tendencies at some point before his aviation career and underwent psychotherapy before he got his pilot's license. Kumpa emphasized there's no evidence suggesting Lubitz was suicidal or acting aggressively before the crash. Investigators are looking into whether Lubitz feared his medical condition would cause him to lose his pilot's license, a European government official briefed on the investigation told CNN on Tuesday. While flying was "a big part of his life," the source said, it\'s only one theory being considered. Another source, a law enforcement official briefed on the investigation, also told CNN that authorities believe the primary motive for Lubitz to bring down the plane was that he feared he would not be allowed to fly because of his medical problems. Lubitz's girlfriend told investigators he had seen an eye doctor and a neuropsychologist, both of whom deemed him unfit to work recently and concluded he had psychological issues, the European government official said. But no matter what details emerge about his previous mental health struggles, there's more to the story, said Brian Russell, a forensic psychologist. "Psychology can explain why somebody would turn rage inward on themselves about the fact that maybe they weren't going to keep doing their job and they're upset about that and so they're suicidal," he said. "But there is no mental illness that explains why somebody then feels entitled to also take that rage and turn it outward on 149 other people who had nothing to do with the person's problems." Germanwings crash compensation: What we know . Who was the captain of Germanwings Flight 9525? CNN's Margot Haddad reported from Marseille and Pamela Brown from Dusseldorf, while Laura Smith-Spark wrote from London. CNN's Frederik Pleitgen, Pamela Boykoff, Antonia Mortensen, Sandrine Amiel and Anna-Maja Rappard contributed to this report."""
)


import torch


import torch._dynamo.config
import torch._inductor.config
from torch.utils import benchmark
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache, DynamicCache, set_seed

# from generation import  sample, model_forward

from typing import Optional


def multinomial_sample_one_no_sync(probs_sort): # Does multinomial sampling without a cuda synchronization
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
# torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future
# TORCH_LOGS="perf_hints,recompiles,graph_breaks" python bench_new.py &> logs.txt

import torch

torch.set_printoptions(linewidth=200)  # you can better see how the mask is shaped



device = "cuda"
attn_implementation = "sdpa"
all_dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", padding_side="left", pad_token = "<s>")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=all_dtype, attn_implementation=attn_implementation).to(device,all_dtype)
model.eval()


# gpt_fast_model = _load_model(device, all_dtype)

def print_results(benchmark_results):
    print("\n")
    compare = benchmark.Compare(benchmark_results)
    compare.trim_significant_figures()
    compare.colorize(rowwise = True)
    compare.print()

def record_function(function, gpt_fast = False):
    with torch.no_grad():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        outputs = function()
        if gpt_fast:
            inputs_ids = sample(outputs,temperature=0.6, top_k=5)[0]
        else: # batch in transformers
            inputs_ids = sample(outputs[0],temperature=0.6, top_k=5)[0]
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end), inputs_ids

BIG_INPUT = tokenizer([FRANCE_ARTICLE]*4, return_tensors="pt").to(device).input_ids

benchmark_results = []
NUM_ITER = 100
for BS in reversed([1, 2, 4]):
    for max_cache_length in reversed([4096, 2048, 1024, 512]):
        for seq_length in [ 512, 1, 1024, 2048]:
            if seq_length>=max_cache_length:
                break
            label = "model(inputs)"
            description = f"batch, cache_len, seq_length: {BS, seq_length, max_cache_length}"
            model.generation_config.max_length = max_cache_length

            input_ids = BIG_INPUT[:BS, :seq_length]

            model.eval()
            model._reset_cache()

            TASK = "eager dynamic cache"
            task_spec = benchmark.TaskSpec(stmt="", setup="", description=TASK,label=label,sub_label=description)
            generated_ids = torch.zeros(BS, NUM_ITER, dtype = torch.long)
            past_key_values = DynamicCache()
            res = []
            set_seed(0)
            import datetime
            start = datetime.datetime.now()
            for i in range(NUM_ITER):
                if i == 0:
                    time, inputs_id = record_function(lambda: model(input_ids, past_key_values=past_key_values, return_dict=False, use_cache = True) )
                else:
                    time, inputs_id = record_function(lambda: model(inputs_id, past_key_values=past_key_values, return_dict=False, use_cache = True) )

                res.append(time)
                generated_ids[:, i] = inputs_id[:,0]

            torch.cuda.synchronize()
            total_time = datetime.datetime.now() - start

            benchmark_results.append(benchmark.Measurement(NUM_ITER-3, res[3:], task_spec, metadata=None))
            task_spec = benchmark.TaskSpec(stmt="", setup="", description=TASK,label="token/s",sub_label=description)
            # benchmark_results.append(benchmark.Measurement(1, [ 100 / torch.Tensor(res[3:]).mean()  ], task_spec, metadata=None))
            print(1, [ 100 / torch.Tensor(res[3:]).mean()  ])
            print(f"tokens: {tokenizer.batch_decode(generated_ids.tolist())}")
            print_results(benchmark_results)


            TASK  = "eager static cache"
            task_spec = benchmark.TaskSpec(stmt="", setup="", description=TASK,label=label,sub_label=description)
            generated_ids = torch.zeros(BS, NUM_ITER)

            res = []
            set_seed(0)
            model._setup_cache(StaticCache, BS, max_cache_len=max_cache_length)
            for i in range(NUM_ITER):
                if i == 0:
                    time, inputs_id = record_function(lambda: model(input_ids, return_dict=False, use_cache = True))
                else:
                    time, inputs_id = record_function(lambda: model(inputs_id, return_dict=False, use_cache = True))
                res.append(time)
                generated_ids[:, i] = inputs_id[:,0]

            benchmark_results.append(benchmark.Measurement(NUM_ITER-3, res[3:], task_spec, metadata=None))
            task_spec = benchmark.TaskSpec(stmt="", setup="", description=TASK,label="token/s",sub_label=description)
            # benchmark_results.append(benchmark.Measurement(1, [ 100 / torch.Tensor(res[3:]).mean()  ], task_spec, metadata=None))
            print(f"tokens: {tokenizer.batch_decode(generated_ids.long().tolist())}")
            print(1, [ 100 / torch.Tensor(res[3:]).mean()  ])

            del past_key_values
            print_results(benchmark_results)


            TASK = "compiled static cache"
            task_spec = benchmark.TaskSpec(stmt="", setup="", description=TASK,label=label,sub_label=description)
            model._setup_cache(StaticCache, BS, max_cache_len=max_cache_length)
            generated_ids = torch.zeros(BS, NUM_ITER)
            res = []
            torch.compiler.reset()

            with torch.no_grad():
                compiled_model = torch.compile(model, mode="reduce-overhead",fullgraph=True)
                set_seed(0)
                for i in range(NUM_ITER):
                    if i == 0:
                        time, inputs_id = record_function(lambda: compiled_model(input_ids, return_dict=False, use_cache = True))
                    else:
                        time, inputs_id = record_function(lambda: compiled_model(inputs_id, return_dict=False, use_cache = True) )

                    res.append(time)
                    generated_ids[:, i] = inputs_id[:,0]

            benchmark_results.append(benchmark.Measurement(NUM_ITER-3, res[3:], task_spec, metadata=None))
            task_spec = benchmark.TaskSpec(stmt="", setup="", description=TASK, label="token/s", sub_label=description)
            print(f"tokens: {tokenizer.batch_decode(generated_ids.long().tolist())}")
            print(1, [ 100 / torch.Tensor(res[3:]).mean()  ])
            print_results(benchmark_results)