from transformers import LlamaForCausalLM, AutoTokenizer, DynamicCache
from transformers.cache_utils import HQQQuantizedCacheStatic, QuantizedCacheConfig
import torch
import time
import os
import pandas as pd

from hqq.models.hf.base import AutoHQQHFModel
from hqq.core.quantize import *
from hqq.utils.patching import prepare_for_inference



os.environ["TOKENIZERS_PARALLELISM"] = "0"
all_dtypes = torch.float16

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True
#torch._dynamo.config.capture_dynamic_output_shape_ops = True
#torch._dynamo.config.capture_scalar_outputs = True
torch.set_float32_matmul_precision('high')

torch._logging.set_logs(graph_breaks=True, recompiles=True)
#torch._dynamo.config.cache_size_limit = 256


model_id = "meta-llama/Llama-2-7b-chat-hf"
model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype=all_dtypes, device_map="auto") 
tokenizer = AutoTokenizer.from_pretrained(model_id)

def quantize_custom(model):
	quant_config = BaseQuantizeConfig(nbits=4, group_size=64, quant_scale=False, quant_zero=False, axis=1) 
	HQQLinear.set_backend(HQQBackend.PYTORCH)
    #HQQLinear.set_backend(HQQBackend.PYTORCH_COMPILE)

	AutoHQQHFModel.quantize_model(model, quant_config=quant_config, compute_dtype=torch.bfloat16, device="cuda")
	prepare_for_inference(model)
	prepare_for_inference(model, backend="torchao_int4")

# quantize_custom(model)


prompt = """<s>Marseille, France (CNN)The French prosecutor leading an investigation into the crash of Germanwings Flight 9525 insisted Wednesday that he was not aware of any video footage from on board the plane. Marseille prosecutor Brice Robin told CNN that "so far no videos were used in the crash investigation." He added, "A person who has such a video needs to immediately give it to the investigators." Robin\'s comments follow claims by two magazines, German daily Bild and French Paris Match, of a cell phone video showing the harrowing final seconds from on board Germanwings Flight 9525 as it crashed into the French Alps. All 150 on board were killed. Paris Match and Bild reported that the video was recovered from a phone at the wreckage site. The two publications described the supposed video, but did not post it on their websites. The publications said that they watched the video, which was found by a source close to the investigation. \"One can hear cries of 'My God' in several languages,\" Paris Match reported. "Metallic banging can also be heard more than three times, perhaps of the pilot trying to open the cockpit door with a heavy object.  Towards the end, after a heavy shake, stronger than the others, the screaming intensifies. Then nothing." "It is a very disturbing scene," said Julian Reichelt, editor-in-chief of Bild online. An official with France's accident investigation agency, the BEA, said the agency is not aware of any such video. Lt. Col. Jean-Marc Menichini, a French Gendarmerie spokesman in charge of communications on rescue efforts around the Germanwings crash site, told CNN that the reports were "completely wrong" and "unwarranted." Cell phones have been collected at the site, he said, but that they "hadn\'t been exploited yet." Menichini said he believed the cell phones would need to be sent to the Criminal Research Institute in Rosny sous-Bois, near Paris, in order to be analyzed by specialized technicians working hand-in-hand with investigators. But none of the cell phones found so far have been sent to the institute, Menichini said. Asked whether staff involved in the search could have leaked a memory card to the media, Menichini answered with a categorical "no." Reichelt told "Erin Burnett: Outfront" that he had watched the video and stood by the report, saying Bild and Paris Match are "very confident" that the clip is real. He noted that investigators only revealed they\'d recovered cell phones from the crash site after Bild and Paris Match published their reports. "That is something we did not know before. ... Overall we can say many things of the investigation weren't revealed by the investigation at the beginning," he said. What was mental state of Germanwings co-pilot? German airline Lufthansa confirmed Tuesday that co-pilot Andreas Lubitz had battled depression years before he took the controls of Germanwings Flight 9525, which he's accused of deliberately crashing last week in the French Alps. Lubitz told his Lufthansa flight training school in 2009 that he had a "previous episode of severe depression," the airline said Tuesday. Email correspondence between Lubitz and the school discovered in an internal investigation, Lufthansa said, included medical documents he submitted in connection with resuming his flight training. The announcement indicates that Lufthansa, the parent company of Germanwings, knew of Lubitz's battle with depression, allowed him to continue training and ultimately put him in the cockpit. Lufthansa, whose CEO Carsten Spohr previously said Lubitz was 100% fit to fly, described its statement Tuesday as a "swift and seamless clarification" and said it was sharing the information and documents -- including training and medical records -- with public prosecutors. Spohr traveled to the crash site Wednesday, where recovery teams have been working for the past week to recover human remains and plane debris scattered across a steep mountainside. He saw the crisis center set up in Seyne-les-Alpes, laid a wreath in the village of Le Vernet, closer to the crash site, where grieving families have left flowers at a simple stone memorial. Menichini told CNN late Tuesday that no visible human remains were left at the site but recovery teams would keep searching. French President Francois Hollande, speaking Tuesday, said that it should be possible to identify all the victims using DNA analysis by the end of the week, sooner than authorities had previously suggested. In the meantime, the recovery of the victims' personal belongings will start Wednesday, Menichini said. Among those personal belongings could be more cell phones belonging to the 144 passengers and six crew on board. Check out the latest from our correspondents . The details about Lubitz's correspondence with the flight school during his training were among several developments as investigators continued to delve into what caused the crash and Lubitz's possible motive for downing the jet. A Lufthansa spokesperson told CNN on Tuesday that Lubitz had a valid medical certificate, had passed all his examinations and "held all the licenses required." Earlier, a spokesman for the prosecutor\'s office in Dusseldorf, Christoph Kumpa, said medical records reveal Lubitz suffered from suicidal tendencies at some point before his aviation career and underwent psychotherapy before he got his pilot's license. Kumpa emphasized there's no evidence suggesting Lubitz was suicidal or acting aggressively before the crash. Investigators are looking into whether Lubitz feared his medical condition would cause him to lose his pilot's license, a European government official briefed on the investigation told CNN on Tuesday. While flying was "a big part of his life," the source said, it\'s only one theory being considered. Another source, a law enforcement official briefed on the investigation, also told CNN that authorities believe the primary motive for Lubitz to bring down the plane was that he feared he would not be allowed to fly because of his medical problems. Lubitz's girlfriend told investigators he had seen an eye doctor and a neuropsychologist, both of whom deemed him unfit to work recently and concluded he had psychological issues, the European government official said. But no matter what details emerge about his previous mental health struggles, there's more to the story, said Brian Russell, a forensic psychologist. "Psychology can explain why somebody would turn rage inward on themselves about the fact that maybe they weren't going to keep doing their job and they're upset about that and so they're suicidal," he said. "But there is no mental illness that explains why somebody then feels entitled to also take that rage and turn it outward on 149 other people who had nothing to do with the person's problems." Germanwings crash compensation: What we know . Who was the captain of Germanwings Flight 9525? CNN's Margot Haddad reported from Marseille and Pamela Brown from Dusseldorf, while Laura Smith-Spark wrote from London. CNN's Frederik Pleitgen, Pamela Boykoff, Antonia Mortensen, Sandrine Amiel and Anna-Maja Rappard contributed to this report.""" * 40

cache_config = QuantizedCacheConfig(
    nbits=4,
    compute_dtype=all_dtypes,
    device=model.device
)

memory_init = torch.cuda.memory_allocated()
torch.cuda.reset_max_memory_allocated()

results_latency, results_memory = {"dynamic": {}, "static-int4": {}}, {"dynamic": {}, "static-int4": {}}

for bs in [1, 4, 16]:
    for max_cache_length in [512, 2048]:
        for seq_length in [1, 1024]:
            description = f"batch {bs}, cache_len {max_cache_length}, seq_length {seq_length}"
            model.generation_config.max_length = max_cache_length
            inputs = tokenizer([prompt]*bs, return_tensors="pt").to(model.device).input_ids[:bs, :seq_length]

            # STATIC CACHE IN LOW-BIT
            past_key_values = HQQQuantizedCacheStatic(
                config=model.config,
                cache_config=cache_config,
                max_batch_size=bs,
                max_cache_len=seq_length+max_cache_length,
            )

            with torch.no_grad():                
                cache_position = torch.arange(seq_length , device=model.device)
                generated_ids = torch.zeros(bs, seq_length+max_cache_length, dtype = torch.int64, device=model.device)
                generated_ids[:, cache_position] = inputs.to(model.device)

                # pre-fill
                outputs = model(inputs, past_key_values=past_key_values, cache_position=cache_position, return_dict=True, use_cache=True)
                next_token = torch.argmax(outputs[0][:, -1:, :], dim=-1)
                generated_ids[:, seq_length] = next_token.squeeze()
                past_key_values = outputs.past_key_values

                torch.compiler.reset()
                compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
                cache_position = torch.tensor([seq_length] , device=model.device)

                # warm-up
                for i in range(1, 5):
                    out = compiled_model(next_token.clone(), past_key_values=past_key_values, cache_position=cache_position, return_dict=True, use_cache=True)
                    new_token = torch.argmax(out[0][:, -1:, :], dim=-1)
                    past_key_values = out.past_key_values
                    generated_ids.index_copy_(1, cache_position, next_token)
                    cache_position += 1
 
                # eval bench static 
                start = time.perf_counter()
                for i in range(5, max_cache_length):
                    out = compiled_model(next_token.clone(), past_key_values=past_key_values, cache_position=cache_position, return_dict=True, use_cache=True)
                    new_token = torch.argmax(out[0][:, -1:, :], dim=-1)
                    past_key_values = out.past_key_values
                    generated_ids.index_copy_(1, cache_position, next_token)
                    cache_position += 1
                    
                torch.cuda.synchronize()
                total_time = time.perf_counter() - start
                forward_average_time =  total_time / (max_cache_length - 5)
                print(f"Compiled static cache\t - Took : {total_time:.4f} sec total, {forward_average_time*1000:.3f} ms per forward, {((max_cache_length - 5)/total_time):.2f} tokens/s")

                peak = torch.cuda.max_memory_allocated()
                del outputs, past_key_values, cache_position
                torch.cuda.empty_cache()
                torch.cuda.reset_max_memory_allocated()
                print(f"Memory peaked: {(peak - memory_init) // 1024 ** 2} MiB")

                results_latency["static-int4"][description] = (total_time, forward_average_time, (max_cache_length - 5)/total_time)
                results_memory["static-int4"][description] = ((peak - memory_init) // 1024 ** 2)

            with torch.no_grad():
                generated_ids = torch.zeros(bs, seq_length + max_cache_length, dtype=torch.long, device=model.device)
                cache_position = torch.arange(seq_length, device=model.device)
                generated_ids[:, cache_position] = inputs.to(model.device)
                past_key_values = DynamicCache()
                
                # pre-fill
                outputs = model(inputs, past_key_values=past_key_values, cache_position=cache_position, return_dict=True, use_cache=True)
                new_token = torch.argmax(outputs[0][:, -1:, :], dim=-1)
                generated_ids[:, seq_length] = new_token.squeeze(-1)
                past_key_values = outputs.past_key_values
                cache_position = torch.tensor([seq_length] , device=model.device)
                
                # warm-up
                for i in range(1, 5):
                    outputs = model(new_token, past_key_values=past_key_values, cache_position=cache_position, return_dict=True, use_cache=True)
                    new_token = torch.argmax(outputs[0][:, -1:, :], dim=-1)
                    cache_position += 1
                    generated_ids[:, i] = new_token.squeeze(-1)
                    past_key_values = outputs.past_key_values

                # eval bench
                start = time.perf_counter()
                for i in range(5, max_cache_length):
                    outputs = model(new_token, past_key_values=past_key_values, cache_position=cache_position, return_dict=True, use_cache=True)
                    new_token = torch.argmax(outputs[0][:, -1:, :], dim=-1)
                    generated_ids[:, i] = new_token.squeeze(-1)
                    cache_position += 1
                    past_key_values = outputs.past_key_values

                torch.cuda.synchronize()
                total_time = time.perf_counter() - start
                forward_average_time =  total_time / (max_cache_length - 5)
                print(f"Dynamic cache\t - Took : {total_time:.4f} sec total, {forward_average_time*1000:.3f} ms per forward, {((max_cache_length - 5)/total_time):.2f} tokens/s")

                peak = torch.cuda.max_memory_allocated()
                print(f"Memory peaked: {(peak - memory_init) // 1024 ** 2} MiB")
                del outputs, past_key_values, cache_position
                torch.cuda.empty_cache()
                torch.cuda.reset_max_memory_allocated()

                results_latency["dynamic"][description] = (total_time, forward_average_time, (max_cache_length - 5)/total_time)
                results_memory["dynamic"][description] = ((peak - memory_init) // 1024 ** 2)


results_memory = pd.DataFrame(results_memory)
results_latency = pd.DataFrame(results_latency)
results_latency.to_csv("results_latency.csv")
results_memory.to_csv("results_memory.csv")

