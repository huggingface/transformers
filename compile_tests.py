import os
import time
import datetime
import logging
from typing import Optional

import torch
import torch._dynamo.config
import torch._inductor.config
from transformers.generation import (
    EncoderNoRepeatNGramLogitsProcessor,
    EncoderRepetitionPenaltyLogitsProcessor,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    ForceTokensLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SequenceBiasLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
    UnbatchedClassifierFreeGuidanceLogitsProcessor,
)

from transformers.generation import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteriaList,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
import time 
from torch.profiler import profile, record_function, ProfilerActivity


os.environ["TOKENIZERS_PARALLELISM"] = "0"
torch.set_float32_matmul_precision('high')

device = "cuda:0"
attn_implementation = "sdpa"
all_dtype = torch.bfloat16

vocab_size = 512
seq_len = 200
batch_size = 10
input_ids = torch.randint(0, vocab_size, size=(batch_size, seq_len), dtype=torch.int64, device=device)
scores = torch.randn(batch_size, vocab_size, device=device)
current_tokens = torch.randint(0, vocab_size, size=(8, seq_len), dtype=torch.int64, device=device)
beam_group_idx = 0


def multinomial_sample_one_no_sync(probs_sort): # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int64)

def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits, temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next

def prefix_allowed_tokens_fn(batch_id, input_ids):
    if input_ids[-1] in range(30):
        return [30]
    return list(range(vocab_size))


processors_beam_search = LogitsProcessorList([
    HammingDiversityLogitsProcessor(diversity_penalty=1.1, num_beams=4, num_beam_groups=2),
])

processors_seq2seq = LogitsProcessorList([
    #EncoderNoRepeatNGramLogitsProcessor(encoder_ngram_size=2, encoder_input_ids=input_ids),
    EncoderRepetitionPenaltyLogitsProcessor(penalty=0.2, encoder_input_ids=input_ids),
    ForcedBOSTokenLogitsProcessor(bos_token_id=1),
])

warpers = LogitsProcessorList([
    TemperatureLogitsWarper(temperature=0.5),
    TopKLogitsWarper(top_k=50),
    TopPLogitsWarper(top_p=0.9),
    TypicalLogitsWarper(),
    EpsilonLogitsWarper(epsilon=0.1),
    EtaLogitsWarper(epsilon=0.1),
])

processors = LogitsProcessorList([
    #ExponentialDecayLengthPenalty(exponential_decay_length_penalty=(2, 1.2), eos_token_id=0, input_ids_seq_length=1),
    ForcedEOSTokenLogitsProcessor(max_length=102, eos_token_id=0), #cur_len = max_len-1
    ForceTokensLogitsProcessor(force_token_map=[[100, 133], [102, 133], [4, 12], [3, 14]]),
    InfNanRemoveLogitsProcessor(),
    LogitNormalization(),
    MinLengthLogitsProcessor(min_length=210, eos_token_id=list(range(20))),
    MinNewTokensLengthLogitsProcessor(prompt_length_to_skip=2, min_new_tokens=200, eos_token_id=0),
    NoBadWordsLogitsProcessor(bad_words_ids=[[213], [12], [93]], eos_token_id=0),
    #NoRepeatNGramLogitsProcessor(ngram_size=2), # index 10
    #PrefixConstrainedLogitsProcessor(prefix_allowed_tokens_fn=prefix_allowed_tokens_fn, num_beams=1), # index 11
    RepetitionPenaltyLogitsProcessor(penalty=1.3),
    #SequenceBiasLogitsProcessor(sequence_bias={(213,): -10.0, (10,): -10.0, (25,): 10.0}),
    SuppressTokensAtBeginLogitsProcessor(begin_suppress_tokens=[11, 12, 13, 14], begin_index=100),
    SuppressTokensLogitsProcessor(suppress_tokens=[11, 12, 13, 14]),
])

criterias = StoppingCriteriaList = ([
    MaxLengthCriteria(max_length=300, max_position_embeddings=512),
    # MaxTimeCriteria(max_time=2), has builtin time.time() every __call__
])


#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", padding_side="left", pad_token = "<s>")
#model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=all_dtype, attn_implementation=attn_implementation).to(device,all_dtype)
#model = model.eval()
#cfg_guidance = LogitsProcessorList([
#    UnbatchedClassifierFreeGuidanceLogitsProcessor(guidance_scale=1.1, model=model) # -> Dynamic control flow in model.forward (update_causal_mask)
#])


benchmark_results = []
PREFILL_NUM_ITER = 3
NUM_ITER = 100

def beam_logits(input_ids, scores, cur_len):
    for processor in processors_beam_search:
        scores = processor(input_ids, scores, current_tokens, beam_group_idx, cur_len=cur_len)
    next_token = sample(scores)
    return next_token

def run_logits_processor(input_ids, scores, cur_len):
    for logits_processor_fn in processors:
        scores = logits_processor_fn(input_ids, scores, cur_len=cur_len)
    scores = scores / max(1, 1e-5)
    probs = torch.nn.functional.softmax(scores, dim=-1)
    token = torch.argmax(probs, dim=-1, keepdim=True).detach().to(torch.int64)
    sampled_token = sample(scores, temperature=0.6, top_k=5)
    return sampled_token

def run_stopping_criteria(input_ids, scores):
    for stopping_criteria in criterias:
        do_stop = stopping_criteria(input_ids, scores)
        if do_stop:
            return []
    sampled_token = sample(scores, temperature=0.6, top_k=5)
    return sampled_token

def check_processors_with_model():
    input_ids = tokenizer(["Today, a dragon flew over Paris, France,"]*2, return_tensors="pt").to(device).input_ids
    seq_length = input_ids.shape[-1]
    model._setup_cache(StaticCache, input_ids.shape[0], max_cache_len=128)

    def decode_one_tokens(model, cur_token, input_pos, generated_ids, cur_len):
        logits = model(cur_token, position_ids=input_pos, return_dict=False, use_cache=True)[0]
        logits = logits[:, -1, :]
        for processor in processors:
            logits = processor(generated_ids, logits, cur_len=cur_len)
        #new_token = sample(logits, temperature=0.6, top_k=5)        
        return torch.tensor([[120], [122]], dtype=torch.int64, device=device)

    # PREFILL
    cache_position = torch.arange(seq_length, device=device)
    generated_ids = torch.zeros(input_ids.shape[0], seq_length+7, dtype = torch.int64, device=device)
    generated_ids[:, cache_position] = input_ids.to(device).to(torch.int64)
    
    # ipnut_dependent init
    processors_seq2seq[0] = EncoderRepetitionPenaltyLogitsProcessor(penalty=0.2, encoder_input_ids=generated_ids)

    logits = model(input_ids, return_dict=False, use_cache=True)[0]
    logits = logits[:, -1, :]
    for processor in processors:
        logits = processor(generated_ids, logits, cur_len=seq_length)
    next_token = sample(logits,temperature=0.6, top_k=5)
    generated_ids[:, seq_length] = next_token[:, 0]
   
    decode_one_tokens = torch.compile(decode_one_tokens, mode="reduce-overhead", fullgraph=True)
    cache_position = torch.tensor([seq_length], device=device)
    cur_len = torch.tensor([seq_length+1], dtype=torch.int, device=device)
    
    ## Check it compiles without exceptions
    for i in range(5):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            next_token = decode_one_tokens(model, next_token.clone(), None, generated_ids.clone(), cur_len)
            generated_ids.index_copy_(1, cache_position, next_token)
        cache_position+=1
        cur_len += 1
    print(generated_ids)


# EncoderNoRepeatNGramLogitsProcessor, NoRepeatNGramLogitsProcessor -> calls for `_get_generated_ngrams` which leads to a graph_breaks. 
# GuardOnDataDependentSymNode: trying to get a value out of symbolic int/float whose value is data-dependent

# PrefixConstrainedLogitsProcessor -> will not work as the user_provided fn will mostly probably lead to dynamic control flow error

# SequenceBiasLogitsProcessor -> prepare_bias_variables being called causes recompilation - check_obj_id(G['processors'][9].prepared_bias_variables, 9478112)
# only if init with NoBadWordsProcessor, both refer to the same `_prepare_bias``

def run_dummy_inputs():
    cur_len = 100
    generated_ids = torch.zeros(input_ids.shape[0], 120, dtype = torch.int64, device=device)
    for i in range(5):
        next_token = run_logits_processor(input_ids, scores, cur_len)
    logits_processor_compiled = torch.compile(run_logits_processor, mode="reduce-overhead", fullgraph=True)
    for i in range(5):
        next_token = logits_processor_compiled(input_ids, scores, cur_len)
        generated_ids[:, cur_len] = next_token[0]
        cur_len += 1


torch._dynamo.config.cache_size_limit = 5
torch._logging.set_logs(dynamo=logging.INFO, graph_breaks=True, recompiles=True)

#check_processors_with_model()
run_dummy_inputs()


def benchmark():
    FRANCE_ARTICLE = (  # @noqa
        """<s>Marseille, France (CNN)The French prosecutor leading an investigation into the crash of Germanwings Flight 9525 insisted Wednesday that he was not aware of any video footage from on board the plane. Marseille prosecutor Brice Robin told CNN that "so far no videos were used in the crash investigation." He added, "A person who has such a video needs to immediately give it to the investigators." Robin\'s comments follow claims by two magazines, German daily Bild and French Paris Match, of a cell phone video showing the harrowing final seconds from on board Germanwings Flight 9525 as it crashed into the French Alps. All 150 on board were killed. Paris Match and Bild reported that the video was recovered from a phone at the wreckage site. The two publications described the supposed video, but did not post it on their websites. The publications said that they watched the video, which was found by a source close to the investigation. \"One can hear cries of 'My God' in several languages,\" Paris Match reported. "Metallic banging can also be heard more than three times, perhaps of the pilot trying to open the cockpit door with a heavy object.  Towards the end, after a heavy shake, stronger than the others, the screaming intensifies. Then nothing." "It is a very disturbing scene," said Julian Reichelt, editor-in-chief of Bild online. An official with France's accident investigation agency, the BEA, said the agency is not aware of any such video. Lt. Col. Jean-Marc Menichini, a French Gendarmerie spokesman in charge of communications on rescue efforts around the Germanwings crash site, told CNN that the reports were "completely wrong" and "unwarranted." Cell phones have been collected at the site, he said, but that they "hadn\'t been exploited yet." Menichini said he believed the cell phones would need to be sent to the Criminal Research Institute in Rosny sous-Bois, near Paris, in order to be analyzed by specialized technicians working hand-in-hand with investigators. But none of the cell phones found so far have been sent to the institute, Menichini said. Asked whether staff involved in the search could have leaked a memory card to the media, Menichini answered with a categorical "no." Reichelt told "Erin Burnett: Outfront" that he had watched the video and stood by the report, saying Bild and Paris Match are "very confident" that the clip is real. He noted that investigators only revealed they\'d recovered cell phones from the crash site after Bild and Paris Match published their reports. "That is something we did not know before. ... Overall we can say many things of the investigation weren't revealed by the investigation at the beginning," he said. What was mental state of Germanwings co-pilot? German airline Lufthansa confirmed Tuesday that co-pilot Andreas Lubitz had battled depression years before he took the controls of Germanwings Flight 9525, which he's accused of deliberately crashing last week in the French Alps. Lubitz told his Lufthansa flight training school in 2009 that he had a "previous episode of severe depression," the airline said Tuesday. Email correspondence between Lubitz and the school discovered in an internal investigation, Lufthansa said, included medical documents he submitted in connection with resuming his flight training. The announcement indicates that Lufthansa, the parent company of Germanwings, knew of Lubitz's battle with depression, allowed him to continue training and ultimately put him in the cockpit. Lufthansa, whose CEO Carsten Spohr previously said Lubitz was 100% fit to fly, described its statement Tuesday as a "swift and seamless clarification" and said it was sharing the information and documents -- including training and medical records -- with public prosecutors. Spohr traveled to the crash site Wednesday, where recovery teams have been working for the past week to recover human remains and plane debris scattered across a steep mountainside. He saw the crisis center set up in Seyne-les-Alpes, laid a wreath in the village of Le Vernet, closer to the crash site, where grieving families have left flowers at a simple stone memorial. Menichini told CNN late Tuesday that no visible human remains were left at the site but recovery teams would keep searching. French President Francois Hollande, speaking Tuesday, said that it should be possible to identify all the victims using DNA analysis by the end of the week, sooner than authorities had previously suggested. In the meantime, the recovery of the victims' personal belongings will start Wednesday, Menichini said. Among those personal belongings could be more cell phones belonging to the 144 passengers and six crew on board. Check out the latest from our correspondents . The details about Lubitz's correspondence with the flight school during his training were among several developments as investigators continued to delve into what caused the crash and Lubitz's possible motive for downing the jet. A Lufthansa spokesperson told CNN on Tuesday that Lubitz had a valid medical certificate, had passed all his examinations and "held all the licenses required." Earlier, a spokesman for the prosecutor\'s office in Dusseldorf, Christoph Kumpa, said medical records reveal Lubitz suffered from suicidal tendencies at some point before his aviation career and underwent psychotherapy before he got his pilot's license. Kumpa emphasized there's no evidence suggesting Lubitz was suicidal or acting aggressively before the crash. Investigators are looking into whether Lubitz feared his medical condition would cause him to lose his pilot's license, a European government official briefed on the investigation told CNN on Tuesday. While flying was "a big part of his life," the source said, it\'s only one theory being considered. Another source, a law enforcement official briefed on the investigation, also told CNN that authorities believe the primary motive for Lubitz to bring down the plane was that he feared he would not be allowed to fly because of his medical problems. Lubitz's girlfriend told investigators he had seen an eye doctor and a neuropsychologist, both of whom deemed him unfit to work recently and concluded he had psychological issues, the European government official said. But no matter what details emerge about his previous mental health struggles, there's more to the story, said Brian Russell, a forensic psychologist. "Psychology can explain why somebody would turn rage inward on themselves about the fact that maybe they weren't going to keep doing their job and they're upset about that and so they're suicidal," he said. "But there is no mental illness that explains why somebody then feels entitled to also take that rage and turn it outward on 149 other people who had nothing to do with the person's problems." Germanwings crash compensation: What we know . Who was the captain of Germanwings Flight 9525? CNN's Margot Haddad reported from Marseille and Pamela Brown from Dusseldorf, while Laura Smith-Spark wrote from London. CNN's Frederik Pleitgen, Pamela Boykoff, Antonia Mortensen, Sandrine Amiel and Anna-Maja Rappard contributed to this report."""
    )

    BIG_INPUT = tokenizer([FRANCE_ARTICLE]*4, return_tensors="pt").to(device).input_ids


    for BS in [1, 2, 4]:
        for max_cache_length in [4096, 2048, 1024, 512]:
            for seq_length in [ 512, 1, 1024, 2048]:
                print(f"{BS}, {seq_length}, {max_cache_length}")
                input_ids = BIG_INPUT[:BS, :seq_length]
                
                def decode_one_tokens(model, cur_token, input_pos, generated_ids, cur_len):
                    logits = model(cur_token, position_ids=input_pos, return_dict=False, use_cache=True)[0]
                    logits = logits[:, -1, :]
                    for processor in processors:
                        logits = processor(generated_ids, logits, cur_len=cur_len)
                    #new_token = sample(logits, temperature=0.6, top_k=5)                    
                    return new_token

                with torch.no_grad():
                    model._setup_cache(StaticCache, BS, max_cache_len=max_cache_length)
                    
                    ### PREFILL
                    cache_position = torch.arange(seq_length , device=device)
                    generated_ids = torch.zeros(BS, seq_length+NUM_ITER+PREFILL_NUM_ITER+1, dtype = torch.int, device=device)
                    generated_ids[:, cache_position] = input_ids.to(device).to(torch.int)
                    
                    logits = model(input_ids, return_dict=False, use_cache = True)[0]
                    logits = logits[:, -1, :]
                    #for processor in processors:
                    #    logits = processor(generated_ids, logits, cur_len=seq_length)
                    next_token = sample(logits,temperature=0.6, top_k=5)
                    generated_ids[:, seq_length] = next_token
                    
                    decode_one_tokens = torch.compile(decode_one_tokens, mode="reduce-overhead",fullgraph=True)

                    ### DRY RUN
                    # input_pos = torch.tensor([[seq_length]], device=device)
                    cache_position = torch.tensor([seq_length] , device=device)
                    cur_len = torch.tensor([seq_length+1], dtype=torch.int, device=device)
                    for i in range(1, PREFILL_NUM_ITER):
                        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                            next_token = decode_one_tokens(model, next_token.clone(), None, generated_ids, cur_len)
                            generated_ids.index_copy_(1, cache_position, next_token)
                        cache_position+=1
                        cur_len += 1

                    ### RUN
                    start = time.perf_counter()
                    for i in range(PREFILL_NUM_ITER, NUM_ITER):
                        torch.cuda.synchronize()
                        _start = time.perf_counter()
                        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                            next_token = decode_one_tokens(model, next_token.clone(), None, generated_ids, cur_len)
                            generated_ids.index_copy_(1, cache_position, next_token)
                        torch.cuda.synchronize()
                        _total_time = time.perf_counter() - _start
                        print(_total_time, end="\t")
                        # input_pos+=1
                        cache_position+=1
                        cur_len += 1
                        
                    torch.cuda.synchronize()
                    total_time = time.perf_counter() - start
                    forward_average_time =  total_time / NUM_ITER
                    print(f"compiled static cache\t({BS},  {seq_length}) - Took : {total_time:.4f} sec total, {forward_average_time*1000:.3f} ms per forward, {(NUM_ITER/total_time):.2f} tokens/s")
                

