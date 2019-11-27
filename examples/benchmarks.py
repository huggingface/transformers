# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Benchmarking the library on inference and training """

# If checking the tensors placement
# tf.debugging.set_log_device_placement(True)

from typing import List
import timeit
from transformers import is_tf_available, is_torch_available
from time import time
import argparse
import csv

if is_tf_available():
    import tensorflow as tf
    from transformers import TFAutoModel

if is_torch_available():
    import torch
    from transformers import AutoModel

from transformers import AutoConfig, AutoTokenizer

input_text = """Bent over their instruments, three hundred Fertilizers were plunged, as 
the Director of Hatcheries and Conditioning entered the room, in the 



scarcely breathing silence, the absent-minded, soliloquizing hum or 
whistle, of absorbed concentration. A troop of newly arrived students, 
very young, pink and callow, followed nervously, rather abjectly, at the 
Director's heels. Each of them carried a notebook, in which, whenever 
the great man spoke, he desperately scribbled. Straight from the 
horse's mouth. It was a rare privilege. The D. H. C. for Central London 
always made a point of personally conducting his new students round 
the various departments. 

"Just to give you a general idea," he would explain to them. For of 
course some sort of general idea they must have, if they were to do 
their work intelligently-though as little of one, if they were to be good 
and happy members of society, as possible. For particulars, as every 
one knows, make for virtue and happiness; generalities are intellectu- 
ally necessary evils. Not philosophers but fret-sawyers and stamp col- 
lectors compose the backbone of society. 

"To-morrow," he would add, smiling at them with a slightly menacing 
geniality, "you'll be settling down to serious work. You won't have time 
for generalities. Meanwhile ..." 

Meanwhile, it was a privilege. Straight from the horse's mouth into the 
notebook. The boys scribbled like mad. 

Tall and rather thin but upright, the Director advanced into the room. 
He had a long chin and big rather prominent teeth, just covered, when 
he was not talking, by his full, floridly curved lips. Old, young? Thirty? 
Fifty? Fifty-five? It was hard to say. And anyhow the question didn't 
arise; in this year of stability, A. F. 632, it didn't occur to you to ask it. 

"I shall begin at the beginning," said the D.H.C. and the more zealous 
students recorded his intention in their notebooks: Begin at the begin- 
ning. "These," he waved his hand, "are the incubators." And opening 
an insulated door he showed them racks upon racks of numbered test- 
tubes. "The week's supply of ova. Kept," he explained, "at blood heat; 
whereas the male gametes," and here he opened another door, "they 
have to be kept at thirty-five instead of thirty-seven. Full blood heat 
sterilizes." Rams wrapped in theremogene beget no lambs. 

Still leaning against the incubators he gave them, while the pencils 
scurried illegibly across the pages, a brief description of the modern 



fertilizing process; spoke first, of course, of its surgical introduc- 
tion-"the operation undergone voluntarily for the good of Society, not 
to mention the fact that it carries a bonus amounting to six months' 
salary"; continued with some account of the technique for preserving 
the excised ovary alive and actively developing; passed on to a consid- 
eration of optimum temperature, salinity, viscosity; referred to the liq- 
uor in which the detached and ripened eggs were kept; and, leading 
his charges to the work tables, actually showed them how this liquor 
was drawn off from the test-tubes; how it was let out drop by drop 
onto the specially warmed slides of the microscopes; how the eggs 
which it contained were inspected for abnormalities, counted and 
transferred to a porous receptacle; how (and he now took them to 
watch the operation) this receptacle was immersed in a warm bouillon 
containing free-swimming spermatozoa-at a minimum concentration 
of one hundred thousand per cubic centimetre, he insisted; and how, 
after ten minutes, the container was lifted out of the liquor and its 
contents re-examined; how, if any of the eggs remained unfertilized, it 
was again immersed, and, if necessary, yet again; how the fertilized 
ova went back to the incubators; where the Alphas and Betas re- 
mained until definitely bottled; while the Gammas, Deltas and Epsilons 
were brought out again, after only thirty-six hours, to undergo Bo- 
kanovsky's Process. 

"Bokanovsky's Process," repeated the Director, and the students un- 
derlined the words in their little notebooks. 

One egg, one embryo, one adult-normality. But a bokanovskified egg 
will bud, will proliferate, will divide. From eight to ninety-six buds, and 
every bud will grow into a perfectly formed embryo, and every embryo 
into a full-sized adult. Making ninety-six human beings grow where 
only one grew before. Progress. 

"Essentially," the D.H.C. concluded, "bokanovskification consists of a 
series of arrests of development. We check the normal growth and, 
paradoxically enough, the egg responds by budding." 

Responds by budding. The pencils were busy. 

He pointed. On a very slowly moving band a rack-full of test-tubes was 
entering a large metal box, another, rack-full was emerging. Machinery 
faintly purred. It took eight minutes for the tubes to go through, he 



told them. Eight minutes of hard X-rays being about as much as an 
egg can stand. A few died; of the rest, the least susceptible divided 
into two; most put out four buds; some eight; all were returned to the 
incubators, where the buds began to develop; then, after two days, 
were suddenly chilled, chilled and checked. Two, four, eight, the buds 
in their turn budded; and having budded were dosed almost to death 
with alcohol; consequently burgeoned again and having budded-bud 
out of bud out of bud-were thereafter-further arrest being generally 
fatal-left to develop in peace. By which time the original egg was in a 
fair way to becoming anything from eight to ninety-six embryos- a 
prodigious improvement, you will agree, on nature. Identical twins-but 
not in piddling twos and threes as in the old viviparous days, when an 
egg would sometimes accidentally divide; actually by dozens, by 
scores at a time. 

"Scores," the Director repeated and flung out his arms, as though he 
were distributing largesse. "Scores." 

But one of the students was fool enough to ask where the advantage 
lay. 

"My good boy!" The Director wheeled sharply round on him. "Can't you 
see? Can't you see?" He raised a hand; his expression was solemn. 
"Bokanovsky's Process is one of the major instruments of social stabil- 
ity!" 

Major instruments of social stability. 

Standard men and women; in uniform batches. The whole of a small 
factory staffed with the products of a single bokanovskified egg. 

"Ninety-six identical twins working ninety-six identical machines!" The 
voice was almost tremulous with enthusiasm. "You really know where 
you are. For the first time in history." He quoted the planetary motto. 
"Community, Identity, Stability." Grand words. "If we could bo- 
kanovskify indefinitely the whole problem would be solved." 

Solved by standard Gammas, unvarying Deltas, uniform Epsilons. Mil- 
lions of identical twins. The principle of mass production at last applied 
to biology. 



"But, alas," the Director shook his head, "we can't bokanovskify indefi- 
nitely." 

Ninety-six seemed to be the limit; seventy-two a good average. From 
the same ovary and with gametes of the same male to manufacture as 
many batches of identical twins as possible-that was the best (sadly a 
second best) that they could do. And even that was difficult. 

"For in nature it takes thirty years for two hundred eggs to reach ma- 
turity. But our business is to stabilize the population at this moment, 
here and now. Dribbling out twins over a quarter of a century-what 
would be the use of that?" 

Obviously, no use at all. But Podsnap's Technique had immensely ac- 
celerated the process of ripening. They could make sure of at least a 
hundred and fifty mature eggs within two years. Fertilize and bo- 
kanovskify-in other words, multiply by seventy-two-and you get an 
average of nearly eleven thousand brothers and sisters in a hundred 
and fifty batches of identical twins, all within two years of the same 
age. 

"And in exceptional cases we can make one ovary yield us over fifteen 
thousand adult individuals." 

Beckoning to a fair-haired, ruddy young man who happened to be 
passing at the moment. "Mr. Foster," he called. The ruddy young man 
approached. "Can you tell us the record for a single ovary, Mr. Foster?" 

"Sixteen thousand and twelve in this Centre," Mr. Foster replied with- 
out hesitation. He spoke very quickly, had a vivacious blue eye, and 
took an evident pleasure in quoting figures. "Sixteen thousand and 
twelve; in one hundred and eighty-nine batches of identicals. But of 
course they've done much better," he rattled on, "in some of the tropi- 
cal Centres. Singapore has often produced over sixteen thousand five 
hundred; and Mombasa has actually touched the seventeen thousand 
mark. But then they have unfair advantages. You should see the way a 
negro ovary responds to pituitary! It's quite astonishing, when you're 
used to working with European material. Still," he added, with a laugh 
(but the light of combat was in his eyes and the lift of his chin was 
challenging), "still, we mean to beat them if we can. I'm working on a 
wonderful Delta-Minus ovary at this moment. Only just eighteen 



months old. Over twelve thousand seven hundred children already, ei- 
ther decanted or in embryo. And still going strong. We'll beat them 
yet." 

"That's the spirit I like!" cried the Director, and clapped Mr. Foster on 
the shoulder. "Come along with us, and give these boys the benefit of 
your expert knowledge." 

Mr. Foster smiled modestly. "With pleasure." They went. 
In the Bottling Room all was harmonious bustle and ordered activity. 
Flaps of fresh sow's peritoneum ready cut to the proper size came 
shooting up in little lifts from the Organ Store in the sub-basement. 
Whizz and then, click! the lift-hatches hew open; the bottle-liner had 
only to reach out a hand, take the flap, insert, smooth-down, and be- 
fore the lined bottle had had time to travel out of reach along the end- 
less band, whizz, click! another flap of peritoneum had shot up from 
the depths, ready to be slipped into yet another bottle, the next of that 
slow interminable procession on the band. 

Next to the Liners stood the Matriculators. The procession advanced; 
one by one the eggs were transferred from their test-tubes to the 
larger containers; deftly the peritoneal lining was slit, the morula 
dropped into place, the saline solution poured in ... and already the 
bottle had passed, and it was the turn of the labellers. Heredity, date 
of fertilization, membership of Bokanovsky Group-details were trans- 
ferred from test-tube to bottle. No longer anonymous, but named, 
identified, the procession marched slowly on; on through an opening in 
the wall, slowly on into the Social Predestination Room. 
"Eighty-eight cubic metres of card-index," said Mr. Foster with relish, 
as they entered."""


def create_setup_and_compute(model_names: List[str],
                             gpu: bool = True,
                             tensorflow: bool = False,
                             average_over: int = 3,
                             torchscript: bool = False,
                             xla: bool = False,
                             amp: bool = False,
                             fp16: bool = False,
                             save_to_csv: bool = False,
                             csv_filename: str = f"results_{round(time())}.csv"):
    if xla:
        tf.config.optimizer.set_jit(True)
    if amp:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

    if tensorflow:
        dictionary = {model_name: {} for model_name in model_names}
        results = _compute_tensorflow(model_names, dictionary, average_over, amp)
    else:
        device = 'cuda' if (gpu and torch.cuda.is_available()) else 'cpu'
        dictionary = {model_name: {} for model_name in model_names}
        results = _compute_pytorch(model_names, dictionary, average_over, device, torchscript, fp16)

    print("=========== RESULTS ===========")
    for model_name in model_names:
        print("\t" + f"======= MODEL CHECKPOINT: {model_name} =======")
        for batch_size in results[model_name]["bs"]:
            print("\t\t" + f"===== BATCH SIZE: {batch_size} =====")
            for slice_size in results[model_name]["ss"]:
                result = results[model_name]['results'][batch_size][slice_size]
                if isinstance(result, str):
                    print(f"\t\t{model_name}/{batch_size}/{slice_size}: "
                          f"{result}")
                else:
                    print(f"\t\t{model_name}/{batch_size}/{slice_size}: "
                          f"{(round(1000 * result) / 1000)}"
                          f"s")

    if save_to_csv:
        with open(csv_filename, mode='w') as csv_file:
            fieldnames = ['model',
                          '1x8', '1x64', '1x128', '1x256', '1x512', '1x1024',
                          '2x8', '2x64', '2x128', '2x256', '2x512', '2x1024',
                          '4x8', '4x64', '4x128', '4x256', '4x512', '4x1024',
                          '8x8', '8x64', '8x128', '8x256', '8x512', '8x1024',
                          ]

            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for model_name in model_names:
                model_results = {
                    f'{bs}x{ss}': results[model_name]['results'][bs][ss]
                    for bs in results[model_name]["results"]
                    for ss in results[model_name]['results'][bs]
                }
                writer.writerow({'model': model_name, **model_results})


def _compute_pytorch(model_names, dictionary, average_over, device, torchscript, fp16):
    for c, model_name in enumerate(model_names):
        print(f"{c + 1} / {len(model_names)}")
        config = AutoConfig.from_pretrained(model_name, torchscript=torchscript)
        model = AutoModel.from_pretrained(model_name, config=config)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        tokenized_sequence = tokenizer.encode(input_text, add_special_tokens=False)

        max_input_size = tokenizer.max_model_input_sizes[model_name]
        batch_sizes = [1, 2, 4, 8]
        slice_sizes = [8, 64, 128, 256, 512, 1024]

        dictionary[model_name] = {"bs": batch_sizes, "ss": slice_sizes, "results": {}}
        dictionary[model_name]["results"] = {i: {} for i in batch_sizes}

        for batch_size in batch_sizes:
            if fp16:
                model.half()
            model.to(device)
            model.eval()
            for slice_size in slice_sizes:
                if max_input_size is not None and slice_size > max_input_size:
                    dictionary[model_name]["results"][batch_size][slice_size] = "N/A"
                else:
                    sequence = torch.tensor(tokenized_sequence[:slice_size], device=device).repeat(batch_size, 1)
                    try:
                        if torchscript:
                            print("Tracing model with sequence size", sequence.shape)
                            inference = torch.jit.trace(model, sequence)
                            inference(sequence)
                        else:
                            inference = model
                            inference(sequence)

                        print("Going through model with sequence of shape", sequence.shape)
                        runtimes = timeit.repeat(lambda: inference(sequence), repeat=average_over, number=3)
                        average_time = sum(runtimes)/float(len(runtimes)) / 3.0
                        dictionary[model_name]["results"][batch_size][slice_size] = average_time
                    except RuntimeError as e:
                        print("Doesn't fit on GPU.", e)
                        torch.cuda.empty_cache()
                        dictionary[model_name]["results"][batch_size][slice_size] = "N/A"
    return dictionary


def _compute_tensorflow(model_names, dictionary, average_over, amp):
    for c, model_name in enumerate(model_names):
        print(f"{c + 1} / {len(model_names)}")
        config = AutoConfig.from_pretrained(model_name)
        model = TFAutoModel.from_pretrained(model_name, config=config)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        tokenized_sequence = tokenizer.encode(input_text, add_special_tokens=False)

        max_input_size = tokenizer.max_model_input_sizes[model_name]
        batch_sizes = [1, 2, 4, 8]
        slice_sizes = [8, 64, 128, 256, 512, 1024]

        dictionary[model_name] = {"bs": batch_sizes, "ss": slice_sizes, "results": {}}
        dictionary[model_name]["results"] = {i: {} for i in batch_sizes}

        print("Using model", model)

        @tf.function
        def inference(inputs):
            return model(inputs)

        for batch_size in batch_sizes:
            for slice_size in slice_sizes:
                if max_input_size is not None and slice_size > max_input_size:
                    dictionary[model_name]["results"][batch_size][slice_size] = "N/A"
                else:
                    sequence = tf.stack([tf.squeeze(tf.constant(tokenized_sequence[:slice_size])[None, :])] * batch_size)

                    try:
                        print("Going through model with sequence of shape", sequence.shape)
                        # To make sure that the model is traced + that the tensors are on the appropriate device
                        inference(sequence)

                        runtimes = timeit.repeat(lambda: inference(sequence), repeat=average_over, number=3)
                        average_time = sum(runtimes)/float(len(runtimes)) / 3.0
                        dictionary[model_name]["results"][batch_size][slice_size] = average_time
                    except tf.errors.ResourceExhaustedError as e:
                        print("Doesn't fit on GPU.", e)
                        torch.cuda.empty_cache()
                        dictionary[model_name]["results"][batch_size][slice_size] = "N/A"
    return dictionary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--models", required=False, type=str, default='all', help="Model checkpoints to be provided "
                                                                                  "to the AutoModel classes. Leave "
                                                                                  "blank to benchmark the base version "
                                                                                  "of all available model "
                                                                                  "architectures.")
    parser.add_argument("--torch", required=False, action="store_true", help="Benchmark the Pytorch version of the "
                                                                             "models")
    parser.add_argument("--torch_cuda", required=False, action="store_true", help="Pytorch only: run on available "
                                                                                  "cuda devices")
    parser.add_argument("--torchscript", required=False, action="store_true", help="Pytorch only: trace the models "
                                                                                   "using torchscript")
    parser.add_argument("--tensorflow", required=False, action="store_true", help="Benchmark the TensorFlow version "
                                                                                  "of the models. Will run on GPU if "
                                                                                  "the correct dependencies are "
                                                                                  "installed")
    parser.add_argument("--xla", required=False, action="store_true", help="TensorFlow only: use XLA acceleration.")
    parser.add_argument("--amp", required=False, action="store_true", help="TensorFlow only: use automatic mixed precision acceleration.")
    parser.add_argument("--fp16", required=False, action="store_true", help="PyTorch only: use FP16 to accelerate inference.")
    parser.add_argument("--keras_predict", required=False, action="store_true", help="Whether to use model.predict "
                                                                                     "instead of model() to do a "
                                                                                     "forward pass.")
    parser.add_argument("--save_to_csv", required=False, action="store_true", help="Save to a CSV file.")
    parser.add_argument("--csv_filename", required=False, default=None, help="CSV filename used if saving results to csv.")
    parser.add_argument("--average_over", required=False, default=30, type=int, help="Times an experiment will be run.")

    args = parser.parse_args()
    if args.models == 'all':
        args.models = [
            "gpt2",
            "bert-base-cased",
            "xlnet-base-cased",
            "xlm-mlm-en-2048",
            "transfo-xl-wt103",
            "openai-gpt",
            "distilbert-base-uncased",
            "distilgpt2",
            "roberta-base",
            "ctrl"
        ]
    else:
        args.models = args.models.split()

    print("Running with arguments", args)

    if args.torch:
        if is_torch_available():
            create_setup_and_compute(
                model_names=args.models,
                tensorflow=False,
                gpu=args.torch_cuda,
                torchscript=args.torchscript,
                fp16=args.fp16,
                save_to_csv=args.save_to_csv,
                csv_filename=args.csv_filename,
                average_over=args.average_over
            )
        else:
            raise ImportError("Trying to run a PyTorch benchmark but PyTorch was not found in the environment.")

    if args.tensorflow:
        if is_tf_available():
            create_setup_and_compute(
                model_names=args.models,
                tensorflow=True,
                xla=args.xla,
                amp=args.amp,
                save_to_csv=args.save_to_csv,
                csv_filename=args.csv_filename,
                average_over=args.average_over
            )
        else:
            raise ImportError("Trying to run a TensorFlow benchmark but TensorFlow was not found in the environment.")

if __name__ == '__main__':
    main()

