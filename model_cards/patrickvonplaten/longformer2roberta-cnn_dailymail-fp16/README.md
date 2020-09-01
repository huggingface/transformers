# Longformer2Roberta Summarization with 🤗 EncoderDecoder Framework

This model is a Longformer2Roberta model fine-tuned on summarization.

Longformer2Roberta is a `EncoderDecoderModel`, meaning that both the encoder is a `allenai/longformer-base-4096` model and the decoder is a `roberta-base` model. Leveraging the [EncoderDecoderFramework](https://huggingface.co/transformers/model_doc/encoderdecoder.html#encoder-decoder-models), the 
two pretrained models can simply be loaded into the framework via:

```python
roberta2roberta = EncoderDecoderModel.from_encoder_decoder_pretrained("allenai/longformer-base-4096", "roberta-base")
```

The decoder of an `EncoderDecoder` model needs cross-attention layers and usually makes use of causal 
masking for auto-regressiv generation.
Thus, ``longformer2roberta`` is consequently fined-tuned on the `CNN/Daily Mail`dataset and the resulting model
`longformer2roberta-cnn_dailymail-fp16` is uploaded here.

## Example

The model is by no means a state-of-the-art model, but nevertheless 
produces reasonable summarization results. It was mainly fine-tuned 
as a proof-of-concept for the 🤗 EncoderDecoder Framework.

The model can be used as follows:

```python
from transformers import LongformerTokenizer, EncoderDecoderModel

model = EncoderDecoderModel.from_pretrained("patrickvonplaten/longformer2roberta-cnn_dailymail-fp16")
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096") 

article = """(CNN)James Holmes made his introduction to the world in a Colorado cinema filled with spectators watching a midnight showing of the new Batman movie, "The Dark Knight Rises," in June 2012. The moment became one of the deadliest shootings in U.S. history. Holmes is accused of opening fire on the crowd, killing 12 people and injuring or maiming 70 others in Aurora, a suburb of Denver. Holmes appeared like a comic book character: He resembled the Joker, with red-orange hair, similar to the late actor Heath Ledger\'s portrayal of the villain in an earlier Batman movie, authorities said. But Holmes was hardly a cartoon. Authorities said he wore body armor and carried several guns, including an AR-15 rifle, with lots of ammo. He also wore a gas mask. Holmes says he was insane at the time of the shootings, and that is his legal defense and court plea: not guilty by reason of insanity. Prosecutors aren\'t swayed and will seek the death penalty. Opening statements in his trial are scheduled to begin Monday. Holmes admits to the shootings but says he was suffering "a psychotic episode" at the time,  according to court papers filed in July 2013 by the state public defenders, Daniel King and Tamara A. Brady. Evidence "revealed thus far in the case supports the defense\'s position that Mr. Holmes suffers from a severe mental illness and was in the throes of a psychotic episode when he committed the acts that resulted in the tragic loss of life and injuries sustained by moviegoers on July 20, 2012," the public defenders wrote. Holmes no longer looks like a dazed Joker, as he did in his first appearance before a judge in 2012. He appeared dramatically different in January when jury selection began for his trial: 9,000 potential jurors were summoned for duty, described as one of the nation\'s largest jury calls. Holmes now has a cleaner look, with a mustache, button-down shirt and khaki pants. In January, he had a beard and eyeglasses. If this new image sounds like one of an academician, it may be because Holmes, now 27, once was one. Just before the shooting, Holmes was a doctoral student in neuroscience, and he was studying how the brain works, with his schooling funded by a U.S. government grant. Yet for all his learning, Holmes apparently lacked the capacity to command his own mind, according to the case against him. A jury will ultimately decide Holmes\' fate. That panel is made up of 12 jurors and 12 alternates. They are 19 women and five men, and almost all are white and middle-aged. The trial could last until autumn. When jury summonses were issued in January, each potential juror stood a 0.2% chance of being selected, District Attorney George Brauchler told the final jury this month. He described the approaching trial as "four to five months of a horrible roller coaster through the worst haunted house you can imagine." The jury will have to render verdicts on each of the 165 counts against Holmes, including murder and attempted murder charges. Meanwhile, victims and their relatives are challenging all media outlets "to stop the gratuitous use of the name and likeness of mass killers, thereby depriving violent individuals the media celebrity and media spotlight they so crave," the No Notoriety group says. They are joined by victims from eight other mass shootings in recent U.S. history. Raised in central coastal California and in San Diego, James Eagan Holmes is the son of a mathematician father noted for his work at the FICO firm that provides credit scores and a registered nurse mother, according to the U-T San Diego newspaper. Holmes also has a sister, Chris, a musician, who\'s five years younger, the newspaper said. His childhood classmates remember him as a clean-cut, bespectacled boy with an "exemplary" character who "never gave any trouble, and never got in trouble himself," The Salinas Californian reported. His family then moved down the California coast, where Holmes grew up in the San Diego-area neighborhood of Rancho Peñasquitos, which a neighbor described as "kind of like Mayberry," the San Diego newspaper said. Holmes attended Westview High School, which says its school district sits in "a primarily middle- to upper-middle-income residential community." There, Holmes ran cross-country, played soccer and later worked at a biotechnology internship at the Salk Institute and Miramar College, which attracts academically talented students. By then, his peers described him as standoffish and a bit of a wiseacre, the San Diego newspaper said. Holmes attended college fairly close to home, in a neighboring area known as Southern California\'s "inland empire" because it\'s more than an hour\'s drive from the coast, in a warm, low-desert climate. He entered the University of California, Riverside, in 2006 as a scholarship student. In 2008 he was a summer camp counselor for disadvantaged children, age 7 to 14, at Camp Max Straus, run by Jewish Big Brothers Big Sisters of Los Angeles. He graduated from UC Riverside in 2010 with the highest honors and a bachelor\'s degree in neuroscience. "Academically, he was at the top of the top," Chancellor Timothy P. White said. He seemed destined for even higher achievement. By 2011, he had enrolled as a doctoral student in the neuroscience program at the University of Colorado Anschutz Medical Campus in Aurora, the largest academic health center in the Rocky Mountain region. The doctoral in neuroscience program attended by Holmes focuses on how the brain works, with an emphasis on processing of information, behavior, learning and memory. Holmes was one of six pre-thesis Ph.D. students in the program who were awarded a neuroscience training grant from the National Institutes of Health. The grant rewards outstanding neuroscientists who will make major contributions to neurobiology. A syllabus that listed Holmes as a student at the medical school shows he was to have delivered a presentation about microRNA biomarkers. But Holmes struggled, and his own mental health took an ominous turn. In March 2012, he told a classmate he wanted to kill people, and that he would do so "when his life was over," court documents said. Holmes was "denied access to the school after June 12, 2012, after he made threats to a professor," according to court documents. About that time, Holmes was a patient of University of Colorado psychiatrist Lynne Fenton. Fenton was so concerned about Holmes\' behavior that she mentioned it to her colleagues, saying he could be a danger to others, CNN affiliate KMGH-TV reported, citing sources with knowledge of the investigation. Fenton\'s concerns surfaced in early June, sources told the Denver station. Holmes began to fantasize about killing "a lot of people" in early June, nearly six weeks before the shootings, the station reported, citing unidentified sources familiar with the investigation. Holmes\' psychiatrist contacted several members of a "behavioral evaluation and threat assessment" team to say Holmes could be a danger to others, the station reported. At issue was whether to order Holmes held for 72 hours to be evaluated by mental health professionals, the station reported. "Fenton made initial phone calls about engaging the BETA team" in "the first 10 days" of June, but it "never came together" because in the period Fenton was having conversations with team members, Holmes began the process of dropping out of school, a source told KMGH. Defense attorneys have rejected the prosecution\'s assertions that Holmes was barred from campus. Citing statements from the university, Holmes\' attorneys have argued that his access was revoked because that\'s normal procedure when a student drops enrollment. What caused this turn for the worse for Holmes has yet to be clearly detailed. In the months before the shooting, he bought four weapons and more than 6,000 rounds of ammunition, authorities said. Police said he also booby-trapped his third-floor apartment with explosives, but police weren\'t fooled. After Holmes was caught in the cinema parking lot immediately after the shooting, bomb technicians went to the apartment and neutralized the explosives. No one was injured at the apartment building. Nine minutes before Holmes went into the movie theater, he called a University of Colorado switchboard, public defender Brady has said in court. The number he called can be used to get in contact with faculty members during off hours, Brady said. Court documents have also revealed that investigators have obtained text messages that Holmes exchanged with someone before the shooting. That person was not named, and the content of the texts has not been made public. According to The New York Times, Holmes sent a text message to a fellow graduate student, a woman, about two weeks before the shooting. She asked if he had left Aurora yet, reported the newspaper, which didn\'t identify her. No, he had two months left on his lease, Holmes wrote back, according to the Times. He asked if she had heard of "dysphoric mania," a form of bipolar disorder marked by the highs of mania and the dark and sometimes paranoid delusions of major depression. The woman asked if the disorder could be managed with treatment. "It was," Holmes wrote her, according to the Times. But he warned she should stay away from him "because I am bad news," the newspaper reported. It was her last contact with Holmes. After the shooting, Holmes\' family issued a brief statement: "Our hearts go out to those who were involved in this tragedy and to the families and friends of those involved," they said, without giving any information about their son. Since then, prosecutors have refused to offer a plea deal to Holmes. For Holmes, "justice is death," said Brauchler, the district attorney. In December, Holmes\' parents, who will be attending the trial, issued another statement: They asked that their son\'s life be spared and that he be sent to an institution for mentally ill people for the rest of his life, if he\'s found not guilty by reason of insanity. "He is not a monster," Robert and Arlene Holmes wrote, saying the death penalty is "morally wrong, especially when the condemned is mentally ill." "He is a human being gripped by a severe mental illness," the parents said. The matter will be settled by the jury. CNN\'s Ana Cabrera and Sara Weisfeldt contributed to this report from Denver."""

input_ids = tokenizer(article, return_tensors="pt").input_ids
output_ids = model.generate(input_ids)

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
# should produce
# James Holmes, 27, is accused of opening fire on a Colorado theater.
# He was a doctoral student at University of Colorado.
# Holmes says he was suffering "a psychotic episode" at the time of the shooting.
# Prosecutors won't say whether Holmes was barred from campus.
```

Such an article has a length of > 2000 tokens, which means that it cannot be handled correctly by Bert or Roberta encoders.

## Training script:

**IMPORTANT**: In order for this code to work, make sure you checkout to the branch 
[more_general_trainer_metric](https://github.com/huggingface/transformers/tree/more_general_trainer_metric), which slightly adapts 
the `Trainer` for `EncoderDecoderModels` according to this PR: https://github.com/huggingface/transformers/pull/5840. 

The following code shows the complete training script that was used to fine-tune `longformer2roberta-cnn_dailymail-fp16
` for reproducability. The training last ~90h on a standard GPU.

```python
#!/usr/bin/env python3
import nlp
import logging
from transformers import LongformerTokenizer, EncoderDecoderModel, Trainer, TrainingArguments

logging.basicConfig(level=logging.INFO)

model = EncoderDecoderModel.from_encoder_decoder_pretrained("allenai/longformer-base-4096", "roberta-base")
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")

# load train and validation data
train_dataset = nlp.load_dataset("cnn_dailymail", "3.0.0", split="train")
val_dataset = nlp.load_dataset("cnn_dailymail", "3.0.0", split="validation[:5%]")

# load rouge for validation
rouge = nlp.load_metric("rouge", experiment_id=0)

# enable gradient checkpointing for longformer encoder
model.encoder.config.gradient_checkpointing = True

# set decoding params
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.max_length = 142
model.config.min_length = 56
model.config.no_repeat_ngram_size = 3
model.early_stopping = True
model.length_penalty = 2.0
model.num_beams = 4

encoder_length = 2048
decoder_length = 128
batch_size = 16


# map data correctly
def map_to_encoder_decoder_inputs(batch):
    # Tokenizer will automatically set [BOS] <text> [EOS]
    # cut off at Longformer at 2048
    inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=encoder_length)
    # force summarization <= 128
    outputs = tokenizer(batch["highlights"], padding="max_length", truncation=True, max_length=decoder_length)

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask

    # set 128 tokens to global attention
    batch["global_attention_mask"] = [[1 if i < 128 else 0 for i in range(sequence_length)] for sequence_length in len(inputs.input_ids) * [encoder_length]]
    batch["decoder_input_ids"] = outputs.input_ids
    batch["labels"] = outputs.input_ids.copy()
    # mask loss for padding
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
    ]
    batch["decoder_attention_mask"] = outputs.attention_mask

    assert all([len(x) == encoder_length for x in inputs.input_ids])
    assert all([len(x) == decoder_length for x in outputs.input_ids])

    return batch


def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.eos_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }


    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }


# make train dataset ready
train_dataset = train_dataset.map(
    map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["article", "highlights"],
)
train_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "global_attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

# same for validation dataset
val_dataset = val_dataset.map(
    map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["article", "highlights"],
)
val_dataset.set_format(
    type="torch", columns=["input_ids", "global_attention_mask", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

# set training arguments - these params are not really tuned, feel free to change
training_args = TrainingArguments(
    output_dir="./",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    predict_from_generate=True,
    evaluate_during_training=True,
    do_train=True,
    do_eval=True,
    logging_steps=1000,
    save_steps=1000,
    eval_steps=1000,
    overwrite_output_dir=True,
    warmup_steps=2000,
    save_total_limit=3,
    fp16=True,
)

# instantiate trainer
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# start training
trainer.train()
```

## Evaluation

The following script evaluates the model on the test set of 
CNN/Daily Mail.

```python
#!/usr/bin/env python3
import nlp
import torch
from transformers import LongformerTokenizer, EncoderDecoderModel

tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
model = EncoderDecoderModel.from_pretrained("patrickvonplaten/longformer2roberta-cnn_dailymail-fp16")
model.to("cuda")

test_dataset = nlp.load_dataset("cnn_dailymail", "3.0.0", split="test")
batch_size = 32

encoder_length = 2048
decoder_length = 128


# map data correctly
def generate_summary(batch):
    # Tokenizer will automatically set [BOS] <text> [EOS]
    # cut off at BERT max length 512
    inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=encoder_length, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")
    global_attention_mask = torch.zeros_like(attention_mask)
    global_attention_mask[:, :decoder_length] = 1

    outputs = model.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)

    # all special tokens including will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch["pred"] = output_str

    return batch


results = test_dataset.map(generate_summary, batched=True, batch_size=batch_size, remove_columns=["article"])

# load rouge for validation
rouge = nlp.load_metric("rouge")

pred_str = results["pred"]
label_str = results["highlights"]

rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

print(rouge_output)
```

The obtained results should be:

| -   |      Rouge2 - mid -precision      |  Rouge2 - mid - recall | Rouge2 - mid - fmeasure |
|----------|:-------------:|:------:|:------:|
| **CNN/Daily Mail** |  12.39 | 15.05 | **13.21** |

**Note** This model was trained to show how Longformer can be used as an Encoder model in a EncoderDecoder setup. 
Better results are obtained for datasets of much longer inputs.
