#!/usr/bin/env python3
from transformers import T5Tokenizer
from transformers import EncoderDecoderModel
import torch

tok = T5Tokenizer.from_pretrained("./roberta2roberta_L-24_cnn_daily_mail")

article = """(CNN)Sigma Alpha Epsilon is under fire for a video showing party-bound fraternity members singing a racist chant. SAE\'s national chapter suspended the students, but University of Oklahoma President David Boren took it a step further, saying the university\'s affiliation with the fraternity is permanently done. The news is shocking, but it\'s not the first time SAE has faced controversy. SAE was founded March 9, 1856, at the University of Alabama, five years before the American Civil War, according to the fraternity website. When the war began, the group had fewer than 400 members, of which "369 went to war for the Confederate States and seven for the Union Army," the website says. The fraternity now boasts more than 200,000 living alumni, along with about 15,000 undergraduates populating 219 chapters and 20 "colonies" seeking full membership at universities. SAE has had to work hard to change recently after a string of member deaths, many blamed on the hazing of new recruits, SAE national President Bradley Cohen wrote in a message on the fraternity\'s website. The fraternity\'s website lists more than 130 chapters cited or suspended for "health and safety incidents" since 2010. At least 30 of the incidents involved hazing, and dozens more involved alcohol. However, the list is missing numerous incidents from recent months. Among them, according to various media outlets: Yale University banned the SAEs from campus activities last month after members allegedly tried to interfere with a sexual misconduct investigation connected to an initiation rite. Stanford University in December suspended SAE housing privileges after finding sorority members attending a fraternity function were subjected to graphic sexual content. And Johns Hopkins University in November suspended the fraternity for underage drinking. "The media has labeled us as the \'nation\'s deadliest fraternity,\' " Cohen said. In 2011, for example, a student died while being coerced into excessive alcohol consumption, according to a lawsuit. SAE\'s previous insurer dumped the fraternity. "As a result, we are paying Lloyd\'s of London the highest insurance rates in the Greek-letter world," Cohen said. Universities have turned down SAE\'s attempts to open new chapters, and the fraternity had to close 12 in 18 months over hazing incidents."""

input_ids = tok(article, return_tensors="pt").input_ids
#input_ids = torch.cat([torch.tensor([[97]]), input_ids], dim=-1)

model = EncoderDecoderModel.from_pretrained("patrickvonplaten/roberta2roberta_L-24_cnn_daily_mail")

model.config.decoder.bos_token_id = 2
model.config.decoder.pad_token_id = 0
model.config.decoder.eos_token_id = 1
model.config.decoder.decoder_start_token_id = 1
model.config.decoder_start_token_id = 1

model.config.encoder.bos_token_id = 2
model.config.encoder.pad_token_id = 0
model.config.encoder.eos_token_id = 1


outputs = model.generate(input_ids, do_sample=False, max_length=10)

print(tok.decode(outputs[0]))
print(outputs[0])
