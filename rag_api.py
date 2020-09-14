#!/usr/bin/env python3
# Option 1 - all in one forward pass

tokenizer = RagTokenForGenerationizer.from_pretrained("facebook/rag")
retriever = RagRetriever.from_pretrained("facebook/rag")
rag = RagForSequenceGeneration.from_pretrained("facebook/rage", retriever=retriever)

# IMPORTANT these input_ids are encoded by the question encoder tokenizer!!!
question_encoder_input_ids = tokenizer("How many people live in Paris?", return_pt="pt").input_ids

# 1 step
answer_ids = rag.generate(input_ids=question_encoder_input_ids)


# Option 2 - step by step manually
tokenizer = RagTokenForGenerationizer.from_pretrained("facebook/rag")
retriever = RagRetriever.from_pretrained("facebook/rag")
rag = RagForSequenceGeneration.from_pretrained("facebook/rag") # Note now there is no retriever used as an input

# IMPORTANT these input_ids are encoded by the question encoder tokenizer!!!
question_encoder_input_ids = tokenizer("How many people live in Paris?", return_pt="pt").input_ids

# 3 steps
question_encoder_hidden_states = rag.question_encoder(question_encoder_input_ids)

# retriever takes care of: 
# - decoding question encoder input ids to encoder input str
# - retrieving the document encodings
# - decoding document encodings
# - post processing: concat doc string and encoder input str
# - encoding concat to generator context input ids
context_input_ids, document_scores = rag.retriever(question_encoder_input_ids, question_encoder_hidden_states)
answer_ids = rag.generate(context_input_ids=context_input_ids, document_scores=document_scores)
