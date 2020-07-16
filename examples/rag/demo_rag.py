import torch

from transformers import (
    DPRContextEncoderTokenizer,
    DPRQuestionEncoder,
    PretrainedConfig,
    PreTrainedModel,
    RagConfig,
    RagDefaultSequenceModel,
    RagDefaultTokenizer,
    RagDefaultTokenModel,
    RagSequenceModel,
    RagTestSequenceModel,
    Retriever,
    T5ForConditionalGeneration,
    T5Tokenizer,
)
from transformers.modeling_rag import RagModel


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# Creating a RAG model on with a DPR question encoder and a T5 generator
class RagWithT5SequenceModel(RagSequenceModel):
    def __init__(self, config):
        dpr_tokenizer = DPRContextEncoderTokenizer.from_pretrained(config.pretrained_context_tokenizer_name_or_path)
        dpr_question_encoder = DPRQuestionEncoder.from_pretrained(config.pretrained_question_encoder_name_or_path)
        dpr_retriever = Retriever(
            config.dataset,
            dataset_name=config.dataset_name,
            dataset_split=config.dataset_split,
            index_name=config.index_name,
        )
        t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
        t5 = T5ForConditionalGeneration.from_pretrained("t5-base")
        super().__init__(config, dpr_retriever, dpr_tokenizer, t5, t5_tokenizer, dpr_question_encoder)


def generate_from_rag(rag_model, questions, inputs, num_beams=4):
    with torch.no_grad():
        rag_model = rag_model.eval().to(device)
        outputs = rag_model.generate(
            inputs,
            num_beams=num_beams,
            min_length=1,  # make sure short answers are allowed
            max_length=10,  # no need for crazy long answers in NQ
            early_stopping=False,
            num_return_sequences=num_beams,
            bad_words_ids=[[0, 0]]
            # BART likes to repeat BOS tokens, dont allow it to generate more than one
        )
        answers = rag_model.model.generator_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for i in range(0, len(questions)):
            print("Question: " + questions[i])
            print(f"Top {num_beams} Answers: ", answers[i * num_beams : (i + 1) * num_beams])


if __name__ == "__main__":
    # the question mark seems crucial for the success of retrieval
    questions = [
        "who sings does he love me with reba",
        "who does eric roberts play in in The Dark Knight",
        "what is the capital of the United Kingdom",
    ]
    rag_tokenizer = RagDefaultTokenizer.from_pretrained("facebook/bart-large")
    inputs = rag_tokenizer.batch_encode_plus(questions, return_tensors="pt", padding=True, truncation=True)[
        "input_ids"
    ].to(device)

    print("RAG with T5 MODEL")
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
    t5_inputs = t5_tokenizer.batch_encode_plus(questions, return_tensors="pt", padding=True, truncation=True)[
        "input_ids"
    ].to(device)
    rag_sequence_model_path = "/private/home/piktus/huggingface_rag/data/rag-sequence-nq"
    rag_sequence_config = RagConfig.from_pretrained(rag_sequence_model_path)
    rag_model = RagWithT5SequenceModel(rag_sequence_config)
    generate_from_rag(rag_model, questions, t5_inputs, num_beams=4)

    print("TOKEN MODEL")
    rag_token_model_path = "/private/home/piktus/huggingface_rag/data/rag-token-nq"
    rag_token_config = RagConfig.from_pretrained(rag_token_model_path)
    rag_model = RagDefaultTokenModel(rag_token_config)
    generate_from_rag(rag_model, questions, inputs, num_beams=4)

    print("SEQUENCE MODEL")
    rag_sequence_model_path = "/private/home/piktus/huggingface_rag/data/rag-sequence-nq"
    rag_sequence_config = RagConfig.from_pretrained(rag_sequence_model_path)
    rag_model = RagDefaultSequenceModel(rag_sequence_config)
    generate_from_rag(rag_model, questions, inputs, num_beams=4)

    rag_sequence_model_path = "/private/home/piktus/huggingface_rag/data/rag-sequence-nq"
    rag_sequence_test_config = RagConfig.from_pretrained(
        rag_sequence_model_path, dataset_name="dummy_psgs_w100_no_embeddings"
    )
    rag_model = RagTestSequenceModel(rag_sequence_test_config)
    generate_from_rag(rag_model, questions, inputs, num_beams=4)

    # Retrieved contexts
    """
    Fabio Lanzoni / Fabio Lanzoni (born 15 March 1959), known simply as Fabio, is an Italian-American actor, fashion model and spokesperson.
    He is for known his appearances as spokesman for I Can't Believe It's Not Butter! and the American Cancer Society. He was also a romance novel cover model throughout the 1980s and 1990s.
    His acting credits include the movies Death Becomes Her (1992),
    """

    """Hurricane Florence (2006) / Hurricane Florence was the first Atlantic hurricane to produce hurricane force winds
     on Bermuda since Hurricane Fabian hit the island in September 2003. The seventh tropical storm and second hurricane of the 2006 Atlantic hurricane season,
      Florence developed from a tropical wave in the tropical Atlantic Ocean on September 3 and followed the track of a Cape Verde-type hurricane.
      Because of unfavorable conditions, the system failed to organize at first, and as a result the storm grew to an unusually large size.
      After several days, Florence encountered an area of lesser wind shear and strengthened into a hurricane on September 10.
    """

    """
    Sal Maroni / Salvatore "The Boss" Maroni is a fictional Batman character who is head leader of the Gotham Mafia after Carmine Falcone
     was killed or taken to Arkham Asylum. In some cases, he was the man who made Havey Dent turn into Two-Face.
     He was played by Eric Roberts in The Dark Knight. Category:Batman characters Category:DC Comics characters
    """
