import torch

from transformers import DprConfig, \
    DprQuestionEncoder, \
    DprContextEncoder, \
    DprTokenizer, \
    BartForConditionalGeneration, \
    BartTokenizer, \
    PreTrainedModel, \
    PretrainedConfig
from transformers.modelling_rag import RagModel, Retriever

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def get_rag_config(generator_config, retriever_config, rag_model_type):
    return PretrainedConfig(
        vocab_size=generator_config.vocab_size,
        is_encoder_decoder=True,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        decoder_start_token_id=2,
        title_sep=" / ",
        doc_sep=" // ",
        rag_model_type=rag_model_type,
        k=5,
        generator_config=generator_config,
        retriever_config=retriever_config
    )


def build_rag_model(dpr_model_file, rag_model_path, rag_model_type):
    # load retriever
    dpr_config = DprConfig(biencoder_model_file=dpr_model_file)
    dpr_tokenizer = DprTokenizer.from_pretrained("dpr-model-base")
    dpr_retriever = Retriever(dpr_config, dpr_tokenizer).to(device=device)
    dpr_retriever.load_index()

    # load generator
    generator_tokenizer = BartTokenizer.from_pretrained(
        "facebook/bart-large", )  # additional_special_tokens=[TITLE_SEP, DOC_SEP]) # Patrick: removing for now
    bart = BartForConditionalGeneration.from_pretrained(rag_model_path).to(device=device)
    rag_config = get_rag_config(bart.config, dpr_config, rag_model_type)

    return RagModel(
        config=rag_config,
        retriever=dpr_retriever,
        retriever_tokenizer=dpr_tokenizer,
        generator=bart,
        generator_tokenizer=generator_tokenizer,
    )


def build_rag_sequence():
    return build_rag_model(
        dpr_model_file="/private/home/plewis/rag/rag_hf/data/dpr_retriever/hf_bert_base.cp",
        rag_model_path="/private/home/plewis/rag/rag_hf/data/rag-sequence-nq",
        rag_model_type='rag_sequence'
    )


def build_rag_token():
    return build_rag_model(
        dpr_model_file="/private/home/plewis/rag/rag_hf/data/dpr_retriever/hf_bert_base.cp",
        rag_model_path="/private/home/plewis/rag/rag_hf/data/rag-token-nq",
        rag_model_type='rag_token'
    )


if __name__ == '__main__':
    questions = [
        "what is fabio lanzoni famous for",
        "who does eric roberts play in in The Dark Knight",
        "what year did hurricane florence hit",
    ]
    num_beams = 4
    for build_rag in [build_rag_sequence, build_rag_token]:
        print('*' * 50)
        print(f'Loading {build_rag.__name__}')
        print('*' * 50)
        rag_model = build_rag().eval().to(device)

        inputs = \
        rag_model.generator_tokenizer.batch_encode_plus(questions, return_tensors="pt", pad_to_max_length=True)[
            "input_ids"].to(device)
        outputs = rag_model.generate(inputs,
                                     num_beams=num_beams,
                                     min_length=1,  # make sure short answers are allowed
                                     max_length=10,  # no need for crazy long answers in NQ
                                     early_stopping=False,
                                     num_return_sequences=num_beams,
                                     bad_words_ids=[[0, 0]]
                                     # BART likes to repeat BOS tokens, dont allow it to generate more than one
                                     )
        answers = rag_model.generator_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for i in range(0, len(questions)):
            print("Question: " + questions[i])
            print(f"Top {num_beams} Answers: ", answers[i * num_beams: (i + 1) * num_beams])

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

















