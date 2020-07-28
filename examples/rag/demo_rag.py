import torch

from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    DPRContextEncoderTokenizer,
    DPRQuestionEncoder,
    HFRetriever,
    PretrainedConfig,
    PreTrainedModel,
    RagConfig,
    RagDefaultSequenceModel,
    RagDefaultTokenizer,
    RagDefaultTokenModel,
    RagSequenceModel,
    RagTestSequenceModel,
    T5ForConditionalGeneration,
    T5Tokenizer,
)


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# Creating a RAG model on with a DPR question encoder and a T5 generator
# todo: override decoder start token id
class RagWithT5SequenceModel(RagSequenceModel):
    def __init__(self, config):
        dpr_tokenizer = DPRContextEncoderTokenizer.from_pretrained(config.pretrained_context_tokenizer_name_or_path)
        dpr_question_encoder = DPRQuestionEncoder.from_pretrained(config.pretrained_question_encoder_name_or_path)
        dpr_retriever = HFRetriever(
            config.dataset,
            dataset_name=config.dataset_name,
            dataset_split=config.dataset_split,
            index_name=config.index_name,
        )
        t5_tokenizer = T5Tokenizer.from_pretrained(config.pretrained_generator_name_or_path)
        t5 = T5ForConditionalGeneration.from_pretrained(config.pretrained_generator_name_or_path)
        super().__init__(config, dpr_retriever, dpr_tokenizer, t5, t5_tokenizer, dpr_question_encoder)


def generate_from_rag(rag_model, tokenizer, questions, inputs, num_beams=4):
    with torch.no_grad():
        rag_model = rag_model.eval().to(device)
        outputs = rag_model.generate(
            inputs,
            num_beams=num_beams,
            min_length=1,  # make sure short answers are allowed
            max_length=50,  # no need for crazy long answers in NQ
            # early_stopping=True,
            num_return_sequences=num_beams,
            # use_cache=False, - doesn't work for bart
            # repetition_penalty=10,
            bad_words_ids=[[0, 0]],
            # BART likes to repeat BOS tokens, dont allow it to generate more than one
        )
        answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for i in range(0, len(questions)):
            print("Question: " + questions[i])
            print("Top {} answers:".format(num_beams))
            for answer in answers[i * num_beams : (i + 1) * num_beams]:
                print("\t", answer)


if __name__ == "__main__":
    questions = [
        "who is the owner of reading football club",  #  ['Xiu Li Dai', 'Dai Yongge', 'Dai Xiuli', 'Yongge Dai']
        "where does the story the great gatsby take place",  # ['Long Island of 1922']
        "what is the meaning of the name sinead",  #  ['God forgave / God gratified']
        "when was the last time minnesota vikings was in the super bowl",  # ['1969', 'XI']
        "prior to 1948 north and south vietnam were a part of",  # ['France']
        "who sings does he love me with reba",
        "who were the two mathematicians that invented calculus",
        "what parts make up the peripheral nervous system",
    ]

    print("\nRAG with T5 MODEL")
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
    t5_inputs = t5_tokenizer.batch_encode_plus(questions, return_tensors="pt", padding=True, truncation=True)[
        "input_ids"
    ].to(device)
    rag_t5_config = RagConfig(pretrained_generator_name_or_path="t5-base")
    rag_model = RagWithT5SequenceModel(rag_t5_config)
    generate_from_rag(rag_model, t5_tokenizer, questions, t5_inputs, num_beams=4)

    rag_tokenizer = RagDefaultTokenizer.from_pretrained("facebook/bart-large")
    inputs = rag_tokenizer.batch_encode_plus(questions, return_tensors="pt", padding=True, truncation=True)[
        "input_ids"
    ].to(device)

    print("\nTOKEN MODEL - HF retriever")
    rag_token_model_path = "/private/home/piktus/rag_huggingface/data/rag-token-nq"
    rag_token_config = RagConfig.from_pretrained(rag_token_model_path, retriever_type="hf_retriever", uncompressed=True)
    rag_model = RagDefaultTokenModel(rag_token_config)
    generate_from_rag(rag_model, rag_tokenizer, questions, inputs, num_beams=10)

    print("\nSEQUENCE MODEL - HF retriever")
    rag_sequence_model_path = "/private/home/piktus/rag_huggingface/data/rag-sequence-nq"
    rag_sequence_config = RagConfig.from_pretrained(
        rag_sequence_model_path, retriever_type="mpi_retriever", uncompressed=True
    )
    rag_model = RagDefaultSequenceModel(rag_sequence_config)
    generate_from_rag(rag_model, rag_tokenizer, questions, inputs, num_beams=6)


    print("\nSEQUENCE MODEL - MPI retriever")
    rag_sequence_model_path = "/private/home/piktus/rag_huggingface/data/rag-sequence-nq"
    rag_sequence_config = RagConfig.from_pretrained(
        rag_sequence_model_path, retriever_type="mpi_retriever"
    )
    rag_model = RagDefaultSequenceModel(rag_sequence_config)
    generate_from_rag(rag_model, rag_tokenizer, questions, inputs, num_beams=6)

    print("\nSEQUENCE MODEL WITH INDEXING ON THE FLY")
    rag_sequence_model_path = "/private/home/piktus/huggingface_rag/data/rag-sequence-nq"
    rag_sequence_test_config = RagConfig.from_pretrained(
        rag_sequence_model_path, dataset_name="dummy_psgs_w100_no_embeddings"
    )
    rag_model = RagTestSequenceModel(rag_sequence_test_config)
    generate_from_rag(rag_model, rag_tokenizer, questions, inputs, num_beams=4)

    # Top contexts
    """
    "Linda Davis" / Linda Davis Linda Kaye Davis (born November 26, 1962) is an American country music singer. Before beginning a career
    as a solo artist, she had three minor country singles in the charts as one half of the duo Skip & Linda. In her solo career, Davis has
    recorded five studio albums for major record labels and more than 15 singles. Her highest chart entry is "Does He Love You", her 1993 duet
    with Reba McEntire, which reached number one on the "Billboard" country charts and won both singers the Grammy for Best Country Vocal
    Collaboration. Her highest solo chart position
    """

    """
    "Isaac Newton" / author of the manuscript "De analysi per aequationes numero terminorum infinitas", sent by Isaac Barrow to John Collins
    in June 1669, was identified by Barrow in a letter sent to Collins in August of that year as "[...] of an extraordinary genius and proficiency
    in these things." Newton later became involved in a dispute with Leibniz over priority in the development of calculus (the Leibniz–Newton
    calculus controversy). Most modern historians believe that Newton and Leibniz developed calculus independently, although with very different
    mathematical notations. Occasionally it has been suggested that Newton published almost nothing about it until 1693, and did
    """

    """
    "Central nervous system"' / found that more than 95% of the 116 genes involved in the nervous system of planarians, which includes genes
    related to the CNS, also exist in humans. Like planarians, vertebrates have a distinct CNS and PNS, though more complex than those
    of planarians. In arthropods, the ventral nerve cord, the subesophageal ganglia and the supraesophageal ganglia are usually seen as making up
    the CNS. The CNS of chordates differs from that of other animals in being placed dorsally in the body, above the gut and notochord/spine.
    The basic pattern of the CNS is highly conserved throughout the different species of')
    """
