import torch

from transformers import BartTokenizer, RagSequenceModel, RagTokenModel, T5ForConditionalGeneration, T5Tokenizer


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def generate_from_rag(rag_model, tokenizer, questions, inputs, num_beams=4):
    with torch.no_grad():
        rag_model = rag_model.eval().to(device)
        outputs = rag_model.generate(
            inputs,
            num_beams=num_beams,
            min_length=1,  # make sure short answers are allowed
            max_length=25,  # no need for crazy long answers in NQ
            # early_stopping=True,
            num_return_sequences=num_beams,
            use_cache=True,
            # use_cache=False, - doesn't work for bart
            # repetition_penalty=10,
            # bad_words_ids=[[0, 0]],
            # BART likes to repeat BOS tokens, dont allow it to generate more than one
        )
        answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for i in range(0, len(questions)):
            print("Question: " + questions[i])
            print("Top {} answers:".format(num_beams))
            for answer in answers[i * num_beams : (i + 1) * num_beams]:
                print("\t", answer)


questions = [
    "who sings does he love me with reba",
    "who were the two mathematicians that invented calculus",
    "what parts make up the peripheral nervous system",
]

if __name__ == "__main__":

    rag_sequence_model_path = "/private/home/piktus/rag_huggingface/data/repro-rag-sequence-99"

    bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    inputs = bart_tokenizer.batch_encode_plus(questions, return_tensors="pt", padding=True, truncation=True)[
        "input_ids"
    ].to(device)

    print(inputs)
    print("\nSEQUENCE MODEL - HF retriever")
    model_path = "/checkpoint/piktus/2020-08-14/rag_seq_hf.rag_sequence.batch_size8.ls0.1.dr0.1.atdr0.1.wd0.001.eps1e-08.clip0.1.lr3e-05.num_epochs100.warm500.ngpu8/checkpoint35/"
    rag_model = RagSequenceModel.from_pretrained(
        rag_sequence_model_path, retriever_type="hf_retriever", uncompressed=False
    )
    generate_from_rag(rag_model, bart_tokenizer, questions, inputs, num_beams=4)

    print("\nTOKEN MODEL - HF retriever")
    rag_model = RagTokenModel.from_pretrained(
        rag_sequence_model_path, retriever_type="hf_retriever", uncompressed=False
    )
    generate_from_rag(rag_model, bart_tokenizer, questions, inputs, num_beams=4)

    print("\SEQUENCE MODEL WITH T5 GENERATOR")
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-large")
    questions = [
        "generate answer: who sings does he love me with reba" + t5_tokenizer.eos_token,
        "generate answer: who sings does he love me with reba",
    ]

    t5_inputs = t5_tokenizer.batch_encode_plus(questions, return_tensors="pt", padding=True, truncation=True)[
        "input_ids"
    ].to(device)
    rag_t5_model = RagSequenceModel.from_pretrained(
        "/private/home/piktus/rag_huggingface/data/test_training_t5_large/", retriever_type="hf_retriever",
    )

    prefix = "generate answer: "
    rag_t5_model.model.generator.config.prefix = prefix

    generate_from_rag(rag_t5_model, t5_tokenizer, questions, t5_inputs, num_beams=1)

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
