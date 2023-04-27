import datasets
import faiss
import numpy as np
import streamlit as st
import torch
from elasticsearch import Elasticsearch
from eli5_utils import (
    embed_questions_for_retrieval,
    make_qa_s2s_model,
    qa_s2s_generate,
    query_es_index,
    query_qa_dense_index,
)

import transformers
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_TYPE = "bart"
LOAD_DENSE_INDEX = True


@st.cache(allow_output_mutation=True)
def load_models():
    if LOAD_DENSE_INDEX:
        qar_tokenizer = AutoTokenizer.from_pretrained("yjernite/retribert-base-uncased")
        qar_model = AutoModel.from_pretrained("yjernite/retribert-base-uncased").to("cuda:0")
        _ = qar_model.eval()
    else:
        qar_tokenizer, qar_model = (None, None)
    if MODEL_TYPE == "bart":
        s2s_tokenizer = AutoTokenizer.from_pretrained("yjernite/bart_eli5")
        s2s_model = AutoModelForSeq2SeqLM.from_pretrained("yjernite/bart_eli5").to("cuda:0")
        save_dict = torch.load("seq2seq_models/eli5_bart_model_blm_2.pth")
        s2s_model.load_state_dict(save_dict["model"])
        _ = s2s_model.eval()
    else:
        s2s_tokenizer, s2s_model = make_qa_s2s_model(
            model_name="t5-small", from_file="seq2seq_models/eli5_t5_model_1024_4.pth", device="cuda:0"
        )
    return (qar_tokenizer, qar_model, s2s_tokenizer, s2s_model)


@st.cache(allow_output_mutation=True)
def load_indexes():
    if LOAD_DENSE_INDEX:
        faiss_res = faiss.StandardGpuResources()
        wiki40b_passages = datasets.load_dataset(path="wiki_snippets", name="wiki40b_en_100_0")["train"]
        wiki40b_passage_reps = np.memmap(
            "wiki40b_passages_reps_32_l-8_h-768_b-512-512.dat",
            dtype="float32",
            mode="r",
            shape=(wiki40b_passages.num_rows, 128),
        )
        wiki40b_index_flat = faiss.IndexFlatIP(128)
        wiki40b_gpu_index_flat = faiss.index_cpu_to_gpu(faiss_res, 1, wiki40b_index_flat)
        wiki40b_gpu_index_flat.add(wiki40b_passage_reps)  # TODO fix for larger GPU
    else:
        wiki40b_passages, wiki40b_gpu_index_flat = (None, None)
    es_client = Elasticsearch([{"host": "localhost", "port": "9200"}])
    return (wiki40b_passages, wiki40b_gpu_index_flat, es_client)


@st.cache(allow_output_mutation=True)
def load_train_data():
    eli5 = datasets.load_dataset("eli5", name="LFQA_reddit")
    eli5_train = eli5["train_eli5"]
    eli5_train_q_reps = np.memmap(
        "eli5_questions_reps.dat", dtype="float32", mode="r", shape=(eli5_train.num_rows, 128)
    )
    eli5_train_q_index = faiss.IndexFlatIP(128)
    eli5_train_q_index.add(eli5_train_q_reps)
    return (eli5_train, eli5_train_q_index)


passages, gpu_dense_index, es_client = load_indexes()
qar_tokenizer, qar_model, s2s_tokenizer, s2s_model = load_models()
eli5_train, eli5_train_q_index = load_train_data()


def find_nearest_training(question, n_results=10):
    q_rep = embed_questions_for_retrieval([question], qar_tokenizer, qar_model)
    D, I = eli5_train_q_index.search(q_rep, n_results)
    nn_examples = [eli5_train[int(i)] for i in I[0]]
    return nn_examples


def make_support(question, source="wiki40b", method="dense", n_results=10):
    if source == "none":
        support_doc, hit_lst = (" <P> ".join(["" for _ in range(11)]).strip(), [])
    else:
        if method == "dense":
            support_doc, hit_lst = query_qa_dense_index(
                question, qar_model, qar_tokenizer, passages, gpu_dense_index, n_results
            )
        else:
            support_doc, hit_lst = query_es_index(
                question,
                es_client,
                index_name="english_wiki40b_snippets_100w",
                n_results=n_results,
            )
    support_list = [
        (res["article_title"], res["section_title"].strip(), res["score"], res["passage_text"]) for res in hit_lst
    ]
    question_doc = "question: {} context: {}".format(question, support_doc)
    return question_doc, support_list


@st.cache(
    hash_funcs={
        torch.Tensor: (lambda _: None),
        transformers.models.bart.tokenization_bart.BartTokenizer: (lambda _: None),
    }
)
def answer_question(
    question_doc, s2s_model, s2s_tokenizer, min_len=64, max_len=256, sampling=False, n_beams=2, top_p=0.95, temp=0.8
):
    with torch.no_grad():
        answer = qa_s2s_generate(
            question_doc,
            s2s_model,
            s2s_tokenizer,
            num_answers=1,
            num_beams=n_beams,
            min_len=min_len,
            max_len=max_len,
            do_sample=sampling,
            temp=temp,
            top_p=top_p,
            top_k=None,
            max_input_length=1024,
            device="cuda:0",
        )[0]
    return (answer, support_list)


st.title("Long Form Question Answering with ELI5")

# Start sidebar
header_html = "<img src='https://huggingface.co/front/assets/huggingface_logo.svg'>"
header_full = """
<html>
  <head>
    <style>
      .img-container {
        padding-left: 90px;
        padding-right: 90px;
        padding-top: 50px;
        padding-bottom: 50px;
        background-color: #f0f3f9;
      }
    </style>
  </head>
  <body>
    <span class="img-container"> <!-- Inline parent element -->
      %s
    </span>
  </body>
</html>
""" % (
    header_html,
)
st.sidebar.markdown(
    header_full,
    unsafe_allow_html=True,
)

# Long Form QA with ELI5 and Wikipedia
description = """
This demo presents a model trained to [provide long-form answers to open-domain questions](https://yjernite.github.io/lfqa.html).
First, a document retriever fetches a set of relevant Wikipedia passages given the question from the [Wiki40b](https://research.google/pubs/pub49029/) dataset,
a pre-processed fixed snapshot of Wikipedia.
"""
st.sidebar.markdown(description, unsafe_allow_html=True)

action_list = [
    "Answer the question",
    "View the retrieved document only",
    "View the most similar ELI5 question and answer",
    "Show me everything, please!",
]
demo_options = st.sidebar.checkbox("Demo options")
if demo_options:
    action_st = st.sidebar.selectbox(
        "",
        action_list,
        index=3,
    )
    action = action_list.index(action_st)
    show_type = st.sidebar.selectbox(
        "",
        ["Show full text of passages", "Show passage section titles"],
        index=0,
    )
    show_passages = show_type == "Show full text of passages"
else:
    action = 3
    show_passages = True

retrieval_options = st.sidebar.checkbox("Retrieval options")
if retrieval_options:
    retriever_info = """
    ### Information retriever options

    The **sparse** retriever uses ElasticSearch, while the **dense** retriever uses max-inner-product search between a question and passage embedding
    trained using the [ELI5](https://arxiv.org/abs/1907.09190) questions-answer pairs.
    The answer is then generated by sequence to sequence model which takes the question and retrieved document as input.
    """
    st.sidebar.markdown(retriever_info)
    wiki_source = st.sidebar.selectbox("Which Wikipedia format should the model use?", ["wiki40b", "none"])
    index_type = st.sidebar.selectbox("Which Wikipedia indexer should the model use?", ["dense", "sparse", "mixed"])
else:
    wiki_source = "wiki40b"
    index_type = "dense"

sampled = "beam"
n_beams = 2
min_len = 64
max_len = 256
top_p = None
temp = None
generate_options = st.sidebar.checkbox("Generation options")
if generate_options:
    generate_info = """
    ### Answer generation options

    The sequence-to-sequence model was initialized with [BART](https://huggingface.co/facebook/bart-large)
    weights and fine-tuned on the ELI5 QA pairs and retrieved documents. You can use the model for greedy decoding with
    **beam** search, or **sample** from the decoder's output probabilities.
    """
    st.sidebar.markdown(generate_info)
    sampled = st.sidebar.selectbox("Would you like to use beam search or sample an answer?", ["beam", "sampled"])
    min_len = st.sidebar.slider(
        "Minimum generation length", min_value=8, max_value=256, value=64, step=8, format=None, key=None
    )
    max_len = st.sidebar.slider(
        "Maximum generation length", min_value=64, max_value=512, value=256, step=16, format=None, key=None
    )
    if sampled == "beam":
        n_beams = st.sidebar.slider("Beam size", min_value=1, max_value=8, value=2, step=None, format=None, key=None)
    else:
        top_p = st.sidebar.slider(
            "Nucleus sampling p", min_value=0.1, max_value=1.0, value=0.95, step=0.01, format=None, key=None
        )
        temp = st.sidebar.slider(
            "Temperature", min_value=0.1, max_value=1.0, value=0.7, step=0.01, format=None, key=None
        )
        n_beams = None

# start main text
questions_list = [
    "<MY QUESTION>",
    "How do people make chocolate?",
    "Why do we get a fever when we are sick?",
    "How can different animals perceive different colors?",
    "What is natural language processing?",
    "What's the best way to treat a sunburn?",
    "What exactly are vitamins ?",
    "How does nuclear energy provide electricity?",
    "What's the difference between viruses and bacteria?",
    "Why are flutes classified as woodwinds when most of them are made out of metal ?",
    "Why do people like drinking coffee even though it tastes so bad?",
    "What happens when wine ages? How does it make the wine taste better?",
    "If an animal is an herbivore, where does it get the protein that it needs to survive if it only eats grass?",
    "How can we set a date to the beginning or end of an artistic period? Doesn't the change happen gradually?",
    "How does New Zealand have so many large bird predators?",
]
question_s = st.selectbox(
    "What would you like to ask? ---- select <MY QUESTION> to enter a new query",
    questions_list,
    index=1,
)
if question_s == "<MY QUESTION>":
    question = st.text_input("Enter your question here:", "")
else:
    question = question_s

if st.button("Show me!"):
    if action in [0, 1, 3]:
        if index_type == "mixed":
            _, support_list_dense = make_support(question, source=wiki_source, method="dense", n_results=10)
            _, support_list_sparse = make_support(question, source=wiki_source, method="sparse", n_results=10)
            support_list = []
            for res_d, res_s in zip(support_list_dense, support_list_sparse):
                if tuple(res_d) not in support_list:
                    support_list += [tuple(res_d)]
                if tuple(res_s) not in support_list:
                    support_list += [tuple(res_s)]
            support_list = support_list[:10]
            question_doc = "<P> " + " <P> ".join([res[-1] for res in support_list])
        else:
            question_doc, support_list = make_support(question, source=wiki_source, method=index_type, n_results=10)
    if action in [0, 3]:
        answer, support_list = answer_question(
            question_doc,
            s2s_model,
            s2s_tokenizer,
            min_len=min_len,
            max_len=int(max_len),
            sampling=(sampled == "sampled"),
            n_beams=n_beams,
            top_p=top_p,
            temp=temp,
        )
        st.markdown("### The model generated answer is:")
        st.write(answer)
    if action in [0, 1, 3] and wiki_source != "none":
        st.markdown("--- \n ### The model is drawing information from the following Wikipedia passages:")
        for i, res in enumerate(support_list):
            wiki_url = "https://en.wikipedia.org/wiki/{}".format(res[0].replace(" ", "_"))
            sec_titles = res[1].strip()
            if sec_titles == "":
                sections = "[{}]({})".format(res[0], wiki_url)
            else:
                sec_list = sec_titles.split(" & ")
                sections = " & ".join(
                    ["[{}]({}#{})".format(sec.strip(), wiki_url, sec.strip().replace(" ", "_")) for sec in sec_list]
                )
            st.markdown(
                "{0:02d} - **Article**: {1:<18} <br>  _Section_: {2}".format(i + 1, res[0], sections),
                unsafe_allow_html=True,
            )
            if show_passages:
                st.write(
                    '> <span style="font-family:arial; font-size:10pt;">' + res[-1] + "</span>", unsafe_allow_html=True
                )
    if action in [2, 3]:
        nn_train_list = find_nearest_training(question)
        train_exple = nn_train_list[0]
        st.markdown(
            "--- \n ### The most similar question in the ELI5 training set was: \n\n {}".format(train_exple["title"])
        )
        answers_st = [
            "{}. {}".format(i + 1, "  \n".join([line.strip() for line in ans.split("\n") if line.strip() != ""]))
            for i, (ans, sc) in enumerate(zip(train_exple["answers"]["text"], train_exple["answers"]["score"]))
            if i == 0 or sc > 2
        ]
        st.markdown("##### Its answers were: \n\n {}".format("\n".join(answers_st)))


disclaimer = """
---

**Disclaimer**

*The intent of this app is to provide some (hopefully entertaining) insights into the behavior of a current LFQA system.
Evaluating biases of such a model and ensuring factual generations are still very much open research problems.
Therefore, until some significant progress is achieved, we caution against using the generated answers for practical purposes.*
"""
st.sidebar.markdown(disclaimer, unsafe_allow_html=True)
