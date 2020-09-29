import random
from time import sleep, time

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch

import altair as alt
import transformers
from transformers import RagConfig, RagRetriever, RagTokenForGeneration, RagTokenizer


st.beta_set_page_config(
    page_title="RAG demo by Hugging Face and FB AI Research",
    page_icon="https://huggingface.co/front/assets/huggingface_logo.svg",
    layout="wide",
    initial_sidebar_state="auto",
)

colors = [
    "#332288",
    "#117733",
    "#882255",
    "#AA4499",
    "#CC6677",
    "#44AA99",
    "#DDCC77",
    "#88CCEE",
]

plt.rc("axes", titlesize=16)  # fontsize of the axes title
plt.rc("axes", labelsize=16)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=12)  # fontsize of the tick labels
plt.rc("ytick", labelsize=12)  # fontsize of the tick labels


@st.cache(
    allow_output_mutation=True,
    hash_funcs={
        torch.Tensor: lambda x: None,
        transformers.tokenization_rag.RagTokenizer: lambda x: None,
        transformers.modeling_rag.RagTokenForGeneration: lambda x: None,
        transformers.retrieval_rag.RagRetriever: lambda x: None,
    },
)
def load_models():
    st_time = time()
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
    rag_conf = RagConfig.from_pretrained("facebook/rag-token-nq")
    rag_conf.index_name = "legacy"
    rag_conf.index_path = "/home/yacine/Data/rag_index"
    print("+++++ loading retriever", time() - st_time)
    retriever = RagRetriever(rag_conf, tokenizer.question_encoder, tokenizer.generator)
    print("+++++ loading model", time() - st_time)
    nq_model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)
    nq_model.rag.config.n_docs = 8
    nq_model.rag.retriever.config.n_docs = 8
    _ = nq_model.to("cuda:0")
    _ = nq_model.eval()
    print("+++++ loaded model nq", time() - st_time)
    jp_model = RagTokenForGeneration.from_pretrained("/home/yacine/Code/nq_jeopardy_model_v2", retriever=retriever)
    jp_model.rag.config.n_docs = 8
    jp_model.rag.retriever.config.n_docs = 8
    _ = jp_model.to("cuda:1")
    _ = jp_model.eval()
    print("+++++ loaded model jp", time() - st_time)
    return (tokenizer, retriever, nq_model, jp_model)


st_time = time()
tokenizer, retriever, nq_model, jp_model = load_models()

print("+++++ reading answer question function", time() - st_time)


@st.cache(
    allow_output_mutation=True,
    hash_funcs={
        torch.Tensor: lambda x: None,
        transformers.tokenization_rag.RagTokenizer: lambda x: None,
        transformers.modeling_rag.RagTokenForGeneration: lambda x: None,
        transformers.retrieval_rag.RagRetriever: lambda x: None,
    },
)
def answer_query(
    query,
    query_type="nq",
    num_beams=4,
    min_length=2,
    max_length=64,
):
    print("+++++ answering query", query_type, query)
    sleep(random.random() * 0.1)
    device = "cuda:0" if query_type == "nq" else "cuda:1"
    model = nq_model if query_type == "nq" else jp_model
    input_ids = tokenizer(query, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        # retrieve support docs
        retrieved_outputs = model(input_ids, labels=None, output_retrieved=True)
        dl_scores = retrieved_outputs.doc_scores[0].tolist()
        dp_scores = retrieved_outputs.doc_scores.softmax(dim=-1)[0].tolist()
        doc_dicts = retriever.index.get_doc_dicts(retrieved_outputs.retrieved_doc_ids)[0]
        support_docs = [
            {"score": ls, "proba": ns, "title": ti, "text": te}
            for ls, ns, ti, te in zip(dl_scores, dp_scores, doc_dicts["title"], doc_dicts["text"])
        ]
        # generate answers
        generated_ids = model.generate(
            input_ids=input_ids,
            context_input_ids=retrieved_outputs.context_input_ids,
            context_attention_mask=retrieved_outputs.context_attention_mask,
            doc_scores=retrieved_outputs.doc_scores,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            min_length=min_length,
            max_length=max_length,
            length_penalty=1.0,
        )
        answer_texts = [
            tokenizer.generator.decode(gen_seq.tolist(), skip_special_tokens=True).strip() for gen_seq in generated_ids
        ]
        # explain answers
        explanations = []
        answer_scores = []
        for a_i in range(generated_ids.shape[0]):
            answer_ids = generated_ids[a_i : a_i + 1]
            eval_out = model(
                input_ids=input_ids,
                labels=answer_ids,
                context_input_ids=retrieved_outputs.context_input_ids,
                context_attention_mask=retrieved_outputs.context_attention_mask,
                doc_scores=retrieved_outputs.doc_scores,
            )
            doc_probs = retrieved_outputs.doc_scores.softmax(dim=-1).view(-1).tolist()
            eval_probs = eval_out.logits.softmax(dim=-1)
            answer_score = eval_out.loss.item()
            answer_scores += [answer_score]
            explained_gen = []
            ans_id_list = answer_ids[0].tolist()
            if tokenizer.generator.eos_token_id in ans_id_list[1:]:
                s_len = ans_id_list[1:].index(tokenizer.generator.eos_token_id)
            else:
                s_len = len(ans_id_list) - 1
            for i in range(1, s_len):
                tid = ans_id_list[i + 1]
                token = tokenizer.generator._convert_id_to_token(tid).replace("Ġ", "_")
                step_probs = eval_probs[:, i, tid].tolist()
                explained_gen += [(token, [(di, p, p * doc_probs[di]) for di, p in enumerate(step_probs)])]
            explanations += [explained_gen]
    return {
        "answers": [(te, -sc) for te, sc in zip(answer_texts, answer_scores)],
        "documents": support_docs,
        "explanations": explanations,
    }


@st.cache(allow_output_mutation=True)
def join_explanation(explained, vis_id):
    token_exp = [(w, sorted(scores, key=lambda x: x[vis_id], reverse=True)[0][0]) for w, scores in explained]
    tokens = [w for w, scores in explained]
    token_start = [0] + [i for i, w in enumerate(tokens) if w.startswith("_")] + [len(token_exp)]
    token_joined = [
        ("".join(tokens[token_start[i] : token_start[i + 1]]).replace("_", ""), token_exp[token_start[i]][1])
        for i in range(len(token_start) - 1)
    ]
    return token_joined


print("----------------- READY")

st.title("Retrieval Augmented Generation")

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

# RAG
description = """
This demo showcases the [Hugging Face implementation](https://huggingface.co/transformers/master/model_doc/rag.html) of the [RAG model](https://arxiv.org/abs/2005.11401) (for Retrieval Augmented Generation).
When presented with a query, the model fetches paragraphs from Wikipedia which contain helpful facts and draws information from them to produce an output.
In addition to providing this output, the present demo helps visualize how these supporting paragraphs are used at each step in the generation.
"""
st.sidebar.markdown(description, unsafe_allow_html=True)

action_list = [
    "Question answering (NaturalQA)",
    "Question generation (Jeopardy!)",
]

action_st = st.sidebar.selectbox(
    "Choose a model setting:",
    action_list,
    index=0,
)
action_ids = ["nq", "jp"]
action = action_ids[action_list.index(action_st)]

n_beams = 4
min_len = 2
max_len = 64

generate_info = """
### Answer generation options
The sequence-to-sequence model uses the [BART](https://huggingface.co/facebook/bart-large)
architecture. You can increase the beam size to get more answers from the model.
"""
generate_options = st.sidebar.checkbox("Generation options")
if generate_options:
    st.sidebar.markdown(generate_info)
    min_len = st.sidebar.slider(
        "Minimum generation length", min_value=0, max_value=32, value=2, step=2, format=None, key=None
    )
    max_len = st.sidebar.slider(
        "Maximum generation length", min_value=8, max_value=128, value=64, step=8, format=None, key=None
    )
    n_beams = st.sidebar.slider("Beam size", min_value=2, max_value=8, value=4, step=2, format=None, key=None)

vis_choice = "weighted"
# visualization_options = st.sidebar.checkbox("Visualization options")
visualization_options = False
if visualization_options:
    vis_choice = st.sidebar.selectbox(
        "Choose whether to weigh token-level provenance by the document score:",
        ["weighted", "un-weighted"],
        index=0,
    )

vis_id = 2 if vis_choice == "weighted" else 1

disclaimer = """
---
**Disclaimer**
*The intent of this app is to provide some insights into the behavior of the RAG system.
Evaluating biases of such a model and ensuring factual generations are still very much open research problems.
Therefore, until some significant progress is achieved, we caution against using the generated answers for practical purposes.*
"""
st.sidebar.markdown(disclaimer, unsafe_allow_html=True)

# Full screen question and answers
st.markdown("---  ")
left_a, right_a = st.columns(2)

# Left: query selection
if action == "nq":
    left_a.markdown("### Ask a factoid question:")
    left_a.markdown(
        "> The model was trained on the [Natural Questions](https://huggingface.co/datasets/natural_questions) dataset to answer factoid questions using information from Wikipedia. You can find some of the training examples [here](https://ai.google.com/research/NaturalQuestions/visualization):"
    )
    questions_list = ["<FREE INPUT>"] + [
        "Who wrote Sister Outsider?",
        "Which book is Ursula Le Guin best know for?",
        "What happened in France in 1789?",
        "What do bears eat?",
        "Which animals are most closely related to dinosaurs?",
    ]
    default_query = "where is blood pumped after it leaves the right ventricle?"
    question_s = left_a.selectbox(
        "What would you like to ask? ---- select <FREE INPUT> to enter a new query",
        questions_list,
        index=1,
    )
elif action == "jp":
    left_a.markdown("### Give a target answer:")
    left_a.markdown(
        "The model was trained to create Jeopardy!-style questions for a given answer. You can find the training dataset and have a look at some examples [here](https://huggingface.co/nlp/viewer/?dataset=jeopardy):"
    )
    questions_list = ["<FREE INPUT>", "Toussaint Louverture", "aurora borealis", "the Lusitania", "zucchini", "foil"]
    default_query = "fencing"
    question_s = left_a.selectbox(
        "Which answer would you like to generate a question for? ---- select <FREE INPUT> to enter a new query",
        questions_list,
        index=1,
    )

if question_s == "<FREE INPUT>":
    question = left_a.text_input("Enter your query here:", default_query)
else:
    question = question_s

# Answer question
model_outputs = answer_query(
    question,
    query_type=action,
    num_beams=n_beams,
    min_length=min_len,
    max_length=max_len,
)
right_a.markdown("### The model generated the following answers:")
right_a.markdown(
    f"> The generator model's beam search allows us to consider several possible outputs for a given query. Here are the top {n_beams} outputs with their generation scores:"
)
answer_md = "- *Model outputs:*\n"
for i, (answer, loss) in enumerate(sorted(model_outputs["answers"], key=lambda x: x[1], reverse=True)):
    answer_md += f"  {i+1}. **{answer}** *({loss:.2f})*\n"

right_a.markdown(answer_md)

# Split into left and right
left_c, right_c = st.columns(2)

col_a, col_b, col_c, col_d = st.columns(4)

# Visualize answer-level behavior
left_c.markdown("--- \n ### Example-level contribution of Wikipedia passages:")
left_c.markdown(
    "> When presented with a query, the model retrieves a set of support documents using a dot product, and weighs each of them using the retrieval score when generating the answer. Scroll down to show the full text of the retrieved passages:"
)

for i, doc in enumerate(model_outputs["documents"][:4]):
    wiki_url = f"https://en.wikipedia.org/wiki/{doc['title'].replace(' ', '_')}"
    title_short = doc["title"] if len(doc["title"]) < 26 else doc["title"][:24] + "..."
    title_url = f"[{title_short}]({wiki_url})"
    col_a.markdown(
        f"<font color={colors[i]}>&#11044; -- Article {i+1} --</font> {title_url}",
        unsafe_allow_html=True,
    )
    df_s = pd.DataFrame({f"d:{i+1}": [f"{i+1}"], f"p({i+1}|q)": [doc["proba"]], "color": colors[i]})
    alt_bar = (
        alt.Chart(df_s)
        .mark_bar()
        .encode(
            x=alt.X(f"p({i+1}|q)", scale=alt.Scale(domain=[0, 0.6]), axis=None),
            y=alt.Y(f"d:{i+1}", axis=None),
            color=alt.Color("color", scale=None),
        )
    )
    alt_text = alt_bar.mark_text(
        align="left",
        baseline="middle",
        fontSize=16,
        fontWeight="bold",
        dx=5,  # Nudges text to right so it doesn't appear on top of the bar
    ).encode(text=alt.Text(f"p({i+1}|q)", format=",.3f"))
    chart = (
        alt.layer(alt_bar, alt_text).configure_axis(grid=False).configure_view(strokeOpacity=0).properties(width=300)
    )
    col_a.write(chart)


for j, doc in enumerate(model_outputs["documents"][4:]):
    i = j + 4
    wiki_url = f"https://en.wikipedia.org/wiki/{doc['title'].replace(' ', '_')}"
    title_short = doc["title"] if len(doc["title"]) < 26 else doc["title"][:24] + "..."
    title_url = f"[{title_short}]({wiki_url})"
    col_b.markdown(
        f"<font color={colors[i]}>&#11044; -- Article {i+1} --</font> {title_url}",
        unsafe_allow_html=True,
    )
    df_s = pd.DataFrame({f"d:{i+1}": [f"{i+1}"], f"p({i+1}|q)": [doc["proba"]], "color": colors[i]})
    alt_bar = (
        alt.Chart(df_s)
        .mark_bar()
        .encode(
            x=alt.X(f"p({i+1}|q)", scale=alt.Scale(domain=[0, 0.6]), axis=None),
            y=alt.Y(f"d:{i+1}", axis=None),
            color=alt.Color("color", scale=None),
        )
    )
    alt_text = alt_bar.mark_text(
        align="left",
        baseline="middle",
        fontSize=16,
        fontWeight="bold",
        dx=5,  # Nudges text to right so it doesn't appear on top of the bar
    ).encode(text=alt.Text(f"p({i+1}|q)", format=",.3f"))
    chart = (
        alt.layer(alt_bar, alt_text).configure_axis(grid=False).configure_view(strokeOpacity=0).properties(width=300)
    )
    col_b.write(chart)


# Visualize token-level behavior
right_c.markdown("--- \n ### Word-level contribution of Wikipedia passages:")
right_c.markdown(
    "> When generating the output, the contribution of each passage to the next token depends both on its retrieval score and on the tokens that have been generated previously. Select a generated output to show token-level provenance information:"
)

answers = [a for a, l in model_outputs["answers"]]
losses = [l for a, l in model_outputs["answers"]]
answer_s = col_c.selectbox(
    "",
    answers,
    index=losses.index(max(losses)),
)

explained = model_outputs["explanations"][answers.index(answer_s)]
joined_exp = join_explanation(explained, 2)

# word-level provenance
joined_an = tuple([(w, str(did + 1), colors[did]) for w, did in joined_exp])
col_c.markdown(
    "- "
    + " ".join([f"<font color={colors[did]}>**{w}**-{did+1}-</font>" for w, did in joined_exp if len(w) > 0])
    + " \n",
    unsafe_allow_html=True,
)

# token-level plot
plot_container = col_d.container()

id_start = col_d.slider(
    "Show token-level passage contributions starting from:",
    min_value=0,
    max_value=max(len(explained), 1),
    value=0,
    step=1,
    format=None,
    key=None,
)

for r_i in range(3):
    if (id_start + 3 * r_i) < len(explained):
        row_charts = []
        for c_i in range(3):
            tok_id = id_start + 3 * r_i + c_i
            w, scores = explained[tok_id] if tok_id < len(explained) else (str(tok_id), [(0, 0, 0)] * 8)
            w = w.replace("'", "`").replace('"', "``")
            df_exp = pd.DataFrame(
                {
                    w: [f"{i}" for i in range(8, 0, -1)],
                    "Proba": [scores[i - 1][vis_id] for i in range(8, 0, -1)],
                    "color": [colors[i - 1] for i in range(8, 0, -1)],
                }
            )
            alt_bar = (
                alt.Chart(df_exp)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "Proba",
                        scale=alt.Scale(domain=[0, 1]),
                        axis=None if tok_id >= len(explained) else alt.Axis(title=w, titleFontSize=12),
                    ),
                    y=alt.Y(w, axis=None if tok_id >= len(explained) else alt.Axis(title="")),
                    color=alt.Color("color", scale=None),
                )
                .properties(width=80, height=80)
            )
            row_charts += [alt_bar]
        alt_row = alt.HConcatChart(hconcat=row_charts).configure_axis(grid=False).configure_view(strokeOpacity=0)
        plot_container.write(alt_row)


# Show retrieved passages:
st.markdown("---  ")

show_list = st.multiselect("Show full text for articles:", [str(i + 1) for i in range(8)])

for i, doc in enumerate(model_outputs["documents"]):
    if str(i + 1) in show_list:
        wiki_url = f"https://en.wikipedia.org/wiki/{doc['title'].replace(' ', '_')}"
        title_url = f"[{doc['title']:128.128}]({wiki_url})"
        st.markdown(
            f"<font color={colors[i]}>&#11044; **-- Article {i+1}:**</font> {title_url}",
            unsafe_allow_html=True,
        )
        st.write(
            '> <span style="font-family:arial; font-size:10pt;">' + doc["text"] + "</span>", unsafe_allow_html=True
        )
