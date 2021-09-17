# How to propose a Flax/JAX + Transformers project 

Great that you've opened this document! 
While we at 🤗 are proposing a couple of projects, we strongly 
believe that the community can come up with much more **creative**, **fun**, and 
**impactful** projects on their own. This being said, we are really looking forward
to seeing your project proposal! 

## What a project should be about

The proposed project should fall into the machine learning fields of **Natural Language Processing (NLP)** and/or **Computer Vision (CV)** (possibly also **Speech Recognition (ASR)** depending on whether Speech Recognition models are available in Flax in due time) and aim at solving a specific task. 
Possible tasks can belong to: 

 * text classification
 * text generation
 * image recognition
 * image processing
 * image captioning
 * audio classification
 * and other tasks you can think of!

The clearer a task is defined, the better your project proposal is.
*E.g.* "Using a T5 model to learn grammar correction in French" or "Adapting a pre-trained CLIP model for zero-shot image classification in Spanish" are **well-defined and clear** project proposals, while something like "Train a language model" or "Image classification" are **too vague**.

There is no limit to your creativity as long as the project is feasible and ethical.
The more creative & specific your project proposal, the more interesting it will be, 
and the more likely will you find motivated team members to work on your project!
To get an idea of how to formulate your project proposals, you can browse through 
existing project proposals on the [forum](https://discuss.huggingface.co/c/flax-jax-projects/22).

## How to submit a project proposal

First, you should make sure that you are [logged in](https://huggingface.co/login?sso=bm9uY2U9OTRlNjZjZmZhYjMwMmJmMWMyYjc5MmFiMTMyMzY5ODYmcmV0dXJuX3Nzb191cmw9aHR0cHMlM0ElMkYlMkZkaXNjdXNzLmh1Z2dpbmdmYWNlLmNvJTJGc2Vzc2lvbiUyRnNzb19sb2dpbg%3D%3D&sig=429ad8924bcb33c40f9823027ea749abb55d393f4f58924f36a2dba3ab0a48da) with your Hugging Face account on the forum. 

Second, make sure that your project idea doesn't already exist by checking [existing projects](https://discuss.huggingface.co/c/flax-jax-projects/22). 
If your project already exists - great! This means that you can comment and improve
the existing idea and join the project to form a team! If your project idea already 
exists for a different language, feel free to submit the same project idea, just in 
a different language.

Third, having ensured that your project doesn't exist, click on the *"New Topic"*
button on the [Flax/JAX Projects Forum category](https://discuss.huggingface.co/c/flax-jax-projects/22) to create a new project proposal.

Fourth, make sure that your project proposal includes the following information:

1. *A clear description of the project*
2. *In which language should the project be conducted?* English, German, Chinese, ...? It can also be a multi-lingual project
3. *Which model should be used?* If you want to adapt an existing model, you can add the link to one of the 4000 available checkpoints in JAX [here](https://huggingface.co/models?filter=jax) If you want to train a model from scratch, you can simply state the model architecture to be used, *e.g.* BERT, CLIP, etc. You can also base your project on a model that is not part of transformers. For an overview of libraries based on JAX, you can take a look at [awesome-jax](https://github.com/n2cholas/awesome-jax#awesome-jax-). **Note** that for a project that is not based on Transformers it will be more difficult for the 🤗 team to help you. Also have a look at the section [Quickstart Flax & Jax in Transformers](https://github.com/huggingface/transformers/tree/master/examples/research_projects/jax-projects#quickstart-flax-and-jax-in-transformers) to see what model architectures are currently supported in 🤗 Transformers.
4. *What data should be used?* It is important to state at least what kind of data you would like to use. Ideally, you can already point to publicly available data or a dataset in the 🤗 Datasets library.
5. *Are similar training scripts available in Flax/JAX?* It would be important to find similar training scripts that already exist in Flax/JAX. *E.g.* if you are working on a Seq-to-Seq task, you can make use of the [`run_summarization_flax.py`](https://github.com/huggingface/transformers/blob/master/examples/flax/summarization/run_summarization_flax.py) script which is very similar to any seq2seq training. Also have a look at the section [Quickstart Flax & Jax in Transformers](https://github.com/huggingface/transformers/tree/master/examples/research_projects/jax-projects#quickstart-flax-and-jax-in-transformers) to see what training scripts are currently supported in 🤗 Transformers.
6. *(Optionally) What are possible challenges?* List possible difficulties with your project. *E.g.* If you know that training convergence usually takes a lot of time, it is worth stating this here!
7. *(Optionally) What is the desired project outcome?* - How would you like to demo your project? One could *e.g.* create a Streamlit application.
8. *(Optionally) Links to read upon* - Can you provide any links that would help the reader to better understand your project idea?

Feel free to copy-paste the following format for your project proposal and fill out the respective sections: 

```
# <FILL ME: Name of project>

<FILL ME: A clear description of the project>

## 2. Language

The model will be trained in <FILL ME: which language?>.

## 3. Model

<FILL ME: 3. Which model should be used?>

## 4. Datasets

<FILL ME: 4. Which data should be used?>

Possible links to publicly available datasets include:
- <FILL ME: Link 1 to dataset> 
- <FILL ME: Link 2 to dataset> 
- <FILL ME: Link 3 to dataset> 

## 5. Training scripts

<FILL ME: 5. Are there publicly available training scripts that can be used/tweaked for the project?>

We can make use of <FILL ME: link to training script> to train the model.>

## 6. (Optional) Challenges

<(Optionally) FILL ME: 6. What are possible challenges?>

## 7. (Optional) Desired project outcome

<(Optionally) FILL ME: 7. What is the desired project outcome? A demo?>

## 8. (Optional) Reads

The following links can be useful to better understand the project and 
what has previously been done.

- <FILL ME: Link 1 to read> 
- <FILL ME: Link 2 to read> 
- <FILL ME: Link 3 to read> 
```

To see how a proposed project looks like, please have a look at submitted project 
proposals [here](https://discuss.huggingface.co/c/flax-jax-projects/22).

## Will my project proposal be selected?

Having submitted a project proposal, you can now promote your idea in the Slack channel `#flax-jax-community-week` to try to convince other participants to join your project! 
Once other people have joined your project, one of the organizers (`@Suzana, @valhalla, @osanseviero, @patrickvonplaten`) will officially create a team for your project and add your project to [this google sheet](https://docs.google.com/spreadsheets/d/1GpHebL7qrwJOc9olTpIPgjf8vOS0jNb6zR_B8x_Jtik/edit?usp=sharing).
