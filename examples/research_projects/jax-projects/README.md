# Flax/JAX community week 🤗

Welcome to the Flax/JAX community week! The goal of this week is to make compute-intensive NLP and CV projects (like pre-training BERT, GPT2, CLIP, ViT) 
practicable for a wider audience of engineers and researchers. 
To do so, we will try to teach **you** how to effectively use JAX/Flax on TPU and help you to complete a fun NLP and/or CV project in JAX/Flax during the community week. 

Free access to a TPUv3-8 will kindly be provided by the Google Cloud team!

In this document, we list all the important information that you will need during the Flax/JAX community week.

Don't forget to sign up [here](https://forms.gle/tVGPhjKXyEsSgUcs8)! 

## Table of Contents

- [Organization](#organization)
- [Important dates](#important-dates)
- [Communication](#communication)
- [Projects](#projects)
	- [How to propose](#how-to-propose-a-project)
	- [How to form a team](#how-to-form-a-team-around-a-project)
- [Tips & Tricks for project](#tips-on-how-to-organize-the-project)
- [How to install flax, jax, optax, transformers, datasets](#how-to-install-relevant-libraries)
- [Quickstart Flax/JAX](#quickstart-flax-and-jax)
- [Quickstart Flax/JAX in 🤗 Transformers](#quickstart-flax-and-jax-in-transformers)
    - [How to use flax models & scripts](#how-to-use-flax-models-and-example-scripts)
    - [Flax design philosophy in 🤗 Transformers](#flax-design-philosophy-in-transformers)
- [How to make a demo for submission](#how-to-make-a-demo)
- [Talks](#talks)
- [How to setup TPU VM](#how-to-setup-tpu-vm)
- [How to use the 🤗 Hub for training and demo](#how-to-use-the-hub-for-training-and-demo)
- [Project evaluation](#project-evaluation)
- [General Tips & Tricks](#general-tips-and-tricks)
- [FAQ](#faq)

## Organization

Participants can propose ideas for an interesting NLP and/or CV project. Teams of 3 to 5 will then be formed around the most promising and interesting projects. Make sure to read through the [Projects](#projects) section on how to propose projects, comment on other participants' project ideas, and create a team.

To help each team successfully finish their project, we have organized talks by leading scientists and engineers from Google, Hugging Face, and the open-source NLP & CV community. The talks will take place before the community week from June 30th to July 2nd. Make sure to attend the talks to get the most out of your participation! Check out the [Talks](#talks) section to get an overview of the talks, including the speaker and the time of the talk.

Each team is then given **free access to a TPUv3-8 VM** from July 7th to July 14th. In addition, we will provide training examples in JAX/Flax for a variety of NLP and Vision models to kick-start your project. During the week, we'll make sure to answer any questions you might have about JAX/Flax and Transformers and help each team as much as possible to complete their project!

At the end of the community week, each team should submit a demo of their project. All demonstrations will be evaluated by a jury and the top-3 demos will be awarded a prize. Check out the [How to submit a demo](#how-to-submit-a-demo) section for more information and suggestions on how to submit your project.

## Important dates

- **23.06.** Official announcement of the community week. Make sure to sign-up in [this google form](https://forms.gle/tVGPhjKXyEsSgUcs8).
- **23.06. - 30.06.** Participants will be added to an internal Slack channel. Project ideas can be proposed here and groups of 3-5 are formed. Read this document for more information. 
- **30.06.** Release of all relevant training scripts in JAX/Flax as well as other documents on how to set up a TPU, how to use the training scripts, how to submit a demo, tips & tricks for JAX/Flax, tips & tricks for efficient use of the   hub. 
- **30.06. - 2.07.** Talks about JAX/Flax, TPU, Transformers, Computer Vision & NLP will be held. 
- **7.07.** Start of the community week! Access to TPUv3-8 will be given to each team.
- **7.07. - 14.07.** The Hugging Face & JAX/Flax & Cloud team will be available for any questions, problems the teams might run into.
- **15.07.** Access to TPU is deactivated and community week officially ends.
- **16.07.** Deadline for each team to submit a demo. 

## Communication

All important communication will take place in an internal Slack channel, called `#flax-jax-community-week`. 
Important announcements of the Hugging Face, Flax/JAX, and Google Cloud team will be posted there. 
Such announcements include general information about the community week (Dates, Rules, ...), release of relevant training scripts (Flax/JAX example scripts for NLP and Vision), release of other important documents (How to access the TPU), etc. 
The Slack channel will also be the central place for participants to post about their results, share their learning experiences, ask questions, etc.

For issues with Flax/JAX, Transformers, Datasets or for questions that are specific to your project we would be **very happy** if you could use the following public repositories and forums:

- Flax: [Issues](https://github.com/google/flax/issues), [Questions](https://github.com/google/flax/discussions)
- JAX: [Issues](https://github.com/google/jax/issues), [Questions](https://github.com/google/jax/discussions)
- 🤗 Transformers: [Issues](https://github.com/huggingface/transformers/issues), [Questions](https://discuss.huggingface.co/c/transformers/9)
- 🤗 Datasets: [Issues](https://github.com/huggingface/datasets/issues), [Questions](https://discuss.huggingface.co/c/datasets/10)
- Project specific questions: [Forum](https://discuss.huggingface.co/c/flax-jax-projects/22)
- TPU related questions: [TODO]()

Please do **not** post the complete issue/project-specific question in the Slack channel, but instead a link to your issue/question that we will try to answer as soon as possible. 
This way, we make sure that the everybody in the community can benefit from your questions - even after the community week - and that the same question is not answered twice.

To be invited to the Slack channel, please make sure you have signed up [on the Google form](https://forms.gle/tVGPhjKXyEsSgUcs8). 

**Note**: If you have signed up on the google form, but you are not in the Slack channel, please leave a message on [(TODO) the official forum announcement]( ) and ping `@Suzana` and `@patrickvonplaten`.

## Projects

During the first week after the community week announcement, **23.06. - 30.06.**, teams will be formed around the most promising and interesting project ideas. Each team can consist of 2 to 10 participants. Projects can be accessed [here](https://discuss.huggingface.co/c/flax-jax-projects/22).

### How to propose a project

Some default project ideas are given by the organizers. **However, we strongly encourage participants to submit their own project ideas!**
Check out the [HOW_TO_PROPOSE_PROJECT.md](https://github.com/huggingface/transformers/tree/master/examples/research_projects/jax-projects/HOW_TO_PROPOSE_PROJECT.md) for more information on how to propose a new project.

### How to form a team around a project

You can check out all existing projects ideas on the forum under [Flax/JAX projects category](https://discuss.huggingface.co/c/flax-jax-projects/22).
Make sure to quickly check out each project idea and leave a ❤️  if you like an idea. 
Feel free to leave comments, suggestions for improvement, or questions about more details directly on the discussion thread. 
If you have found the project that you ❤️  the most, leave a message "I would like to join this project" on the discussion thread. 
We strongly advise you to also shortly state who you are, which time zone you are in and why you would like to work on this project, how you can contribute to the project and what your vision is for the project.
For projects that see a lot of interest and for which enough participants have expressed interest in joining, an official team will be created by the organizers. 
One of the organizers (`@Suzana`, `@valhalla`, `@osanseviero`, `@patrickvonplaten`) will leave a message "For this project the team: `<team_name>`, `<team_members>` , is officially created" on the thread and note down the teams on [this google sheet](https://docs.google.com/spreadsheets/d/1GpHebL7qrwJOc9olTpIPgjf8vOS0jNb6zR_B8x_Jtik/edit?usp=sharing).

Once created, the team can start refining their project:

- What is the goal of the project? *E.g.*, Present a language model that writes poetry in Russian.
- What model will we use? *E.g.*, FlaxGPT2
- What data will we use? *E.g.* Russian dataset of OSCAR & publicly available book on poetry
- Should we use a pre-trained model or train a model from scratch? E.g. Train a model from scratch
- What training scripts do we need? *E.g.* `transformers/examples/flax/run_clm_flax.py` can be used
- What kind of demo would we like to present? E.g. Text-generation API of the 🤗 Hub in combination with a Streamlit demo that lets the user generate a poem of a given length
- How will the work be divided? *E.g.* Team member 1 works on data preprocessing, Team member 2 works on adapting the Flax script, ...

We highly recommend that each team discusses all relevant ideas for their project directly on the forum thread. 
This way valuable learning experiences are shared and accessible by the whole community in the future. 
Additionally, the organizers, other participants, or anybody in the community really can read through your discussions and leave comments/tips for improvement. Obviously, you can also create private chats, ... to discuss more sensitive topics, etc.

**Important**:

- For project ideas that see a lot of interest, we are more than happy to create more than one team.
- Participants are welcome to join multiple teams, even though we encourage them to only work on a single project.
- Under special circumstances, participants can change/create new teams. Please note that we would like to keep this the exception. If however, you would like to change/leave existing teams, please leave a post on the project's thread where you ping the corresponding organizer that created the group.
 - It is often easy to propose/join a project that is done in your native language. Feel free to reach out to existing [language-specific groups](https://discuss.huggingface.co/c/languages-at-hugging-face/15) to look for community members that might be interested in joining your project.

## Tips on how to organize the project

This section gives you some tips on how to most efficiently & effectively 
work as a team to achieve your goal. It is by no means a strict recipe to follow, 
but rather a collection of tips from the 🤗 team.

Once your team is defined, you can start working on the project as soon as possible. 


### Communication

At first, it is always useful to get to know each other and to set up a means of communication.
While we recommend that all technical aspects of work can be discussed directly on the [forum](https://discuss.huggingface.co/c/flax-jax-projects/22) under your project thread, 
it can be very helpful to have a more direct way of communicating, *e.g.* in a channel. 
For this we have created a discord that you can access [here](https://discord.com/channels/858019234139602994/858019234139602997). 
This discord will not be managed by anybody and is just there so that you can communicate more effectively with your team members. 
Feel free to create a new channel for you and your team where you can discuss everything. If you and your team have already set up other ways of communicating, it is absolutely not required to make use of the discord. However, we do recommend each team to set up some kind 
of channel or group for quick discussions.

### Project definition

In the very beginning, you should make sure your project is well-defined and that 
everybody in the team understands the goal of the project and the work that needs to be 
done in order to achieve the goal. A well-defined project:

- has defined the task on which the model will be trained
- has defined the model that will be trained
- has defined the datasets that will be used for training
- has defined the type of training scripts that need to be written
- has defined the desired outcome of the project
- has defined the workflows

By "has defined" we don't meant that the corresponding code already has to be written and ready 
to be used, but that everybody in team is on the same page on what type of model, data and training script should be used.

To give an example, a well-defined project would be the following:

- task: summarization
- model: [t5-small](https://huggingface.co/t5-small)
- dataset: [CNN/Daily mail](https://huggingface.co/datasets/cnn_dailymail)
- training script: [run_summarization_flax.py](https://github.com/huggingface/transformers/blob/master/examples/flax/summarization/run_summarization_flax.py)
- outcome: t5 model that can summarize news
- work flow: adapt `run_summarization_flax.py` to work with `t5-small`.

This example is a very easy and not the most interesting project since a `t5-small`
summarization model exists already for CNN/Daily mail and pretty much no code has to be 
written. 
A well-defined project does not need to have the dataset be part of 
the `datasets` library and the training script already be pre-written, however it should 
be clear how the desired dataset can be accessed and how the training script can be 
written. 

It is also important to have a clear plan regarding the workflow. Usually, the 
data processing is done in a first step. Once the data is in a format that the model can 
work with, the training script can be written, etc. These steps should be more detailed 
once the team has a clearly defined project. It can be helpful to set deadlines for each step.

### Workload division

To effectively work as a team, it is crucial to divide the workload among everybody.
Some team members will be more motivated and experienced than others and 
some team members simply want to participate to learn more and cannot contribute that 
much to the team. This is totally fine! One cannot expect everybody in the team to have the same level of experience and time/motivation during the community week.

As a conclusion, being honest about one's expected involvement is crucial so that 
the workload can be divided accordingly. If someone doesn't think her/his tasks are feasible - let 
the team know early on so that someone else can take care of it!

It is recommended that the motivated and experienced team members take the lead in dividing the work and are ready to take over the tasks of another team member if necessary. 

The workload can often be divided according to:

- data preprocessing (load the data and preprocess data in the correct format)
- data tokenization / data collator (process data samples into tokens or images)
- model configuration (writing the code that defines the model)
- model forward pass (make sure input / output work correctly)
- loss function (define the loss function)
- putting the pieces together in a training script

Many of the steps above require other steps to be finished, so it often makes sense 
to use dummy data in the expected format to start, *e.g.*, with the model forward pass 
before the data preprocessing is done.

### Expectations

It is also very important to stay realistic with the scope of your project. Each team 
has access to a TPUv3-8 for only *ca.* 10 days, so it's important to keep the scope of 
the project reasonable. While we do want each team to work on interesting projects, each 
team should make sure that the project goals can be achieved within the provided compute 
time on TPU. For instance, pretraining a 11 billion parameters T5 model is not really a realistic 
task with just 10 days of TPUv3-8 compute. 
Also, it might be difficult to finish a project where the whole modeling, dataset and training code has to be written from scratch.

Having defined your project, feel free to reach out on Slack or the forum for feedback from the organizers. We can surely give you our opinion on whether the project is feasible and what can be done to improve it.
the project is feasible.

### Other tips

Here is a collection of some more tips:

- We strongly recommend to work as publicly and collaboratively as possible during the week so that other teams 
and the organizers can best help you. This includes publishing important discussions on 
the forum and making use of the [🤗 hub](http://huggingface.co/) to have a version 
control for your models and training logs.
- When debugging, it is important that the debugging cycle is kept as short as possible to 
be able to effectively debug. *E.g.* if there is a problem with your training script, 
you should run it with just a couple of hundreds of examples and not the whole dataset script. This can be done by either making use of [datasets streaming](https://huggingface.co/docs/datasets/master/dataset_streaming.html?highlight=streaming) or by selecting just the first 
X number of data samples after loading:

```python
datasets["train"] = datasets["train"].select(range(1000))
```
- Ask for help. If you are stuck, use the public Slack channel or the [forum](https://discuss.huggingface.co/c/flax-jax-projects/22) to ask for help.

## How to install relevant libraries

It is recommended to install all relevant libraries both on your local machine 
and on the TPU virtual machine. This way, quick prototyping and testing can be done on
your local machine and the actual training can be done on the TPU VM.

The following libraries are required to train a JAX/Flax model with 🤗 Transformers and 🤗 Datasets:

- [JAX](https://github.com/google/jax/)
- [Flax](https://github.com/google/flax)
- [Optax](https://github.com/deepmind/optax)
- [Transformers](https://github.com/huggingface/transformers)
- [Datasets](https://github.com/huggingface/datasets)

You should install the above libraries in a [virtual environment](https://docs.python.org/3/library/venv.html). 
If you're unfamiliar with Python virtual environments, check out the [user guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/). Create a virtual environment with the version of Python you're going
to use and activate it.

We strongly recommend to make use of the provided JAX/Flax examples scripts in [transformers/examples/flax](https://github.com/huggingface/transformers/tree/master/examples/flax) even if you want to train a JAX/Flax model of another github repository that is not integrated into 🤗 Transformers.
In all likelihood, you will need to adapt one of the example scripts, so we recommend forking and cloning the 🤗 Transformers repository as follows. 
Doing so will allow you to share your fork of the Transformers library with your team members so that the team effectively works on the same code base. It will also automatically install the newest versions of `flax`, `jax` and `optax`.

**IMPORTANT**: If you are setting up your environment on a TPU VM, make sure to 
install JAX's TPU version before cloning and installing the transformers repository. 
Otherwise, an incorrect version of JAX will be installed, and the following commands will 
throw an error. 
To install JAX's TPU version simply run the following command:

```
$ pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

To verify that JAX was correctly installed, you can run the following command:

```python
import jax
jax.device_count()
```

This should display the number of TPU cores, which should be 8 on a TPUv3-8 VM.

Now you can run the following steps as usual.

1. Fork the [repository](https://github.com/huggingface/transformers) by
   clicking on the 'Fork' button on the repository's page. This creates a copy of the code
   under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote:

   ```bash
   $ git clone https://github.com/<your Github handle>/transformers.git
   $ cd transformers
   $ git remote add upstream https://github.com/huggingface/transformers.git
   ```

3. Create a new branch to hold your development changes. This is especially useful to share code changes with your team:

   ```bash
   $ git checkout -b a-descriptive-name-for-my-project
   ```

4. Set up a flax environment by running the following command in a virtual environment:

   ```bash
   $ pip install -e ".[flax]"
   ```

   (If transformers was already installed in the virtual environment, remove
   it with `pip uninstall transformers` before reinstalling it in editable
   mode with the `-e` flag.)

   If you have already cloned that repo, you might need to `git pull` to get the most recent changes in the `datasets`
   library.

   Running this command will automatically install `flax`, `jax` and `optax`.

Next, you should also install the 🤗 Datasets library. We strongly recommend installing the 
library from source to profit from the most current additions during the community week.

Simply run the following steps:

```
$ cd ~/
$ git clone https://github.com/huggingface/datasets.git
$ cd datasets
$ pip install -e ".[streaming]"
```

If you plan on contributing a specific dataset during 
the community week, please fork the datasets repository and follow the instructions 
[here](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-create-a-pull-request).

To verify that all libraries are correctly installed, you can run the following command.
It assumes that both `transformers` and `datasets` were installed from master - otherwise
datasets streaming will not work correctly.

```python
from transformers import FlaxRobertaModel, RobertaTokenizerFast
from datasets import load_dataset
import jax

dataset = load_dataset('oscar', "unshuffled_deduplicated_en", split='train', streaming=True)

dummy_input = next(iter(dataset))["text"]

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
input_ids = tokenizer(dummy_input, return_tensors="np").input_ids[:, :10]

model = FlaxRobertaModel.from_pretrained("julien-c/dummy-unknown")

# run a forward pass, should return an object `FlaxBaseModelOutputWithPooling`
model(input_ids)
```


## Quickstart flax and jax

[JAX](https://jax.readthedocs.io/en/latest/index.html) is Autograd and XLA, brought together for high-performance numerical computing and machine learning research. It provides composable transformations of Python+NumPy programs: differentiate, vectorize, parallelize, Just-In-Time compile to GPU/TPU, and more. A great place for getting started with JAX is the [JAX 101 Tutorial](https://jax.readthedocs.io/en/latest/jax-101/index.html).

[Flax](https://flax.readthedocs.io/en/latest/index.html) is a high-performance neural network library designed for flexibility built on top of JAX. It aims to provide users with full control of their training code and is carefully designed to work well with JAX transformations such as `grad` and `pmap` (see the [Flax philosophy](https://flax.readthedocs.io/en/latest/philosophy.html)). For an introduction to Flax see the [Flax Basics Colab](https://flax.readthedocs.io/en/latest/notebooks/flax_basics.html) or the list of curated [Flax examples](https://flax.readthedocs.io/en/latest/examples.html).

## Quickstart flax and jax in transformers

Currently, we support the following models in Flax. 
Note that some models are about to be merged to `master` and will 
be available in a couple of days.

- [BART](https://github.com/huggingface/transformers/blob/master/src/transformers/models/bart/modeling_flax_bart.py)
- [BERT](https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_flax_bert.py)
- [BigBird](https://github.com/huggingface/transformers/blob/master/src/transformers/models/big_bird/modeling_flax_big_bird.py)
- [CLIP](https://github.com/huggingface/transformers/blob/master/src/transformers/models/clip/modeling_flax_clip.py)
- [ELECTRA](https://github.com/huggingface/transformers/blob/master/src/transformers/models/electra/modeling_flax_electra.py)
- [GPT2](https://github.com/huggingface/transformers/blob/master/src/transformers/models/gpt2/modeling_flax_gpt2.py)
- [(TODO) MBART](https://github.com/huggingface/transformers/blob/master/src/transformers/models/mbart/modeling_flax_mbart.py)
- [RoBERTa](https://github.com/huggingface/transformers/blob/master/src/transformers/models/roberta/modeling_flax_roberta.py)
- [T5](https://github.com/huggingface/transformers/blob/master/src/transformers/models/t5/modeling_flax_t5.py)
- [ViT](https://github.com/huggingface/transformers/blob/master/src/transformers/models/vit/modeling_flax_vit.py)
- [(TODO) Wav2Vec2](https://github.com/huggingface/transformers/blob/master/src/transformers/models/wav2vec2/modeling_flax_wav2vec2.py)

You can find all available training scripts for JAX/Flax under the 
official [flax example folder](https://github.com/huggingface/transformers/tree/master/examples/flax). Note that a couple of training scripts will be released in the following week.

- [Causal language modeling (GPT2)](https://github.com/huggingface/transformers/blob/master/examples/flax/language-modeling/run_clm_flax.py)
- [Masked language modeling (BERT, RoBERTa, ELECTRA, BigBird)](https://github.com/huggingface/transformers/blob/master/examples/flax/language-modeling/run_mlm_flax.py)
- [Text classification (BERT, RoBERTa, ELECTRA, BigBird)](https://github.com/huggingface/transformers/blob/master/examples/flax/text-classification/run_flax_glue.py)
- [Summarization / Seq2Seq (BART, MBART, T5)](https://github.com/huggingface/transformers/blob/master/examples/flax/summarization/run_summarization_flax.py)
- [(TODO) Masked Seq2Seq pret-training (T5)]( )
- [(TODO) Image classification (ViT)]( )
- [(TODO) CLIP pretraining, fine-tuning (CLIP)]( )


### How to use flax models and example scripts

TODO (should be filled by 29.06.)

### Flax design philosophy in transformers

TODO (should be filled by 29.06.)

## How to make a demo

TODO (should be filled by 30.06.)...

## Talks

TODO (should be filled by 29.06.)...

## How to setup TPU VM

TODO (should be filled by 2.07.)...

## How to use the hub for training and demo
 
TODO (should be filled by 1.07.)...

## Project evaluation

TODO (should be filled by 5.07.)...

## General tips and tricks

TODO (will be filled continuously)...

## FAQ

TODO (will be filled continuously)...
