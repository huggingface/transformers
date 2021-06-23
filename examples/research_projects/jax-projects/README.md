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
- [Quickstart Flax/JAX](#quickstart-flax-and-jax)
- [Quickstart Flax/JAX in 🤗 Transformers](#quickstart-flax-and-jax-in-transformers)
- [How to install flax, jax, optax, transformers, datasets](#how-to-install-relevant-libraries)
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

During the first week after the community week announcement, **23.06. - 30.06.**, teams will be formed around the most promising and interesting project ideas. Each team can consist of 2 to 5 participants. Projects can be accessed [here](https://discuss.huggingface.co/c/flax-jax-projects/22).

### How to propose a project

Some default project ideas are given by the organizers. **However, we strongly encourage participants to submit their own project ideas!**
Check out the [(TODO) HOW_TO_PROPOSE_PROJECT.md]( ) for more information on how to propose a new project.

### How to form a team around a project

You can check out all existing projects ideas on the forum under [Flax/JAX projects category](https://discuss.huggingface.co/c/flax-jax-projects/22).
Make sure to quickly check out each project idea and leave a ❤️  if you like an idea. 
Feel free to leave comments, suggestions for improvement, or questions about more details directly on the discussion thread. 
If you have found the project that you ❤️  the most, leave a message "I would like to join this project" on the discussion thread. 
We strongly advise you to also shortly state who you are, which time zone you are in and why you would like to work on this project, how you can contribute to the project and what your vision is for the project.
For projects that see a lot of interest and for which enough participants have expressed interest in joining, an official team will be created by the organizers. 
One of the organizers (`@Suzana`, `@valhalla`, `@osanseviero`, `@patrickvonplaten`) will leave a message "For this project the team: `<team_name>`, `<team_members>` , is officially created" on the thread and note down the teams on [(TODO) this google sheet]().

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

TODO (should be filled by 24.06.)...

## Quickstart flax and jax

TODO (should be filled by 25.06.)...

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
- [(TODO) T5](https://github.com/huggingface/transformers/blob/master/src/transformers/models/t5/modeling_flax_t5.py)
- [ViT](https://github.com/huggingface/transformers/blob/master/src/transformers/models/vit/modeling_flax_vit.py)
- [(TODO) Wav2Vec2](https://github.com/huggingface/transformers/blob/master/src/transformers/models/wav2vec2/modeling_flax_wav2vec2.py)

You can find all available training scripts for JAX/Flax under the 
official [flax example folder](https://github.com/huggingface/transformers/tree/master/examples/flax). Note that a couple of training scripts will be released in the following week.

- [Causal language modeling (GPT2)](https://github.com/huggingface/transformers/blob/master/examples/flax/language-modeling/run_clm_flax.py)
- [Masked language modeling (BERT, RoBERTa, ELECTRA, BigBird)](https://github.com/huggingface/transformers/blob/master/examples/flax/language-modeling/run_mlm_flax.py)
- [Text classification (BERT, RoBERTa, ELECTRA, BigBird)](https://github.com/huggingface/transformers/blob/master/examples/flax/text-classification/run_flax_glue.py)
- [(TODO) Summarization / Seq2Seq (BART, MBART, T5)]( )
- [(TODO) Masked Seq2Seq pret-training (T5)]( )
- [(TODO) Image classification (ViT)]( )
- [(TODO) CLIP pretraining, fine-tuning (CLIP)]( )

For more in-detail information on how to use/adapt Transformers Flax models and 
example scripts, please have a look at [(TODO by 25.06.) HOW_TO_USE_FLAX_IN_TRANSFORMERS]( ).

## How to install relevant libraries

TODO (should be filled by 25.06.) ...

## How to make a demo

TODO (should be filled by 28.06.)...

## Talks

TODO (should be filled by 28.06.)...

## How to setup TPU VM

TODO (should be filled by 2.07.)...

## How to use the hub for training and demo
 
TODO (should be filled by 2.07.)...

## Project evaluation

TODO (should be filled by 5.07.)...

## General tips and tricks

TODO (will be filled continuously)...

## FAQ

TODO (will be filled continuously)...
