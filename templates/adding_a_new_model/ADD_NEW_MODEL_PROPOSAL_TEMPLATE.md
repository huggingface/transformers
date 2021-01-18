# *TEMPLATE*

*search & replace the following keywords:*

-[name of model]

-[name of teacher]

-[start date]

-[end date]

-[camelcase name of mode]

# How to add [name of model] to 🤗Transformers
	
Teacher: [name of hugging face teacher]
	
Begin: [start date]
	
Estimated time: [end date]

The following sections explain in detail how to add [name of model] 
to 🤗Transformers. You will work closely with [name of hugging face teacher] to
integrate [name of model] into 🤗Transformers. By doing so, you will both gain a 
theoretical and deep practical understanding of [name of model]. But more importantly, 
you will have made a major open-source contribution to 🤗Transformers. Along the way,
you will:
	
- get insights into open-source best practices, 
- understand the design principles of one of the most popular NLP libraries,
- learn how to do efficiently test large NLP models,
- learn how to integrate python utilities like `black`, `isort`, `make fix-copies` into a library 
  to always ensure clean and readable code.
  
The following list is a summary of everything that has to be done to add a model and can be used 
by you as a To-Do List:

- [ ] Understood theoretical aspects
- [ ] Prepared environment
- [ ] Set up debugging environment of original repository
- [ ] Created script that successfully runs forward pass using original repository and checkpoint
- [ ] Successfully converted original checkpoint to Transformers checkpoint
- [ ] Successfully ran forward pass in Transformers that gives identical output to original checkpoint
- [ ] Finished model tests in Transformers
- [ ] Succesfully added Tokenizer in Transformers
- [ ] Finished docs
- [ ] Uploaded model weights to the hub
- [ ] Merged the pull request
- [ ] (Optionally) added a notebook

To begin with, you should start by getting a good understanding of the model.
  
## Theoretical aspects of [name of model]
  
### Paper

  You should take some time to read [[name of model]'s paper]([link to paper]).
  There might be large sections of the paper that are difficult to understand.
  If this is the case, this is fine - don't worry! The goal is not to get 
  a deep theoretical understanding of the paper, but to extract the necessary information 
  required to effectively re-implement the model to 🤗Transformers.
  That being said, you don't have to spend too much time on the theoretical aspects,
  but rather focus on the practical ones, namely:
  
  - What time of model is [name of model]? BERT-like encoder-only model? GPT2-like decoder-only model? BART-like encoder-decoder model?
  - What are the applications of [name of model]? Text classification? Text generation? Seq2Seq tasks, *e.g.* summarization?
  - What is the novel feature of the model making it different from BERT or BART (if it's an encoder-decoder model)?
  - Which of the already existing [🤗Transformers models](https://huggingface.co/transformers/#contents) is most similar to [name of model]?
  - What type of tokenizer is used? A sentence piece tokenizer? Word piece tokenizer? Is it the same tokenizer as used for BERT or BART?
  
 After you feel like you have gotten a good overview of the architecture of the model, you might want 
 to write [name of hugging face teacher] for any questions you might have.
 This might include questions regarding the model's architecture, its attention layer, etc. [name of hugging face teacher] will be more 
 than happy to help you.
 
### Additional resources

 Before diving into the code, here are some additional resources that might be worth taking a look at:
 
 - [link 1]
 - [link 2]
 - [link 3]
 - ...
 
### Make sure you've understood the fundamental aspects of [name of model]
 
Alright, now you should be ready to take a closer look into the actual code of [name of model].
You should have understood the following aspects of [name of model] by now:

- [characteristic 1 of [name of model]]
- [characteristic 2 of [name of model]]
- ...

If any of the mentioned aspects above are **not** clear to you, now is a great time to talk to [name of hugging face teacher] again!

## Next prepare your environment

1. Fork the [repository](https://github.com/huggingface/transformers) by clicking on the 'Fork' button on the repository's page.
This creates a copy of the code under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote:

	```bash
	git clone https://github.com/[your Github handle]/transformers.git
	cd transformers
	git remote add upstream https://github.com/huggingface/transformers.git
	```
 
3. Set up a development environment, for instance by running the following command:

	```bash
	conda create -n env python=3.7 --y
	conda activate env
	pip install -e ".[dev]"
	```
  
  and return to the parent directory
  
  ```bash
  cd ..
  ```

4. We recommend to add the PyTorch version of [name of model] to Transformers. To install PyTorch,
  please follow the instructions on https://pytorch.org/. 
  
  **Note:** You don't need to have CUDA install. It is sufficient to just be working on CPU.

5. To port [name of model], you will also need access to its [original repository]([link to original repo]):
  
  ```bash
  git clone [clone link to original repo]
  cd [name of repo]
  pip install -e .
  ```

Now you have set up a development environment to port [name of model] to 🤗Transformers.
  
## Run a pretrained checkpoint using the original repository

At first, you will work on the original repository. Often, the original implementation is very
 "researchy" meaning that documentation might be lacking and the code can be difficult to understand.
But this should be exactly your motivation to reimplement [name of the model]. At Hugging Face, one of our main goals is to
*make people stand on the shoulders of giants* which translates here very well into taking a working 
model and rewriting it to make it as **accessible, user-friendly, and beautiful** as possible.
This is the #1 motivation to re-implement models into 🤗Transformers - trying to maximize access 
to a complex new NLP technology for **everybody**.
	
You should start thereby by diving into the original repository.

### Get familiar with the original repository.

Successfully running the official pre-trained model in the original repository is often 
**the most difficult** step. From our experience, it is very important to spend some time to get familiar with the [original codebase]([link to original repo]). You should find out

- Where to find the pre-trained weights
- How to load the pre-trained weights into its corresponding model
- Trace one forward pass so that you know which classes and functions are required for a simple forward pass
  . Usually, you only have to reimplement those functions.
- Be able to locate the important components of the model: Where is the model class? Are there submodel 
  classes, *e.g.* EncoderModel, DecoderModel? Where is the self-attention layer? 
  Are there multiple different attention layers, *e.g.* *self-attention*, *cross-attention*...?
- How can you debug the model in the original environment of the repo? Do you have to set `print` statements
  or can you work with an interactive debugger like `ipdb`?
  
It is very important that before you start opening a PR in 🤗Transformers that you can **efficiently** 
debug code in the original repository! This means that you should be able to run a forward pass and print out the actual values of the output of a layer. *I.e* you can load a pre-trained model, pass an input vector of 
token ids, *i.e.* `input_ids = [0, 1, 4, 5, ...]` to the model's forward function and you can print out the 
intermediate outputs of - let's say - the first self-attention layer that could look something like this: 

```bash
[[
 [-0.1465, -0.6501,  0.1993,  ...,  0.1451,  0.3430,  0.6024],
 [-0.4417, -0.5920,  0.3450,  ..., -0.3062,  0.6182,  0.7132],
 [-0.5009, -0.7122,  0.4548,  ..., -0.3662,  0.6091,  0.7648],
 ...,
 [-0.5613, -0.6332,  0.4324,  ..., -0.3792,  0.7372,  0.9288],
 [-0.5416, -0.6345,  0.4180,  ..., -0.3564,  0.6992,  0.9191],
 [-0.5334, -0.6403,  0.4271,  ..., -0.3339,  0.6533,  0.8694]]],
```

This means that your debugging environment should consists of a short script (ideally written by you) that 
does the following (in pseudocode):

```bash
model = [name of model]Model.load_pretrained_checkpoint(/path/to/checkpoint/)
input_ids = ... # vector of input ids
original_output = model.predict(input_ids)
```

By running such a script, you should be able to print out intermediate values or hit a breakpoint
in the clone of the original repository that is saved locally on your computer.

We expect that every model added to 🤗Transformers passes a couple of integration tests, meaning that the original 
model and the reimplemented version in 🤗Transformers have to give the exact same output up to a precision of 0.001! 
It is not enough if the model gives nearly the same output, they have to be the same. Therefore, you will certainly compare the intermediate outputs of the 🤗Transformers version multiple times against the intermediate outputs of the original implementation of [name of model] in which case an **efficient** debugging environment of the original repository is absolute key. Here is some advice is to make your debugging environment as efficient as possible.
	
- Find the best way of debugging intermediate results. Is the original repository written in PyTorch? Then you should be able to use a simple debugger like [ipdb](https://pypi.org/project/ipdb/) to print out intermediate values. Is the original repository written in Tensorflow 1? Then you might have to rely on TensorFlow print operations like 
  https://www.tensorflow.org/api_docs/python/tf/print to output intermediate values. Is the original repository written in Jax? Then make sure that the model is **not jitted** when running the forward pass, 
  *e.g.* check-out [this link](https://github.com/google/jax/issues/196).
- Use the smallest pre-trained checkpoint you can find. The smaller the checkpoint, the faster your debug cycle becomes. It is not efficient if your pre-trained model is so big that your forward pass takes more than 10 seconds. In case only very large checkpoints 
  are available, it might make more sense to create a dummy model in the new environment with randomly initialized weights 
  and save those weights for comparison with the 🤗Transformers version of your model
- Make sure you are using the easiest way of calling a forward pass in the original repository. Ideally, you want to find the function
  in the original repository that **only** calls a single forward pass, *i.e.* that is often called `predict`, `evaluate`, `forward` or `__call__`. 
  You don't want to debug a function that calls `forward` multiple times, *e.g.* to generate text, like `autoregressive_sample`, `generate`.
- Try to separate the tokenization from the model's `forward` pass. If the original repository shows examples where you have to input a string, then try to find out where in the forward call the string input is changed to input ids and start from this point. This might mean that you have to possibly write a small script yourself or change the original code so that you can directly input the ids instead of an input string.
- Make sure that the model in your debugging setup is **not** in training mode, which often causes the model to yield random outputs due to multiple dropout layers in the model. Make sure that the forward pass in your debugging environment is **deterministic** so that the dropout layers are not used.
	
The following section gives you more specific details/tips on how you can do this for [name of model].
	
### More details on how to create a debugging environment for [name of model] 
	
[Here the teacher should add very specific information on what the student should do]
[to set up an efficient environment for the special requirements of this model]

## Implement [name of model] into 🤗Transformers
	
Next, you can finally add code to 🤗Transformers. Go into the clone 
of your 🤗Transformers' fork:

```
cd transformers
```

### Use the Cookiecutter to automatically generate the model's code

To begin with head over to the [🤗Transformers templates](https://github.com/huggingface/transformers/tree/master/templates/adding_a_new_model) to make
use of our `cookiecutter` implementation to automatically generate all the relevant files for your model. Again, we recommend only adding the PyTorch version of the model at first.
Make sure you follow the instructions of the `README.md` on the [🤗Transformers templates](https://github.com/huggingface/transformers/tree/master/templates/adding_a_new_model) carefully.

### Open a Pull Request on the main `huggingface/transformers` repo

Before starting to adapt the automatically generated code, now is the time to open a "Work in progress (WIP)" pull request, *e.g.* "[WIP] Add [name of model]",
in 🤗Transformers so that you and [name of teacher] can work side-by-side on integrating the model into 🤗Transformers.

You should do the following:

1. Commit the automatically generated code:

```
git add .
git commit
```

2. Fetch and rebase to current master

```
git fetch upstream
git rebase upstream/master
```

3. Push the changes to your account using:

```
git push -u origin a-descriptive-name-for-my-changes
```

4. Once you are satisfied, go to the webpage of your fork on GitHub. Click on "Pull request". Make sure to add the GitHub handle of [name of teacher] as 
reviewer, so that [name of teacher] gets notified for future changes.

5. Change the PR into a draft by clicking on "Convert to draft" on the right of the GitHub pull request web page.

In the following, whenever you have done some progress, don't forget to commit your work and push it to your account so that is shows in the pull request. 
Additionally, you should make sure to update your work with the current master from time to time by doing:

```
git fetch upstream
git merge upstream/master
```

In general, all questions you might have regarding the model or your implementation should be asked in your PR and discussed/solved in the PR.
This way, [name of teacher] will always be notified when you are committing new code or if you have a question. It is often very helpful to point [name of teacher] to your added code so that [name of teacher] can efficiently understand your problem or question.

To do so, you can go to the "Files changed" tab where you see all of your changes, go to a line regarding which you want to ask a question and click on the "+" symbol to add a comment. Whenever a question or problem has been solved, you can click on the "Resolve" button of the created comment.

In the same way, [name of teacher] will open comments when reviewing your code. We recommend asking most questions on GitHub on your PR. For some very general
questions that are not very useful for the public, feel free to ping [name of teacher] by Slack or mail.

### Adapt the generated model's code for [name of model]
	
At first, we will focus only on the model itself and not care about the tokenizer. All the relevant code should be found in 
the generated files `src/transformers/models/[lowercase name of model]/modeling_[lowercase name of model].py`
and `src/transformers/models/[lowercase name of model]/configuration_[lowercase name of model].py`.

Now you can finally start coding :). The generated code in `src/transformers/models/[lowercase name of model]/modeling_[lowercase name of model].py` will
either has the same architecture as BERT or BART if it's an encoder-decoder model.
At this point, you should remind yourself what you've learned in the beginning about the theoretical aspects of the model: *How is the model different from BERT or BART?*". Implement those changes which often means to change the *self-attention* layer, the order of the normalization layer, etc...
Here it is often useful to look at the similar architecture of already existing models in Transformers.

**Note** that at this point, you don't have to be very sure that your code is fully correct or clean.
Rather, it is advised to add a first *unclean*, copy-pasted version of the original code to 
`src/transformers/models/[lowercase name of model]/modeling_[lowercase name of model].py` until you feel like all the 
necessary code is added. From our experience, it is much more efficient to quickly add a first version of the required code
and improve/correct the code iteratively with the conversion script as described in the next section. The only thing that has to work at this 
point is that you can instantiate the 🤗Transformers implementation of [name of model], *i.e.* the following command works:
	
```python 
from transformers import [camelcase name of model]Model, [camelcase name of model]Config

model = [camelcase name of model]Model([camelcase name of model]Config())
```

The above command will create a model according to the default parameters as defined in `[camelcase name of model]Config()`
with random weights, thus making sure that the `init()` methods of all components works.

In the case of [name of model], you should at least have to do the following changes:
	
[Here the teacher should add very specific information on what exactly has to be changed for this model]
[...]
[...]

### Write a conversion script

Now, you should write a conversion script that lets you convert the checkpoint you used to debug 
[name of model] in the original repository to your just created 🤗Transformers implementation of [name of model].
Here you should not try to write the conversion script from scratch, but find similar models in 🤗Transformers
that require similar conversion scripts, *i.e.* whose original repository was written with the same framework as
[name of model].
	
In the conversion script, you should make sure that you set the parameters correctly in `[camelcase name of model]Config()`
to exactly match those that were used for the checkpoint you want to convert. Also, it is very important that before
you set the retrieved weights to the new weight, *e.g.* via

```python
layer.weight.data = array
```

you make sure that both their **shape and name match**. If either the shape or the name doesn't match, you probably assigned the wrong 
checkpoint weight to a randomly initialized layer of the 🤗Transformers implementation. Therefore, it is ** necessary** to
add assert statements for the shape and print out the names of the checkpoints weights. E.g. you should add statements like:

```python
assert (
	pointer_random_weight.shape == checkpoint_weight.shape
), f"Pointer shape of random weight {pointer_random_weight.shape} and array shape of checkpoint weight {checkpoint_weight.shape} mismatched
```

and 

```python
logger.info(f"Initialize PyTorch weight {pointer_random_weight.name} from {checkpoint_weight.name}")
```

for verification. Besides, you should also check that **all** required weights are initialized and print
out all checkpoint weights that were not used for initialization to make sure the model is correctly converted.
It is completely normal, that the conversion trials fail with either a wrong shape statement or wrong name assignment.
This is most likely because either you used incorrect parameters in `[camelcase name of model]Config()`, have a wrong architecture 
in the 🤗Transformers implementation, you have a bug in the `init()` functions of one of the components of 
the 🤗Transformers implementation or you need to transpose one of the checkpoint weights.

This step should be iterated with the previous step until all weights of the checkpoint are correctly loaded in the Transformers model.
Having correctly loaded the checkpoint into the 🤗Transformers implementation, you can then save the model under a folder of your choice `/path/to/converted/checkpoint/folder` that should include both a `pytorch_model.bin` file and a `config.json` file.


In the case of [name of model], you should probably do the following:
	
[Here the teacher should add very specific information on what exactly has to be done for the conversion of this model]
[...]
[...]
	
### Implement the forward pass

Having managed to correctly load the pretrained weights into the 🤗Transformers implementation, you should now make sure that the forward pass is correctly implemented. In *Get familiar with the original repository.*, you have already created a script
that runs a forward pass of the model using the original repository. Now you should write an analogous script using the 
🤗Transformers implementation instead of the original one. It should look as follows:

[Here the model name might have to be adapted, *e.g.* maybe [name of model]ForConditionalGeneration instead of [name of model]Model]

```python
model = [name of model]Model.from_pretrained(/path/to/converted/checkpoint/folder)
input_ids = ... # the exact same vector of input ids in PyTorch as those used in the *Get familiar with the original repository.* section
output = model(input_ids).last_hidden_states
```

It is very likely that the 🤗Transformers implementation and the original model implementation don't give the exact same output the very first time or that the forward pass throws an error. Don't be disappointed - it's expected! First, you should make sure that the forward pass doesn't throw any errors. It often happens that the wrong dimensions are used leading to a `Dimensionality mismatch` error or that the wrong data type object is used,
*e.g.* `torch.long` instead of `torch.float32`. Don't hesitate to ask [name of teacher] for help, if you don't manage to solve certain errors.

The final part to make sure the 🤗Transformers implementation works correctly is to ensure that the outputs are equivalent to a precision of `1e-3`.
First, you should ensure that the output shapes are identical, *i.e.* `outputs.shape` should yield the same value for the script of the 🤗Transformers implementation and the original implementation. Next, you should make sure that the output values are identical as well. This one of the most difficult parts of adding a new model. Common mistakes why the outputs are not identical are:

- Some layers were not added, *i.e.* an `activation` layer was not added, or the residual connection was forgotten
- The word embedding matrix was not tied
- The wrong positional embeddings are used because the original implementation uses on offset
- Dropout is applied during the forward pass. To fix this make sure `model.training is False` and that no dropout layer is falsely activated during the 
  forward pass, *i.e.* pass `self.training` to [PyTorch's functional dropout](https://pytorch.org/docs/stable/nn.functional.html?highlight=dropout#torch.nn.functional.dropout)
  
The best way to fix the problem is usually to look at the forward pass of the original implementation and the 🤗Transformers implementation side-by-side
and check if there are any differences. Ideally, you should debug/print out intermediate outputs of both implementations of the forward pass to find the exact position in the network where the 🤗Transformers implementation shows a different output than the original implementation. First, make sure that the hard-coded
`input_ids` in both scripts are identical. Next, verify that the outputs of the first transformation of the `input_ids` (usually the word embeddings)
are identical. And then work your way up to the very last layer of the network. At some point, you will notice a difference between the two implementations, which
should point you to the bug in the 🤗Transformers implementation. From our experience, a simple and efficient way is to add many print statements in both the original implementation and 🤗Transformers implementation, at the same positions in the network respectively, and to successively remove print statements showing
the same values for intermediate presentions.

When you're confident that both implementations yield the same output, verifying the outputs with `torch.allclose(original_output, output, atol=1e-3)`, you're done with the most difficult part! Congratulations - the work left to be done should be a cakewalk 😊.
  
### Adding all necessary model tests

At this point, you have successfully added a new model. However, it is very much possible that the model does not yet fully comply with the required design. To make sure, the implementation is fully compatible with 🤗Transformers, all common tests should pass. The Cookiecutter should have automatically added a test file for your model, probably under the same `tests/test_modeling_[name of model].py`. Run this test file to verify that all common tests pass:

```python
pytest tests/test_modeling_[name of model].py
```

[Here the teacher should add very specific information on what tests are likely to fail after having implemented the model
, e.g. given the model, it might be very likely that `test_attention_output` fails]
[...]
[...]

Having fixed all common tests, it is now crucial to ensure that all the nice work you have done is well tested, so that 

- a) The community can easily understand your work by looking at specific tests of [name of model]
- b) Future changes to your model will not break any important feature of the model.

At first, integration tests should be added. Those integration tests essentially do the same as the debugging scripts you used earlier to implement the model to 🤗Transformers. A template of those model tests is already added by the Cookiecutter, called 
`[camelcase name of model]ModelIntegrationTests` and only has to be filled out by you. To ensure that those tests are passing, 
run 

```python
RUN_SLOW=1 pytest tests/test_modeling_[name of model].py::[camelcase name of model]ModelIntegrationTests
```

Second, all features that are special to [name of model] should be tested additionally in a separate test under `[camelcase name of model]ModelTester`/`[camelcase name of model]ModelTest`. This part is often forgotten, but is extremely useful in two ways:

- It helps to transfer the knowledge you have acquired during the model addition to the community by showing how the special feature of [name of model] should work
- Future contributors can quickly test changes to the model by running those special tests

[Here the teacher should add very specific information on what special features of the model should be tested additionally]
[...]
[...]

### Implement the tokenizer

Next, we should add the tokenizer of [name of model]. Usually, the tokenizer is equivalent or very similar to an already existing tokenizer of 
🤗Transformers. 

[Here the teacher should add a comment whether a new tokenizer is required or if this is not the case which existing tokenizer closest resembles 
 [name of model]'s tokenizer and how the tokenizer should be implemented]
 [...]
 [...]
 
 Next, it is very important to find/extract the original tokenizer file and to manage to load this file into the 🤗Transformers' implementation 
 of the tokenizer.

For [name of model], the tokenizer files can be found here:
- [To be filled out by teacher]

and having implemented the  🤗Transformers' version of the tokenizer can be loaded as follows:

[To be filled out by teacher]
 
 To ensure that the tokenizer works correctly, it is recommend to analogous to the model, first create a script in the original repository that 
 inputs a string and returns the `input_ids`. It could look similar to this (in pseudo code):
 
 ```bash
 input_str = "This is a long example input string containing special characters .$?-, numbers 2872 234 12 and words."
 model = [name of model]Model.load_pretrained_checkpoint(/path/to/checkpoint/)
 input_ids = model.tokenize(input_str)
 ```
 
 You might have to take a deeper look again into the original repository to find the correct tokenizer function or you might even have to do 
 changes to your clone of the original repository to only output the `input_ids`. Having written a functional tokenization script that uses the original repository, an analogous script for 🤗Transformers should be created. It should look similar to this:
 
 ```python
 from transformers import [camelcase name of model]Tokenizer
 input_str = "This is a long example input string containing special characters .$?-, numbers 2872 234 12 and words."
 
 tokenizer = [camelcase name of model]Tokenizer.from_pretrained(/path/to/tokenizer/folder/)
 
 input_ids = tokenizer(input_str).input_ids
 ```
 
 When both `input_ids` yield the same values, as a final step a tokenizer test file should also be added. 
 
 [Here teacher should write how to do so. Tokenizer test files strongly vary between different model implementations]
 
 Analogous to the modeling test files of [name of model], the tokenization test files of [name of model] should 
 contain a couple of hard-coded integration tests.
 
 [Here teacher should point the student to test files of similar tokenizers]
 
 As a final step, it is recommended to add one large integration tests to `tests/test_modeling_[name of model].py` that does some end-to-end
 testing using both the model and the tokenizer.
 
 [Here teacher should again point to an existing similar test of another model that the student can copy & adapt]

### Add Docstring

Now, all the necessary functionality for [name of model] is added - you're almost done! The only thing left to add is a nice docstring and a doc page. The Cookiecutter should have added a template file called `docs/source/model_doc/[name of model].rst`
that you should fill out. Users of your model will usually first look at this page before using your model. Hence, the documentation must be understandable and concise. It is very useful for the community to add some *Tips* to show how the model should be used.
Don't hesitate to ping [name of teacher] regarding the docstrings.

Next, make sure that the docstring added to `src/transformers/models/[name of model]/modeling_[name of model].py` is correct and included all necessary
inputs and outputs. It is always to good to remind oneself that documentation should be treated at least as carefully as the code in 🤗Transformers since the documentation is usually the first contact point of the community with the model.

### Code refactor

Great, now you have added all the necessary code for [name of model]. 
At this point, you should correct some potential incorrect code style by running:

```bash
make style
```

and verify that your coding style passes the quality check:

```bash
make quality
```

There are a couple of other very strict design tests in 🤗Transformers that might still be failing, 
which shows up in the tests of your pull request. This is often because of some missing information in the docstring or some incorrect naming. [name of teacher] will surely help you if you're stuck here.

Lastly, it is always a good idea to refactor one's code after having ensured that the code works correctly. 
With all tests passing, now it's a good time to go over the added code again and do some refactoring.

You have now finished the coding part, congratulation! 🎉 You are Awesome! 😎

### Upload the models to the modeling hub

In this final part, you should convert and upload all checkpoints to the model hub and add a model card for each uploaded model checkpoint. You should work alongside [name of teacher] here to decide on a fitting name 
for each checkpoint and to get the required access rights to be able to upload the model under the author's organization
of [name of model].

It is worth spending some time to create fitting model cards for each checkpoint. The model cards should highlight the specific characteristics
of this particular checkpoint, *e.g.* On which dataset was the checkpoint pretrained/fine-tuned on? On what down-stream task should the model be used? and also include some code on how to correctly use the model. 

### (Optional) Add notebook

It is very helpful to add a notebook that showcases in-detail how [name of model] can be used for inference and/or fine-tuned 
on a down-stream task. This is not mandatory to merge your PR, but very useful for the community. 

## Share your work!!

Now, it's time to get some credit from the community for your work! Having completed a model addition is a major contribution to Transformers and the whole NLP community. Your code will certainly be used by hundreds of developers and researchers. You should 
be proud of your work and share your achievement with the community.

**You have made another model that is super easy to access for everyone in the community! 🤯**
