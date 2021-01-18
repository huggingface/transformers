# How to add a model to 🤗Transformers
	
TODO

## General overview of 🤗TTransformers

TODO

- Mention some general design principles
- Mention the single file policy
- Mention the copy over abstraction policy
- ...

### Overview of PreTrainedModel

TODO

- Add graphic

### Overview of PreTrainedTokenizer

TODO

- Add graphic

## Current open projects of models to add

TODO

- list all model proposals here


## Prepare your environment

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
  
## Implement a new model architecture into 🤗Transformers
	
Next, you can finally add code to 🤗Transformers. Go into the clone 
of your 🤗Transformers' fork:

### Use the Cookiecutter to automatically generate the model's code

To begin with head over to the [🤗Transformers templates](https://github.com/huggingface/transformers/tree/master/templates/adding_a_new_model) to make
use of our `cookiecutter` implementation to automatically generate all the relevant files for your model. Again, we recommend only adding the PyTorch version of the model at first.
Make sure you follow the instructions of the `README.md` on the [🤗Transformers templates](https://github.com/huggingface/transformers/tree/master/templates/adding_a_new_model) carefully.

### Adapt the generated code for the model
	
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

### Adapt the generated code for the tokenizer

Next, we should add the tokenizer of [name of model]. Usually, the tokenizer is equivalent or very similar to an already existing tokenizer of 
🤗Transformers.                                                                                                                                                                                                                                     ### Adapt the generated code for the test

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


### Adapt the docstring

Now, all the necessary functionality for [name of model] is added - you're almost done! The only thing left to add is a nice docstring and a doc page. The Cookiecutter should have added a template file called `docs/source/model_doc/[name of model].rst`
that you should fill out. Users of your model will usually first look at this page before using your model. Hence, the documentation must be understandable and concise. It is very useful for the community to add some *Tips* to show how the model should be used.
Don't hesitate to ping [name of teacher] regarding the docstrings.

Next, make sure that the docstring added to `src/transformers/models/[name of model]/modeling_[name of model].py` is correct and included all necessary
inputs and outputs. It is always to good to remind oneself that documentation should be treated at least as carefully as the code in 🤗Transformers since the documentation is usually the first contact point of the community with the model.

## Models added by the community

TODO 

- list models that were added by the community here
