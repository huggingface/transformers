# Modular transformers

On of the long standing critic we have seen is also the core philosophy of transformers: [`single model, single file`](https://huggingface.co/blog/transformers-design-philosophy).
For the past 4 years we were adamant on having no inheritance in `transformers`. The core motivation stems from our values and are the following:

- code readability: we want people to understand how a model works without being exposed to parameters or arguments that are related to `transformers` but not the model itself. 
    -> we do not allow code paths
    -> we do not allow one liners
    -> we force meaningful variable names
    -> we do not allow inheriting from other models
- reproducibility: we want people to easily reproduce, debug and run models. For this, we want them to easily copy paste a modeling code and make it work (with deletion yes, but without having to add more code from other modules).
- education: as a lot of our code ends up being reused, we have a duty to write code that is educative and understandable to anyone that wants to understand how that code works. 
- debugging: inheritance is particularly annoying when debugging, we made a moral choice to ease debugging experience, only having to go in the modeling.

We introduces the `make fix-copies` along with the `# Copied from` macros, which help re-using the code, but in the past 2 years we realized the limitation from it. 

## What is it?

It's an alpha tool / feature that introduces a new `linter`. The linter converts a `modular` file to a `single model, single file`. The key idea is to allow people to add new models using inheritance, which will make identifying differences between models a lot easier, but keeping the single model single file policy. 

When you design your new model using modularity, you just need to think in the most pythonic way, the linter is just going to "flatten" or "unravel" 1 level of inheritance.

The linter will convert the `modular_my_model.py` to the classic `modeling_my_model.py`, from which we will import everything as usual. But, **importing from the modular file** or **importing from the modeling file** should yield the same results! 

It is also a drop in replacement for our `# Copied from` markers.

### Details 

The "linter", which unravels the inheritance and creates all single-files from the modular file, will flatten the inheritance while trying to be invisible to Python users. For example:

- If a configuration class inherits from another and adds/deletes an argument, the generated file will either directly reference it (in case of addition) or completely remove it (in case of deletion).
- If a class inherits from another, for example: class GemmaModel(LlamaModel):, dependencies are automatically inferred. All submodules will be automatically inferred from the superclass.

You should be able to write everything (the tokenizer, the image processor, the model, the config) in this `modular` file, and the corresponding files will be created for you. 

### Enforcement

We are also introducing a new test, that makes sure the generated content matches what is present in the `modular_xxxx.py`



### Examples
You can find a list of examples


## What it is not

It is not a replacement for the modeling code (yet?), and if your model is not based on anything else that ever existed, then you can add a `modeling` file as usual.