# `transformers` philosophy

In this file, we would like to sum up the core philosophy amd coding standards that we go by when reviewing PRs.

# API
### Transformers API

- All models and their building blocks should be prefixed with the model name (camel-cased).
- All models should take **the config in their init** instead of multiple keyword arguments. Keyword arguments are only allowed in a building block for arguments that can change (example: hidden size changes in a ResNet depending on the layer index).
- `XxxxxConfig`'s names should almost never be new. Ex: for a MoE, look at other MoEs and use the same config names like `router_jitter_noise` or `num_experts`.
- All public models should return a `ModelOutput` or a tuple (depending on `return_dict`).
- If a model is added in an Auto class, its API (inputs/outputs) should be fully compatible with other models in the same auto class.
- As little code paths as possible. `if self.config.use_swin_norm` should not exist. Either use `ALL_LAYER_NORM` or make sure pretrained checkpoints all use it / don't. We are not providing a research codebase.

# Code

### Code style/quality

1. Simplicity

Aim for simple code, and help our contributors with your knowledge of the library, of torch, of python etc. Doubt them, think about vectorizing, think about our API in general etc.

We probably already have EVERYTHING implemented in `transformers` let’s not re-invent boiling water, and look for the closest model / closest implementation.

1. Readability

Let’s make the code readable. Once you simplify you should ask to split in 2 lines wherever possible, and look at `Gemma` or `Llava` for what an almost ideal model looks like

1. Explainability

Code to be self-explanatory, easy to debug, easy to re-use. This means, **no single letter variables**, **variable names that explain what is happening.**

But also **sources of new class / code**. If there is a new `MyAwesomeAttention` , should link to the original implementation, code or repo where it was introduced. We don't add Mr XXX's custom design code.

## Understandable code >> short code
Don’t try to make the code shorter if it harms readability. A lot of users dive into the source code and tweak it (that’s why we have the one file per model policy), so we really want the code to be as clear as possible. That means:

- don’t use one-letter variable names except in short loops, always use meaningful names.
- don’t use refactor one/two lines of code behind a separate function a reader will have to look for.
- a function that is used only once is probably unhelpful and you can put the same code where the function is used.
- a function that does its result in one line should probably not exist.
- use comments whenever you write a line that is obscure to explain its goal.
- use ternary operators in the obvious cases only (`variable = this if condition else that` ), for more complex tests with several elif branches, revert to the usual syntax.


## Fragment code for easy debugging

[CODE STYLE]

It’s easier to debug code when each line does one operation instead of adding multiple `.do_this()` , `.and_that()` , `also_this()` , `and_let_us_not_forget_that()` in a single line. Similarly, avoid using lambda functions (except maybe when sorting things and providing a `key` function).

This will also make your code easier to read, in accordance with the previous point.

## f-string >> everything else

[CODE STYLE]

We always use f-strings because they make code easier to read and are faster than any other option for formatting. The only exception is if a given string needs to be formatted with values that are not available right now.


## Use proper errors in code, no asserts

[CODE STYLE]

Asserts can be disabled by the user if they execute their script with a flag. Therefore, they aren’t suitable for errors you want to raise all the time. In general in the source code, use a test and raise the appropriate exception with a clear message. In tests however, you can use assert statements.