# What does this PR do?

<!--
Congratulations! You've made it this far! You're not quite done yet though.

Once merged, your PR is going to appear in the release notes with the title you set, so make sure it's a great title that fully reflects the extent of your awesome contribution.

Then, please replace this with a description of the change and which issue is fixed (if applicable). Please also include relevant motivation and context. List any dependencies (if any) that are required for this change.

Once you're done, someone will review your PR shortly (see the section "Who can review?" below to tag some potential reviewers). They may suggest changes to make the code even better. If no one reviewed your PR after a week has passed, don't hesitate to post a new comment @-mentioning the same persons---sometimes notifications get lost.
-->

<!-- Remove if not applicable -->

Fixes # (issue)

## Code Agent Policy

The Transformers repo is currently being overwhelmed by a large number of PRs and issue comments written by
code agents. We are currently bottlenecked by our ability to review and respond to them. As a result, 
**we ask that new users do not submit pure code agent PRs** at this time. 
You may use code agents in drafting or to help you diagnose issues. We'd also ask autonomous "OpenClaw"-like agents
not to open any PRs or issues for the moment.

PRs that appear to be fully agent-written will probably be closed without review, and we may block users who do this
repeatedly or maliciously. 

This is a rapidly-evolving situation that's causing significant shockwaves in the open-source community. As a result, 
this policy is likely to be updated regularly in the near future. For more information, please read [`CONTRIBUTING.md`](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md).

- [ ] I confirm that this is not a pure code agent PR.

## Before submitting
- [ ] This PR fixes a typo or improves the docs (you can dismiss the other checks if that's the case).
- [ ] Did you read the [contributor guideline](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md#create-a-pull-request),
      Pull Request section?
- [ ] Was this discussed/approved via a Github issue or the [forum](https://discuss.huggingface.co/)? Please add a link
      to it if that's the case.
- [ ] Did you make sure to update the documentation with your changes? Here are the
      [documentation guidelines](https://github.com/huggingface/transformers/tree/main/docs), and
      [here are tips on formatting docstrings](https://github.com/huggingface/transformers/tree/main/docs#writing-source-documentation).
- [ ] Did you write any new necessary tests?


## Who can review?

Anyone in the community is free to review the PR once the tests have passed. Feel free to tag
members/contributors who may be interested in your PR.

<!-- Your PR will be replied to more quickly if you can figure out the right person to tag with @

 If you know how to use git blame, that is the easiest way, otherwise, here is a rough guide of **who to tag**.
 Please tag fewer than 3 people.

Models:

- text models: @ArthurZucker @Cyrilvallez
- vision models: @yonigozlan @molbap
- audio models: @eustlb @ebezzam @vasqu
- multimodal models: @zucchini-nlp
- graph models: @clefourrier

Library:

- generate: @zucchini-nlp (visual-language models) or @gante (all others)
- continuous batching: @remi-or @ArthurZucker @McPatate
- pipelines: @Rocketknight1
- tokenizers: @ArthurZucker and @itazap
- trainer: @SunMarc
- attention: @vasqu @ArthurZucker @CyrilVallez
- model loading (from pretrained, etc): @CyrilVallez
- distributed: @3outeille @ArthurZucker
- CIs: @ydshieh

Integrations:

- ray/raytune: @richardliaw, @amogkam
- Big Model Inference: @SunMarc
- quantization: @SunMarc
- kernels: @drbh
- peft: @BenjaminBossan @githubnemo

Devices/Backends:

- AMD ROCm: @ivarflakstad
- Intel XPU: @IlyasMoutawwakil
- Ascend NPU: @ivarflakstad 

Documentation: @stevhliu

Research projects are not maintained and should be taken as is.

 -->
