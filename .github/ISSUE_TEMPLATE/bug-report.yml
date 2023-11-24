name: "\U0001F41B Bug Report"
description: Submit a bug report to help us improve transformers
body:
  - type: textarea
    id: system-info
    attributes:
      label: System Info
      description: Please share your system info with us. You can run the command `transformers-cli env` and copy-paste its output below.
      placeholder: transformers version, platform, python version, ...
    validations:
      required: true

  - type: textarea
    id: who-can-help
    attributes:
      label: Who can help?
      description: |
        Your issue will be replied to more quickly if you can figure out the right person to tag with @
        If you know how to use git blame, that is the easiest way, otherwise, here is a rough guide of **who to tag**.
        
        All issues are read by one of the core maintainers, so if you don't know who to tag, just leave this blank and
        a core maintainer will ping the right person.
        
        Please tag fewer than 3 people.
        
        Models:

          - text models: @ArthurZucker and @younesbelkada
          - vision models: @amyeroberts
          - speech models: @sanchit-gandhi
          - graph models: @clefourrier
        
        Library:
        
          - flax: @sanchit-gandhi
          - generate: @gante
          - pipelines: @Narsil
          - tensorflow: @gante and @Rocketknight1
          - tokenizers: @ArthurZucker
          - trainer: @muellerzr and @pacman100
        
        Integrations:
        
          - deepspeed: HF Trainer/Accelerate: @pacman100
          - ray/raytune: @richardliaw, @amogkam
          - Big Model Inference: @SunMarc
          - quantization (bitsandbytes, autogpt): @SunMarc and @younesbelkada
        
        Documentation: @stevhliu and @MKhalusova
        
        Model hub:

          - for issues with a model, report at https://discuss.huggingface.co/ and tag the model's creator.
        
        HF projects:
        
          - accelerate: [different repo](https://github.com/huggingface/accelerate)
          - datasets: [different repo](https://github.com/huggingface/datasets)
          - diffusers: [different repo](https://github.com/huggingface/diffusers)
          - rust tokenizers: [different repo](https://github.com/huggingface/tokenizers)
        
        Maintained examples (not research project or legacy):
        
          - Flax: @sanchit-gandhi
          - PyTorch: See Models above and tag the person corresponding to the modality of the example.
          - TensorFlow: @Rocketknight1

        Research projects are not maintained and should be taken as is.

      placeholder: "@Username ..."

  - type: checkboxes
    id: information-scripts-examples
    attributes:
      label: Information
      description: 'The problem arises when using:'
      options:
        - label: "The official example scripts"
        - label: "My own modified scripts"

  - type: checkboxes
    id: information-tasks
    attributes:
      label: Tasks
      description: "The tasks I am working on are:"
      options:
        - label: "An officially supported task in the `examples` folder (such as GLUE/SQuAD, ...)"
        - label: "My own task or dataset (give details below)"

  - type: textarea
    id: reproduction
    validations:
      required: true
    attributes:
      label: Reproduction
      description: |
        Please provide a code sample that reproduces the problem you ran into. It can be a Colab link or just a code snippet.
        If you have code snippets, error messages, stack traces please provide them here as well.
        Important! Use code tags to correctly format your code. See https://help.github.com/en/github/writing-on-github/creating-and-highlighting-code-blocks#syntax-highlighting
        Do not use screenshots, as they are hard to read and (more importantly) don't allow others to copy-and-paste your code.

      placeholder: |
        Steps to reproduce the behavior:
          
          1.
          2.
          3.
          

  - type: textarea
    id: expected-behavior
    validations:
      required: true
    attributes:
      label: Expected behavior
      description: "A clear and concise description of what you would expect to happen."
