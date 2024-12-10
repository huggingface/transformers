# Security Policy

## Hugging Face Hub, remote artefacts, and remote code

Transformers is open-source software that is tightly coupled to the Hugging Face Hub. While you have the ability to use it
offline with pre-downloaded model weights, it provides a very simple way to download, use, and manage models locally.

When downloading artefacts that have been uploaded by others on any platform, you expose yourself to risks. Please
read below for the security recommendations in order to keep your runtime and local environment safe.

### Remote artefacts

Models uploaded on the Hugging Face Hub come in different formats. We heavily recommend uploading and downloading
models in the [`safetensors`](https://github.com/huggingface/safetensors) format (which is the default prioritized
by the transformers library), as developed specifically to prevent arbitrary code execution on your system.

To avoid loading models from unsafe formats(e.g. [pickle](https://docs.python.org/3/library/pickle.html), you should use the `use_safetensors` parameter. If doing so, in the event that no .safetensors file is present, transformers will error when loading the model.

### Remote code

#### Modeling

Transformers supports many model architectures, but is also the bridge between your Python runtime and models that
are stored in model repositories on the Hugging Face Hub.

These models require the `trust_remote_code=True` parameter to be set when using them; please **always** verify
the content of the modeling files when using this argument. We recommend setting a revision in order to ensure you
protect yourself from updates on the repository.

#### Tools

Through the `Agent` framework, remote tools can be downloaded to be used by the Agent. You're to specify these tools
yourself, but please keep in mind that their code will be run on your machine if the Agent chooses to run them.

Please inspect the code of the tools before passing them to the Agent to protect your runtime and local setup.

## Reporting a Vulnerability

Feel free to submit vulnerability reports to [security@huggingface.co](mailto:security@huggingface.co), where someone from the HF security team will review and recommend next steps. If reporting a vulnerability specific to open source, please note [Huntr](https://huntr.com) is a vulnerability disclosure program for open source software.
