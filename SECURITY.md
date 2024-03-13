# Security Policy

## Hugging Face Hub, remote artefacts, and remote code

Transformers is an open-source software that has strong ties to the Hugging Face Hub. While it can be used completely
offline with models downloaded locally, it provides a very simple way to download, use, and manage models locally.

When downloading artefacts that have been uploaded by others on any platform, you expose yourself to risks. Please
read below for the security recommendations in order to keep your runtime and local environment safe.

### Remote artefacts

Models uploaded on the Hugging Face Hub come in different formats. We heavily recommend uploading and downloading
models in the [`safetensors`](https://github.com/huggingface/safetensors) format (which is the default prioritized
by the transformers library), as developed specifically to prevent code execution in your runtime.

Transformers will default to downloading models in this format if available, but will get other formats available
if `safetensors` isn't available. You can force this format by using the `use_safetensors` parameter.

### Remote code

#### Modeling

Transformers is host to many model architectures, but is also the bridge between your Python runtime and models that
are defined within the model repositories on the Hugging Face Hub.

These models require the `trust_remote_code=True` parameter to be set when using them; please **always** verify
the content of the modeling files when using this argument. We recommend setting a revision in order to ensure you
protect yourself from updates on the repository.

#### Tools

Through the `Agent` framework, remote tools can be downloaded to be used by the Agent. You're to specify these tools
yourself, but please keep in mind that their code will be run on your machine if the Agent chooses to run them.

Please inspect the code of the tools before passing them to the Agent to protect your runtime and local setup.

## Reporting a Vulnerability

🤗 We have our bug bounty program set up with HackerOne. Please feel free to submit vulnerability reports to our private program at https://hackerone.com/hugging_face.
Note that you'll need to be invited to our program, so send us a quick email at security@huggingface.co if you've found a vulnerability.
