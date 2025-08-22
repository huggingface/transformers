<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Sharing

The Hugging Face [Hub](https://hf.co/models) is a platform for sharing, discovering, and consuming models of all different types and sizes. We highly recommend sharing your model on the Hub to push open-source machine learning forward for everyone!

This guide will show you how to share a model to the Hub from Transformers.

## Set up

To share a model to the Hub, you need a Hugging Face [account](https://hf.co/join). Create a [User Access Token](https://hf.co/docs/hub/security-tokens#user-access-tokens) (stored in the [cache](./installation#cache-directory) by default) and login to your account from either the command line or notebook.

<hfoptions id="share">
<hfoption id="huggingface-CLI">

```bash
hf auth login
```

</hfoption>
<hfoption id="notebook">

```py
from huggingface_hub import notebook_login

notebook_login()
```

</hfoption>
</hfoptions>

## Repository features

<Youtube id="XvSGPZFEjDY"/>

Each model repository features versioning, commit history, and diff visualization.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vis_diff.png"/>
</div>

Versioning is based on [Git](https://git-scm.com/) and [Git Large File Storage (LFS)](https://git-lfs.github.com/), and it enables revisions, a way to specify a model version with a commit hash, tag or branch.

For example, use the `revision` parameter in [`~PreTrainedModel.from_pretrained`] to load a specific model version from a commit hash.

```py
model = AutoModel.from_pretrained(
    "julien-c/EsperBERTo-small", revision="4c77982"
)
```

Model repositories also support [gating](https://hf.co/docs/hub/models-gated) to control who can access a model. Gating is common for allowing a select group of users to preview a research model before it's made public.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/gated-model.png"/>
</div>

A model repository also includes an inference [widget](https://hf.co/docs/hub/models-widgets) for users to directly interact with a model on the Hub.

Check out the Hub [Models](https://hf.co/docs/hub/models) documentation to for more information.

## Model framework conversion

Reach a wider audience by making a model available in PyTorch, TensorFlow, and Flax. While users can still load a model if they're using a different framework, it is slower because Transformers needs to convert the checkpoint on the fly. It is faster to convert the checkpoint first.

<hfoptions id="convert">
<hfoption id="PyTorch">

Set `from_tf=True` to convert a checkpoint from TensorFlow to PyTorch and then save it.

```py
from transformers import DistilBertForSequenceClassification

pt_model = DistilBertForSequenceClassification.from_pretrained("path/to/awesome-name-you-picked", from_tf=True)
pt_model.save_pretrained("path/to/awesome-name-you-picked")
```

</hfoption>
<hfoption id="TensorFlow">

Set `from_pt=True` to convert a checkpoint from PyTorch to TensorFlow and then save it.

```py
from transformers import TFDistilBertForSequenceClassification

tf_model = TFDistilBertForSequenceClassification.from_pretrained("path/to/awesome-name-you-picked", from_pt=True)
tf_model.save_pretrained("path/to/awesome-name-you-picked")
```

</hfoption>
<hfoption id="Flax">

Set `from_pt=True` to convert a checkpoint from PyTorch to Flax and then save it.

```py
from transformers import FlaxDistilBertForSequenceClassification
flax_model = FlaxDistilBertForSequenceClassification.from_pretrained(
    "path/to/awesome-name-you-picked", from_pt=True
)
flax_model.save_pretrained("path/to/awesome-name-you-picked")
```

</hfoption>
</hfoptions>

## Uploading a model

There are several ways to upload a model to the Hub depending on your workflow preference. You can push a model with [`Trainer`], a callback for TensorFlow models, call [`~PreTrainedModel.push_to_hub`] directly on a model, or use the Hub web interface.

<Youtube id="Z1-XMy-GNLQ"/>

### Trainer

[`Trainer`] can push a model directly to the Hub after training. Set `push_to_hub=True` in [`TrainingArguments`] and pass it to [`Trainer`]. Once training is complete, call [`~transformers.Trainer.push_to_hub`] to upload the model.

[`~transformers.Trainer.push_to_hub`] automatically adds useful information like training hyperparameters and results to the model card.

```py
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="my-awesome-model", push_to_hub=True)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.push_to_hub()
```

### PushToHubCallback

For TensorFlow models, add the [`PushToHubCallback`] to the [fit](https://keras.io/api/models/model_training_apis/#fit-method) method.

```py
from transformers import PushToHubCallback

push_to_hub_callback = PushToHubCallback(
    output_dir="./your_model_save_path", tokenizer=tokenizer, hub_model_id="your-username/my-awesome-model"
)
model.fit(tf_train_dataset, validation_data=tf_validation_dataset, epochs=3, callbacks=push_to_hub_callback)
```

### PushToHubMixin

The [`~utils.PushToHubMixin`] provides functionality for pushing a model or tokenizer to the Hub.

Call [`~utils.PushToHubMixin.push_to_hub`] directly on a model to upload it to the Hub. It creates a repository under your namespace with the model name specified in [`~utils.PushToHubMixin.push_to_hub`].

```py
model.push_to_hub("my-awesome-model")
```

Other objects like a tokenizer or TensorFlow model are also pushed to the Hub in the same way.

```py
tokenizer.push_to_hub("my-awesome-model")
```

Your Hugging Face profile should now display the newly created model repository. Navigate to the **Files** tab to see all the uploaded files.

Refer to the [Upload files to the Hub](https://hf.co/docs/hub/how-to-upstream) guide for more information about pushing files to the Hub.

### Hub web interface

The Hub web interface is a no-code approach for uploading a model.

1. Create a new repository by selecting [**New Model**](https://huggingface.co/new).

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/new_model_repo.png"/>
</div>

Add some information about your model:

- Select the **owner** of the repository. This can be yourself or any of the organizations you belong to.
- Pick a name for your model, which will also be the repository name.
- Choose whether your model is public or private.
- Set the license usage.

2. Click on **Create model** to create the model repository.

3. Select the **Files** tab and click on the **Add file** button to drag-and-drop a file to your repository. Add a commit message and click on **Commit changes to main** to commit the file.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/upload_file.png"/>
</div>

## Model card

[Model cards](https://hf.co/docs/hub/model-cards#model-cards) inform users about a models performance, limitations, potential biases, and ethical considerations. It is highly recommended to add a model card to your repository!

A model card is a `README.md` file in your repository. Add this file by:

- manually creating and uploading a `README.md` file
- clicking on the **Edit model card** button in the repository

Take a look at the Llama 3.1 [model card](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) for an example of what to include on a model card.

Learn more about other model card metadata (carbon emissions, license, link to paper, etc.) available in the [Model Cards](https://hf.co/docs/hub/model-cards#model-cards) guide.
