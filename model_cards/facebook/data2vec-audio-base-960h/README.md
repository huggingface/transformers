<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
           <img alt="Data2Vec Audio Model" src="" >
    </div>
</div>

# facebook/data2vec-audio-base-960h

[Data2Vec: A General Framework for Self-supervised Learning in Speech, Vision and Language](https://arxiv.org/abs/2202.03555)

`data2vec` is Meta AI’s unified self-supervised learning method that works across audio, text, and vision. This specific version is trained on **960 hours of speech data (LibriSpeech)** and fine-tuned for **automatic speech recognition (ASR)**.

What makes it unique? Instead of learning to predict tokens or raw inputs like traditional models, Data2Vec predicts latent representations, making it **more general and scalable** across modalities.

You can find all the original Data2Vec checkpoints under the [data2vec collection](https://huggingface.co/models?search=data2vec).

> [!TIP]
> This model was contributed by [facebook](https://huggingface.co/facebook).  
> Click on the data2vec models in the right sidebar for more examples of how to apply Data2Vec to **speech recognition** tasks.

---

## Usage Examples

<hfoptions id="usage">

<hfoption id="Pipeline">

```python
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="facebook/data2vec-audio-base-960h")
result = pipe("path/to/audio.wav")
print(result["text"])