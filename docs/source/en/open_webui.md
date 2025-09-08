#  Audio transcriptions with WebUI and `transformers serve`

This guide shows how to do audio transcription for chat purposes, using `transformers serve` and [Open WebUI](https://openwebui.com/). This guide assumes you have Open WebUI installed on your machine and ready to run. Please refer to the examples above to use the text functionalities of `transformer serve` with Open WebUI -- the instructions are the same.

To start, let's launch the server. Some of Open WebUI's requests require [CORS](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/CORS), which is disabled by default for security reasons, so you need to enable it:

```shell
transformers serve --enable-cors
```

Before you can speak into Open WebUI, you need to update its settings to use your server for speech to text (STT) tasks. Launch Open WebUI, and navigate to the audio tab inside the admin settings. If you're using Open WebUI with the default ports, [this link (default)](http://localhost:3000/admin/settings/audio) or [this link (python deployment)](http://localhost:8080/admin/settings/audio) will take you there. Do the following changes there:
1. Change the type of "Speech-to-Text Engine" to "OpenAI";
2. Update the address to your server's address -- `http://localhost:8000/v1` by default;
3. Type your model of choice into the "STT Model" field, e.g. `openai/whisper-large-v3` ([available models](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=trending)).

If you've done everything correctly, the audio tab should look like this

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_openwebui_stt_settings.png"/>
</h3>

You're now ready to speak! Open a new chat, utter a few words after hitting the microphone button, and you should see the corresponding text on the chat input after the model transcribes it.
