# Using Cursor as a client of transformers serve

This example shows how to use `transformers serve` as a local LLM provider for [Cursor](https://cursor.com/), the popular IDE. In this particular case, requests to `transformers serve` will come from an external IP (Cursor's server IPs), which requires some additional setup. Furthermore, some of Cursor's requests require [CORS](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/CORS), which is disabled by default for security reasons.

To launch a server with CORS enabled, run

```shell
transformers serve --enable-cors
```

You'll also need to expose your server to external IPs. A potential solution is to use [`ngrok`](https://ngrok.com/), which has a permissive free tier. After setting up your `ngrok` account and authenticating on your server machine, you run

```shell
ngrok http [port]
```

where `port` is the port used by `transformers serve` (`8000` by default). On the terminal where you launched `ngrok`, you'll see a https address in the "Forwarding" row, as in the image below. This is the address to send requests to.

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_serve_ngrok.png"/>
</h3>

You're now ready to set things up on the app side! In Cursor, while you can't set a new provider, you can change the endpoint for OpenAI requests in the model selection settings. First, navigate to "Settings" > "Cursor Settings", "Models" tab, and expand the "API Keys" collapsible. To set your `transformers serve` endpoint, follow this order:
1. Unselect ALL models in the list above (e.g. `gpt4`, ...);
2. Add and select the model you want to use (e.g. `Qwen/Qwen3-4B`)
3. Add some random text to OpenAI API Key. This field won't be used, but it can’t be empty;
4. Add the https address from `ngrok` to the "Override OpenAI Base URL" field, appending `/v1` to the address (i.e. `https://(...).ngrok-free.app/v1`);
5. Hit "Verify".

After you follow these steps, your "Models" tab should look like the image below. Your server should also have received a few requests from the verification step.

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_serve_cursor.png"/>
</h3>

You are now ready to use your local model in Cursor! For instance, if you toggle the AI Pane, you can select the model you added and ask it questions about your local files.

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_serve_cursor_chat.png"/>
</h3>


