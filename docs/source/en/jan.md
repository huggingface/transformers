# Jan: using the serving API as a local LLM provider

This example shows how to use `transformers serve` as a local LLM provider for the [Jan](https://jan.ai/) app. Jan is a ChatGPT-alternative graphical interface, fully running on your machine. The requests to `transformers serve` come directly from the local app -- while this section focuses on Jan, you can extrapolate some instructions to other apps that make local requests.

## Running models locally

To connect `transformers serve` with Jan, you'll need to set up a new model provider ("Settings" > "Model Providers"). Click on "Add Provider", and set a new name. In your new model provider page, all you need to set is the "Base URL" to the following pattern:

```shell
http://[host]:[port]/v1
```

where `host` and `port` are the `transformers serve` CLI parameters (`localhost:8000` by default). After setting this up, you should be able to see some models in the "Models" section, hitting "Refresh". Make sure you add some text in the "API key" text field too -- this data is not actually used, but the field can't be empty. Your custom model provider page should look like this:

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_serve_jan_model_providers.png"/>
</h3>

You are now ready to chat!

> [!TIP]
> You can add any `transformers`-compatible model to Jan through `transformers serve`. In the custom model provider you created, click on the "+" button in the "Models" section and add its Hub repository name, e.g. `Qwen/Qwen3-4B`.

## Running models on a separate machine

To conclude this example, let's look into a more advanced use-case. If you have a beefy machine to serve models with, but prefer using Jan on a different device, you need to add port forwarding. If you have `ssh` access from your Jan machine into your server, this can be accomplished by typing the following to your Jan machine's terminal

```
ssh -N -f -L 8000:localhost:8000 your_server_account@your_server_IP -p port_to_ssh_into_your_server
```

Port forwarding is not Jan-specific: you can use it to connect `transformers serve` running in a different machine with an app of your choice.
