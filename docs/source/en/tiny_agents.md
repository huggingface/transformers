### `tiny-agents` CLI and MCP Tools

To showcase the use of MCP tools, let's see how to integrate the `transformers serve` server with the [`tiny-agents`](https://huggingface.co/blog/python-tiny-agents) CLI.

> [!TIP]
> Many Hugging Face Spaces can be used as MCP servers, as in this example. You can find all compatible Spaces [here](https://huggingface.co/spaces?filter=mcp-server).

The first step to use MCP tools is to let the model know which tools are available. As an example, let's consider a `tiny-agents` configuration file with a reference to an [image generation MCP server](https://evalstate-flux1-schnell.hf.space/).

```json
{
    "model": "Menlo/Jan-nano",
    "endpointUrl": "http://localhost:8000",
    "servers": [
        {
            "type": "sse",
            "url": "https://evalstate-flux1-schnell.hf.space/gradio_api/mcp/sse"
        }
    ]
}
```

You can then launch your `tiny-agents` chat interface with the following command.

```bash
tiny-agents run path/to/your/config.json
```

If you have `transformers serve` running in the background, you're ready to use MCP tools from a local model! For instance, here's the example of a chat session with `tiny-agents`:

```bash
Agent loaded with 1 tools:
 • flux1_schnell_infer
»  Generate an image of a cat on the moon
<Tool req_0_tool_call>flux1_schnell_infer {"prompt": "a cat on the moon", "seed": 42, "randomize_seed": true, "width": 1024, "height": 1024, "num_inference_steps": 4}

Tool req_0_tool_call
[Binary Content: Image image/webp, 57732 bytes]
The task is complete and the content accessible to the User
Image URL: https://evalstate-flux1-schnell.hf.space/gradio_api/file=/tmp/gradio/3dbddc0e53b5a865ed56a4e3dbdd30f3f61cf3b8aabf1b456f43e5241bd968b8/image.webp
380576952

I have generated an image of a cat on the moon using the Flux 1 Schnell Image Generator. The image is 1024x1024 pixels and was created with 4 inference steps. Let me know if you would like to make any changes or need further assistance!
```

