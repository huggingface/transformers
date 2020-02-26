import logging
from argparse import ArgumentParser, Namespace
from typing import Any, List, Optional

import typer

from transformers.cli._types import PipelineTask
from transformers.pipelines import SUPPORTED_TASKS, Pipeline, PipelineDataFormat, pipeline


try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Body
    from fastapi.routing import APIRoute
    from pydantic import BaseModel
    from starlette.responses import JSONResponse

    _serve_dependencies_installed = True
except (ImportError, AttributeError):
    BaseModel = object

    def Body(*x, **y):
        pass

    _serve_dependencies_installed = False


logger = logging.getLogger("transformers-cli/serving")


class ServeModelInfoResult(BaseModel):
    """
    Expose model information
    """

    infos: dict


class ServeTokenizeResult(BaseModel):
    """
    Tokenize result model
    """

    tokens: List[str]
    tokens_ids: Optional[List[int]]


class ServeDeTokenizeResult(BaseModel):
    """
    DeTokenize result model
    """

    text: str


class ServeForwardResult(BaseModel):
    """
    Forward result model
    """

    output: Any


def serve(
    task: PipelineTask,
    model: str = typer.Option(None, help="Name or path to the model to instantiate."),
    config: str = typer.Option(None, help="Name or path to the model's config to instantiate."),
    tokenizer: str = typer.Option(None, help="Name of the tokenizer to use. (default: same as model)"),
    host: str = "localhost",
    port: int = 8888,
    workers: int = typer.Option(1, help="Number of HTTP workers"),
    device: int = typer.Option(
        -1, help="Indicate the device to run onto, -1 indicates CPU, >= 0 indicates GPU (default: -1)"
    ),
):
    """Serve a supported pipeline task as an HTTP API."""

    nlp = pipeline(task=task.value, model=model if model else None, config=config, tokenizer=tokenizer, device=device,)

    if not _serve_dependencies_installed:
        raise RuntimeError(
            "Using serve command requires FastAPI and unicorn. "
            'Please install transformers with [serving]: pip install "transformers[serving]".'
            "Or install FastAPI and unicorn separately."
        )
    else:
        logger.info("Serving model over {}:{}".format(host, port))

        def model_info():
            return ServeModelInfoResult(infos=vars(nlp.model.config))

        def tokenize(text_input: str = Body(None, embed=True), return_ids: bool = Body(False, embed=True)):
            """
            Tokenize the provided input and eventually returns corresponding tokens id:
            - **text_input**: String to tokenize
            - **return_ids**: Boolean flags indicating if the tokens have to be converted to their integer mapping.
            """
            try:
                tokens_txt = nlp.tokenizer.tokenize(text_input)

                if return_ids:
                    tokens_ids = nlp.tokenizer.convert_tokens_to_ids(tokens_txt)
                    return ServeTokenizeResult(tokens=tokens_txt, tokens_ids=tokens_ids)
                else:
                    return ServeTokenizeResult(tokens=tokens_txt)

            except Exception as e:
                raise HTTPException(status_code=500, detail={"model": "", "error": str(e)})

        def detokenize(
            tokens_ids: List[int] = Body(None, embed=True),
            skip_special_tokens: bool = Body(False, embed=True),
            cleanup_tokenization_spaces: bool = Body(True, embed=True),
        ):
            """
            Detokenize the provided tokens ids to readable text:
            - **tokens_ids**: List of tokens ids
            - **skip_special_tokens**: Flag indicating to not try to decode special tokens
            - **cleanup_tokenization_spaces**: Flag indicating to remove all leading/trailing spaces and intermediate ones.
            """
            try:
                decoded_str = nlp.tokenizer.decode(tokens_ids, skip_special_tokens, cleanup_tokenization_spaces)
                return ServeDeTokenizeResult(model="", text=decoded_str)
            except Exception as e:
                raise HTTPException(status_code=500, detail={"model": "", "error": str(e)})

        def forward(inputs=Body(None, embed=True)):
            """
            **inputs**:
            **attention_mask**:
            **tokens_type_ids**:
            """

            # Check we don't have empty string
            if len(inputs) == 0:
                return ServeForwardResult(output=[], attention=[])

            try:
                # Forward through the model
                output = nlp(inputs)
                return ServeForwardResult(output=output)
            except Exception as e:
                raise HTTPException(500, {"error": str(e)})

        app = FastAPI(
            routes=[
                APIRoute(
                    "/", model_info, response_model=ServeModelInfoResult, response_class=JSONResponse, methods=["GET"],
                ),
                APIRoute(
                    "/tokenize",
                    tokenize,
                    response_model=ServeTokenizeResult,
                    response_class=JSONResponse,
                    methods=["POST"],
                ),
                APIRoute(
                    "/detokenize",
                    detokenize,
                    response_model=ServeDeTokenizeResult,
                    response_class=JSONResponse,
                    methods=["POST"],
                ),
                APIRoute(
                    "/forward",
                    forward,
                    response_model=ServeForwardResult,
                    response_class=JSONResponse,
                    methods=["POST"],
                ),
            ],
            timeout=600,
        )

        uvicorn.run(app, host=host, port=port, workers=workers)
