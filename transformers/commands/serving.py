from argparse import ArgumentParser, Namespace
from typing import List, Optional, Union, Any

import logging

try:
    from uvicorn import run
    from fastapi import FastAPI, HTTPException, Body
    from pydantic import BaseModel
    _serve_dependancies_installed = True
except (ImportError, AttributeError):
    BaseModel = object
    Body = lambda *x, **y: None
    _serve_dependancies_installed = False

from transformers import Pipeline
from transformers.commands import BaseTransformersCLICommand
from transformers.pipelines import SUPPORTED_TASKS, pipeline

logger = logging.getLogger('transformers-cli/serving')

def serve_command_factory(args: Namespace):
    """
    Factory function used to instantiate serving server from provided command line arguments.
    :return: ServeCommand
    """
    nlp = pipeline(task=args.task,
                   model=args.model if args.model else None,
                   config=args.config,
                   tokenizer=args.tokenizer,
                   device=args.device)
    return ServeCommand(nlp, args.host, args.port)


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


class ServeCommand(BaseTransformersCLICommand):

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        """
        Register this command to argparse so it's available for the transformer-cli
        :param parser: Root parser to register command-specific arguments
        :return:
        """
        serve_parser = parser.add_parser('serve', help='CLI tool to run inference requests through REST and GraphQL endpoints.')
        serve_parser.add_argument('--task', type=str, choices=SUPPORTED_TASKS.keys(), help='The task to run the pipeline on')
        serve_parser.add_argument('--host', type=str, default='localhost', help='Interface the server will listen on.')
        serve_parser.add_argument('--port', type=int, default=8888, help='Port the serving will listen to.')
        serve_parser.add_argument('--model', type=str, help='Model\'s name or path to stored model.')
        serve_parser.add_argument('--config', type=str, help='Model\'s config name or path to stored model.')
        serve_parser.add_argument('--tokenizer', type=str, help='Tokenizer name to use.')
        serve_parser.add_argument('--device', type=int, default=-1, help='Indicate the device to run onto, -1 indicates CPU, >= 0 indicates GPU (default: -1)')
        serve_parser.set_defaults(func=serve_command_factory)

    def __init__(self, pipeline: Pipeline, host: str, port: int):

        self._pipeline = pipeline

        self._host = host
        self._port = port
        if not _serve_dependancies_installed:
            raise ImportError("Using serve command requires FastAPI and unicorn. "
                                "Please install transformers with [serving]: pip install transformers[serving]." 
                                "Or install FastAPI and unicorn separatly.")
        else:
            logger.info('Serving model over {}:{}'.format(host, port))
            self._app = FastAPI()

            # Register routes
            self._app.add_api_route('/', self.model_info, response_model=ServeModelInfoResult, methods=['GET'])
            self._app.add_api_route('/tokenize', self.tokenize, response_model=ServeTokenizeResult, methods=['POST'])
            self._app.add_api_route('/detokenize', self.detokenize, response_model=ServeDeTokenizeResult, methods=['POST'])
            self._app.add_api_route('/forward', self.forward, response_model=ServeForwardResult, methods=['POST'])

    def run(self):
        run(self._app, host=self._host, port=self._port)

    def model_info(self):
        return ServeModelInfoResult(infos=vars(self._pipeline.model.config))

    def tokenize(self, text_input: str = Body(None, embed=True), return_ids: bool = Body(False, embed=True)):
        """
        Tokenize the provided input and eventually returns corresponding tokens id:
        - **text_input**: String to tokenize
        - **return_ids**: Boolean flags indicating if the tokens have to be converted to their integer mapping.
        """
        try:
            tokens_txt = self._pipeline.tokenizer.tokenize(text_input)

            if return_ids:
                tokens_ids = self._pipeline.tokenizer.convert_tokens_to_ids(tokens_txt)
                return ServeTokenizeResult(tokens=tokens_txt, tokens_ids=tokens_ids)
            else:
                return ServeTokenizeResult(tokens=tokens_txt)

        except Exception as e:
            raise HTTPException(status_code=500, detail={"model": '', "error": str(e)})

    def detokenize(self, tokens_ids: List[int] = Body(None, embed=True),
                   skip_special_tokens: bool = Body(False, embed=True),
                   cleanup_tokenization_spaces: bool = Body(True, embed=True)):
        """
        Detokenize the provided tokens ids to readable text:
        - **tokens_ids**: List of tokens ids
        - **skip_special_tokens**: Flag indicating to not try to decode special tokens
        - **cleanup_tokenization_spaces**: Flag indicating to remove all leading/trailing spaces and intermediate ones.
        """
        try:
            decoded_str = self._pipeline.tokenizer.decode(tokens_ids, skip_special_tokens, cleanup_tokenization_spaces)
            return ServeDeTokenizeResult(model='', text=decoded_str)
        except Exception as e:
            raise HTTPException(status_code=500, detail={"model": '', "error": str(e)})

    def forward(self, inputs: Union[str, dict, List[str], List[int], List[dict]] = Body(None, embed=True)):
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
            output = self._pipeline(inputs)
            return ServeForwardResult(output=output)
        except Exception as e:
            raise HTTPException(500, {"error": str(e)})
