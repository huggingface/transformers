from argparse import ArgumentParser, Namespace
from typing import List, Optional, Union, Any

import torch
from fastapi import FastAPI, HTTPException, Body
from logging import getLogger

from pydantic import BaseModel
from uvicorn import run

from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers.commands import BaseTransformersCLICommand


def serve_command_factory(args: Namespace):
    """
    Factory function used to instantiate serving server from provided command line arguments.
    :return: ServeCommand
    """
    return ServeCommand(args.host, args.port, args.model, args.graphql)


class ServeResult(BaseModel):
    """
    Base class for serving result
    """
    model: str


class ServeModelInfoResult(ServeResult):
    """
    Expose model information
    """
    infos: dict


class ServeTokenizeResult(ServeResult):
    """
    Tokenize result model
    """
    tokens: List[str]
    tokens_ids: Optional[List[int]]


class ServeDeTokenizeResult(ServeResult):
    """
    DeTokenize result model
    """
    text: str


class ServeForwardResult(ServeResult):
    """
    Forward result model
    """
    tokens: List[str]
    tokens_ids: List[int]
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
        serve_parser.add_argument('--host', type=str, default='localhost', help='Interface the server will listen on.')
        serve_parser.add_argument('--port', type=int, default=8888, help='Port the serving will listen to.')
        serve_parser.add_argument('--model', type=str, required=True, help='Model\'s name or path to stored model to infer from.')
        serve_parser.add_argument('--graphql', action='store_true', default=False, help='Enable GraphQL endpoints.')
        serve_parser.set_defaults(func=serve_command_factory)

    def __init__(self, host: str, port: int, model: str, graphql: bool):
        self._logger = getLogger('transformers-cli/serving')

        self._logger.info('Loading model {}'.format(model))
        self._model_name = model
        self._model = AutoModel.from_pretrained(model)
        self._tokenizer = AutoTokenizer.from_pretrained(model)

        self._logger.info('Serving model over {}:{}'.format(host, port))
        self._host = host
        self._port = port
        self._app = FastAPI()

        # Register routes
        self._app.add_api_route('/', self.model_info, response_model=ServeModelInfoResult, methods=['GET'])
        self._app.add_api_route('/tokenize', self.tokenize, response_model=ServeTokenizeResult, methods=['POST'])
        self._app.add_api_route('/detokenize', self.detokenize, response_model=ServeDeTokenizeResult, methods=['POST'])
        self._app.add_api_route('/forward', self.forward, response_model=ServeForwardResult, methods=['POST'])

    def run(self):
        run(self._app, host=self._host, port=self._port)

    def model_info(self):
        return ServeModelInfoResult(model=self._model_name, infos=vars(self._model.config))

    def tokenize(self, text_input: str = Body(None, embed=True), return_ids: bool = Body(False, embed=True)):
        """
        Tokenize the provided input and eventually returns corresponding tokens id:
        - **text_input**: String to tokenize
        - **return_ids**: Boolean flags indicating if the tokens have to be converted to their integer mapping.
        """
        try:
            tokens_txt = self._tokenizer.tokenize(text_input)

            if return_ids:
                tokens_ids = self._tokenizer.convert_tokens_to_ids(tokens_txt)
                return ServeTokenizeResult(model=self._model_name, tokens=tokens_txt, tokens_ids=tokens_ids)
            else:
                return ServeTokenizeResult(model=self._model_name, tokens=tokens_txt)

        except Exception as e:
            raise HTTPException(status_code=500, detail={"model": self._model_name, "error": str(e)})

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
            decoded_str = self._tokenizer.decode(tokens_ids, skip_special_tokens, cleanup_tokenization_spaces)
            return ServeDeTokenizeResult(model=self._model_name, text=decoded_str)
        except Exception as e:
            raise HTTPException(status_code=500, detail={"model": self._model_name, "error": str(e)})

    def forward(self, inputs: Union[str, List[str], List[int]] = Body(None, embed=True),
                attention_mask: Optional[List[int]] = Body(None, embed=True),
                tokens_type_ids: Optional[List[int]] = Body(None, embed=True)):
        """
        **inputs**:
        **attention_mask**:
        **tokens_type_ids**:
        """

        # Check we don't have empty string
        if len(inputs) == 0:
            return ServeForwardResult(model=self._model_name, output=[], attention=[])

        if isinstance(inputs, str):
            inputs_tokens = self._tokenizer.tokenize(inputs)
            inputs_ids = self._tokenizer.convert_tokens_to_ids(inputs_tokens)

        elif isinstance(inputs, List):
            if isinstance(inputs[0], str):
                inputs_tokens = inputs
                inputs_ids = self._tokenizer.convert_tokens_to_ids(inputs_tokens)
            elif isinstance(inputs[0], int):
                inputs_tokens = []
                inputs_ids = inputs
            else:
                error_msg = "inputs should be string, [str] of [int] (got {})".format(type(inputs[0]))
                raise HTTPException(423, detail={"error": error_msg})
        else:
            error_msg = "inputs should be string, [str] of [int] (got {})".format(type(inputs))
            raise HTTPException(423, detail={"error": error_msg})

        try:
            # Forward through the model
            t_input_ids = torch.tensor(inputs_ids).unsqueeze(0)
            output = self._model(t_input_ids, attention_mask, tokens_type_ids)

            return ServeForwardResult(
                model=self._model_name, tokens=inputs_tokens,
                tokens_ids=inputs_ids, output=output[0].tolist()
            )
        except Exception as e:
            raise HTTPException(500, {"error": str(e)})
