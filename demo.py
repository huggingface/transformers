import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers.generation.logits_process import GrammarConstrainedLogitsProcessor
from transformers.generation.grammar_utils import IncrementalGrammarAcceptor

import logging
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    torch.manual_seed(2)

    # set logging level
    logger.setLevel(logging.DEBUG)
    # logging.getLogger("transformers").setLevel(logging.DEBUG)

    # Initialize tokenizer and grammar
    # tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    with open("examples/grammars/json.gbnf", "r") as file:
        grammar_str = file.read()
    grammar = IncrementalGrammarAcceptor(grammar_str, "root", tokenizer)


    # Initialize ids and logits_processor
    # input_ids = [[1]]
    # input_ids = tokenizer(["{ 2 : [ 3 ] , 4 : [ 5 ] }"],add_special_tokens=False)["input_ids"]

    # prefix="(1+3)*(2+43"
    # prefix = '{ "a": 1'
    prefix1= "This is a valid json string for email:"
    prefix2= "This is a valid json string for shopping cart:"
    input_ids = tokenizer([prefix1, prefix2],add_special_tokens=False, return_tensors="pt", padding=True)["input_ids"]
    logits_processor = GrammarConstrainedLogitsProcessor(grammar, batch_size=2, num_beams=1)


    max_new_tokens = 30

    # Generate tokens
    for i in range(max_new_tokens):
        logits = model(input_ids)[0][:, -1, :]
        logits = logits_processor.process_logits(input_ids, logits)
        new_token_ids = torch.argmax(logits, dim=-1)
        # concat new token ids to input_ids
        input_ids = torch.cat((input_ids, new_token_ids.unsqueeze(1)), dim=-1)

    generations = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    print(generations)
    """
    'This is a valid json string for email:{ "title": "Theory", "text": "Theory", "type": "text", "text": "Theory", "type',
    'This is a valid json string for shopping cart:{ "name": "MyCart", "price": "10", "price": "10", "price": "10", "price": "'
    """