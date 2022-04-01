import json
import multiprocessing
import os
import re
from collections import defaultdict

import torch
from datasets import load_dataset, load_metric
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

import transformers
from accelerate import Accelerator
from arguments import HumanEvalArguments
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    StoppingCriteria,
    StoppingCriteriaList,
    set_seed,
)


EOF_STRINGS = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif"]


class TokenizeDataset(IterableDataset):
    """Tokenize and preprocess the dataset
    Multiple copies of the same prompt are sent sequentially.
    See compute_code for more details.
    """

    def __init__(self, tokenizer, dataset, n_tasks=None, n_copies=1):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.n_tasks = len(dataset) if n_tasks is None else n_tasks
        self.n_copies = n_copies

    def __iter__(self):
        for task in range(self.n_tasks):
            # without strip, the model generate commented codes ...
            prompt = self.tokenizer.eos_token + self.dataset[task]["prompt"].strip()
            # codeparrot model is not robust to padding
            input_ids = self.tokenizer(prompt)["input_ids"]
            for _ in range(self.n_copies):
                yield {"ids": torch.tensor(input_ids), "task_id": task}


class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""

    def __init__(self, start_length, eof_strings, tokenizer):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(input_ids[:, self.start_length :])
        done = []
        for decoded_generation in decoded_generations:
            done.append(any([stop_string in decoded_generation for stop_string in self.eof_strings]))
        return all(done)


def remove_last_block(string):
    """Remove the last block of the code containing EOF_STRINGS"""
    string_list = re.split("(%s)" % "|".join(EOF_STRINGS), string)
    # last string should be ""
    return "".join(string_list[:-2])


def repeat_elements(x, k):
    return [e for e in x for i in range(k)]


def complete_code(accelerator, model, tokenizer, dataloader, batch_size=20, **gen_kwargs):
    """Generate multiple codes for each task in the dataset. This function leverage accelerator to distribute
    the processing to multiple GPUs.
    dataloader, a wrapper around a TokenizeDataset objectm is supposed to send all the prompts from
    the evalution dataset to the modelm as the following:
    [p_0_0, p_0_1, ..., p_0_nc-1, p_1_0, ..., p_nt-1_nc-1]
    where nc is the number of copies of the prompt, and nt is the number of tasks.
    nc is such that num_sample = nc * batch_size
    In this way, the same prompt is sent to the model on all the GPUs to avoid zero padding.
    (accelerate require all the batch having the same shape before feeding it to multiple GPUs)

    Parameters
    ----------
    accelerator: Accelerator

    model: transformers.PreTrainedModel
        Code generation model. AutoTokenizer.from_pretrained(model_ckpt), ex model_ckpt = "lvwerra/codeparrot"

    tokenizer: transformers.AutoTokenizer
        The tokenizer used to train model

    dataloader: DataLoader
        The dataloader is a wrapper around a TokenizeDataset object. It is designed to be used with multiple GPUs.

    batch_size: int
        num_return_sequences per copy of the prompt such that num_sample = batch_size * n_copies

    gen_kwargs: dict
        Keyword arguments for the generation function of the model.

    Returns
    -------
    code_gens_dict: dict of task_num -> list
        Dict of generated codes for each task.
        Each element is a list of generated codes for each task, with length num_samples
    """
    gen_token_dict = defaultdict(list)  # dict of list of generated tokens
    for step, batch in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            gen_kwargs["stopping_criteria"][0].start_length = batch["ids"].shape[-1]
            generated_tokens = accelerator.unwrap_model(model).generate(
                input_ids=batch["ids"], num_return_sequences=batch_size, **gen_kwargs
            )
            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
            generated_tasks = accelerator.gather(batch["task_id"]).cpu().numpy()
            # each task is generated batch_size times
            generated_tasks = repeat_elements(generated_tasks, batch_size)

            for task, generated_tokens in zip(generated_tasks, generated_tokens):
                gen_token_dict[task].append(generated_tokens)

    code_gens_dict = defaultdict(list)
    for task, generated_tokens in gen_token_dict.items():
        for s in generated_tokens:
            gen_code = tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            code_gens_dict[task].append(remove_last_block(gen_code))
    return code_gens_dict


def main():
    # Setup configuration
    parser = HfArgumentParser(HumanEvalArguments)
    args = parser.parse_args()

    transformers.logging.set_verbosity_error()
    # enables code execution in code_eval metric
    os.environ["HF_ALLOW_CODE_EVAL"] = args.HF_ALLOW_CODE_EVAL
    # make sure tokenizer plays nice with multiprocessing
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if args.num_workers is None:
        args.num_workers = multiprocessing.cpu_count()

    set_seed(args.seed)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt)
    model = AutoModelForCausalLM.from_pretrained(args.model_ckpt)

    # Generation settings
    gen_kwargs = {
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "stopping_criteria": StoppingCriteriaList([EndOfFunctionCriteria(0, EOF_STRINGS, tokenizer)]),
    }

    # Load evaluation dataset and metric
    human_eval = load_dataset("openai_humaneval")
    code_eval_metric = load_metric("code_eval")

    # Use dataset load to feed to accelerate
    accelerator = Accelerator()

    n_tasks = args.num_tasks if args.num_tasks is not None else len(human_eval["test"])
    n_copies = args.n_samples // args.batch_size
    if n_copies % accelerator.num_processes != 0:
        raise ValueError(
            f"n_samples({args.n_samples}) should be a mulitple of batch_size({args.batch_size}) x num_processes({accelerator.num_processes})"
        )
    human_eval_td = TokenizeDataset(tokenizer, human_eval["test"], n_copies=n_copies, n_tasks=n_tasks)
    # do not confuse args.batch_size, which is actually the num_return_sequences
    human_eval_loader = DataLoader(human_eval_td, batch_size=1)

    # Run a quick test to see if code evaluation is enabled
    try:
        _ = code_eval_metric.compute(references=[""], predictions=[[""]])
    except ValueError as exception:
        print(
            'Code evaluation not enabled. Read the warning below carefully and then use `--HF_ALLOW_CODE_EVAL="1"` flag to enable code evaluation.'
        )
        raise exception

    model, human_eval_loader = accelerator.prepare(model, human_eval_loader)

    generations = [[] for _ in range(n_tasks)]
    generation_dict = complete_code(
        accelerator, model, tokenizer, human_eval_loader, batch_size=args.batch_size, **gen_kwargs
    )
    for task, gen_list in generation_dict.items():
        generations[task].extend(gen_list)

    if accelerator.is_main_process:
        references = []

        for task in tqdm(range(n_tasks)):
            test_func = human_eval["test"][task]["test"]
            entry_point = f"check({human_eval['test'][task]['entry_point']})"
            references.append("\n" + test_func + "\n" + entry_point)

        # Evaluate completions with "code_eval" metric
        pass_at_k, _ = code_eval_metric.compute(
            references=references, predictions=generations, num_workers=args.num_workers
        )
        print(f"Results: {pass_at_k}")

        # Save results to json file
        with open(args.output_file, "w") as fp:
            json.dump(pass_at_k, fp)


# For some reason the folliwng seems to be necessary sometimes for code_eval to work nice with multiprocessing
# https://stackoverflow.com/questions/60804599/python-multiprocessing-keeps-spawning-the-whole-script
if __name__ == "__main__":
    main()
