# What are the steps in conversion?

# 1) Convert all functions and classes in the model file
# 2) Add the new classes to the module __init__
# 3) Add the new classes to the module autodoc
# 4) Update the root __init__ to import the new classes
# 5) Add the missing Auto classes
# 6) Add the module file to the doctest list
#   - Happens automatically: Update the model support checklist
#   - Happens automatically: Update the dummies

# TODO 1: Get GPT to generate the input shapes when it converts a module (not at top-level class!)
# TODO 2: Port weights from PT to TF versions and do equivalence testing
# TODO 3:

from pathlib import Path
import re
import openai
from time import sleep
from tqdm import tqdm
from argparse import ArgumentParser

def get_module_name(module_text: str):
    module_name = re.search(r"(?:class |def )(\w+)", module_text).group(1)
    return module_name

def translate_fn(module_text: str):
    system_text = """
    You are a translation bot designed to translate code in the Hugging Face Transformers library from PyTorch to TensorFlow.
    
    You will be passed a PyTorch class from a tests file. Your goal is to translate this class to test the equivalent TensorFlow model.
    
    There are some guidelines you should follow when translating the code:
    
    - Most model classes should be renamed by adding "TF" at the start of their name. However, tokenizers, processors and config classes are universal and so not renamed.
    - You don't need to add any extra imports, you can assume that any other functions or classes you call will be imported for you.
    - If the class calls other classes in the same module, you can assume that these have already been converted. Please add "TF" to the start of their name if required.
    - You can use tensor.shape.rank as a TensorFlow replacement for tensor.ndim.
    - Please use the Hugging Face function shape_list(), which returns a list, instead of tensor.shape or tf.shape(tensor) unless you need to treat the output as a tensor.
    - PyTorch methods like .detach(), .eval() and .to(device) are not needed in TensorFlow. 
    - In tests, it's completely acceptable to construct arrays in NumPy instead of TensorFlow.
    - HuggingFace methods like ids_tensor, floats_tensor and random_attention_mask are universal, so you can use them without changes.
    - If the test looks like it should work without changes, please simply repeat it back without changes.
    """
    module_name = get_module_name(module_text)
    if "load_tf_weights" in module_name:
        print("Skipping", module_name)
        return ""
    prompt = [{"role": "system", "content": system_text}, {"role": "user", "content": module_text}]
    for i in range(5):
        try:
            response = openai.ChatCompletion.create(model="gpt-4", messages=prompt, temperature=0, stream=True)
            break
        except openai.error.RateLimitError:
            print(f"Rate limited, retrying ({i + 1} of 5)")
            sleep(15)
    else:
        raise RuntimeError("Rate limited too many times")
    chunks = []
    for chunk in tqdm(response, desc=f"Translating {module_name}", dynamic_ncols=True, unit=" tokens"):
        chunk_message = chunk['choices'][0]['delta']
        chunks.append(chunk_message)
    translated_function = ''.join([m.get('content', '') for m in chunks])
    return translated_function


def split_file(source_file: Path):
    text = source_file.read_text()
    top_level_fns = list(re.finditer(r"\n\n((?:@|class |def ).*?)(?=\n\n@|\n\nclass |\n\ndef |$)", text, flags=re.DOTALL))
    for i in range(len(top_level_fns) - 1):
        assert top_level_fns[i].end() == top_level_fns[i + 1].start()
    preamble = text[:top_level_fns[0].start()]
    all_texts = [preamble] + [m.group(0) for m in top_level_fns]
    for i in range(len(all_texts) - 1):
        class_text = all_texts[i]
        if len(class_text) > 8000:
            # Probably a big test class - let's extract methods individually
            # Remember they're each going to be indented by 4 spaces!
            methods = list(re.finditer(r"\n    (def .*?)(?=\n    def |$)", class_text, flags=re.DOTALL))
            method_preamble = class_text[:methods[0].start()]
            all_class_texts = [method_preamble] + [m.group(0) for m in methods]
            all_texts[i] = all_class_texts
        else:
            all_texts[i] = class_text.strip()

    return all_texts


def main():
    path = Path("tests/models/gpt_neo/test_modeling_gpt_neo.py")
    out_path = Path("tests/models/gpt_neo/test_modeling_tf_gpt_neo.py")
    split_classes = split_file(path)
    translated_classes = [split_classes[0]]
    for split_class in split_classes[1:]:
        if isinstance(split_class, list):
            partial_translation = [split_class[0]]
            for method in split_class[1:]:
                partial_translation.append(translate_fn(method))
            translated_classes.append('\n'.join(partial_translation))
        else:
            translated_classes.append(translate_fn(split_class))
    output = '\n'.join(translated_classes)
    out_path.write_text(output)


if __name__ == '__main__':
    main()
