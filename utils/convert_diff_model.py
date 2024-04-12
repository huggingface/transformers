import regex as re
import inspect
# pattern = re.compile(r'(class|def|XXXConverter\.register)\s+[\w.()]+\s*:(\s*(?:[^class|def|XXXConverter\.register]|\n)+)', re.MULTILINE)
# For each and every diff files we should import all packages from the modules that are imported.
# pattern = r"((    [\s\S]*?)\n\n(?=    \S))|((    [\s\S]*?)(?=\Z))" is super important

# TODO in order to get everything from LLAMA we need to copy each line from Llama
# only updating the attention classes. 
# Need to keep the order correct

from transformers.models.llama.modeling_llama import *
from transformers.models.cohere.diff_cohere import *

# 1. all the imports from the original file should be copied until end of header? __HEADER__
# with open(CohereConverter.original_file, 'r') as file, open("result.py", "w+") as modeling:
#         pass
# TODO also copy and import from all modules in CohereConverter.modules_to_import to be able to use inspect

# 2. Write all the classes. Use the `CohereConverter` class for this.
with open(CohereConverter.diff_file, 'r') as file, open("result.py", "w+") as modeling:
    function_set = {}
    for line in file: 
            if "Converter.register" in line: # TODO use map() to map lines to this 
                # write the code of the original model
                class_to_use, old_class = re.search(r'Converter\.register\(\"(.*?)\", (.*?)\)', line).groups()
                model_identifier_camel = re.findall(r'[A-Z][a-z0-9]*', class_to_use)[0]
                old_model_identifier_camel = re.findall(r'[A-Z][a-z0-9]*', old_class)[0]
                source_code = inspect.getsource(CohereConverter.registered_classes[class_to_use]).replace(old_class, class_to_use)
                source_code = source_code.replace(old_model_identifier_camel, model_identifier_camel)
                modeling.write(source_code)
            elif match:=re.match(r"class (\w+)\((\w+)\):", line):
                class_name, parent_class = match.groups()
                pattern = re.compile(r"((    [\s\S]*?)\n(\n)?(?=    \S))|((    [\s\S]*?)(?=\Z))", re.MULTILINE)

                parent_class_def = inspect.getsource(eval(parent_class))
                modeling.write(parent_class_def.split('\n')[0].replace(parent_class,class_name)+"\n")
            
                matches = pattern.finditer(parent_class_def)
                function_set = {}
                for match in matches:
                    full_function = match.group()
                    function_set[full_function.split("(")[0]] = full_function

                class_def = inspect.getsource(eval(class_name))
                matches = pattern.finditer(class_def)
                for match in matches:
                        full_function = match.group()
                        function_set[full_function.split("(")[0]] = full_function

                modeling.write("".join(function_set.values())) # TODO we wrote the code, next lines shall be ignored
            elif line not in "".join(function_set.values()) or line=="\n":
                modeling.write(line) 

# 3. Apply ruff fix to remove unused imports
# 4. Run a tiny test to import from this new file.