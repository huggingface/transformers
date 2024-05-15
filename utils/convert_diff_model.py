import importlib
import argparse
import glob
from transformers import MODEL_NAMES_MAPPING
import regex as re
import inspect
# pattern = re.compile(r'(class|def|XXXConverter\.register)\s+[\w.()]+\s*:(\s*(?:[^class|def|XXXConverter\.register]|\n)+)', re.MULTILINE)
# For each and every diff files we should import all packages from the modules that are imported.
# pattern = r"((    [\s\S]*?)\n\n(?=    \S))|((    [\s\S]*?)(?=\Z))" is super important

# TODO in order to get everything from LLAMA we need to copy each line from Llama
# only updating the attention classes. 
# Need to keep the order correct
# TODO the imports here are not correctly done. We should dynamically import what we need
APACHE_LICENCE = """# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
# 1. all the imports from the original file should be copied until end of header? __HEADER__
# with open(CohereConverter.original_file, 'r') as file, open("result.py", "w+") as modeling:
#         pass
# TODO also copy and import from all modules in CohereConverter.modules_to_import to be able to use inspect
def replace_super_calls_in_method(method_body, parent_method_body, method_name):
    # Indent parent method body to match the child's method indentation
    indent = re.match(r'(\s*)def ', method_body).group(1)
    indented_parent_method_body = "\n".join([indent + line if line.strip() else line for line in parent_method_body.split('\n')])
    method_name = method_name.strip()
    # Handle super().method_name(args) and return super().method_name(args)
    super_call_pattern = re.compile(r'(\s*)return super\(\)\.' + method_name + r'\((.*?)\)')
    method_body = super_call_pattern.sub(r'\1return (\2\n' + indented_parent_method_body + r'\1)', method_body)

    super_call_pattern_no_return = re.compile(r'(\s*)super\(\)\.' + method_name + r'\((.*?)\)')
    method_body = super_call_pattern_no_return.sub(r'\1\2\n' + indented_parent_method_body, method_body)

    return method_body
# 2. Write all the classes. Use the `CohereConverter` class for this.
def create_single_model_file(converter):
    if hasattr(converter, "diff_file"):
        model_identifier = converter.diff_file.split("diff_")
        # temporarily add the source to the path in order to load everything?
        # 1. Import all modules from the registered classes
        modules = set([ _class.__module__ for _class in converter.registered_classes.values()]) or set()
        for module in modules | {re.sub(r'.*src/(.*)\.py', r'\1', converter.diff_file).replace('/', '.')}:
            modeling_ = importlib.import_module(module)
            globals().update({k: getattr(modeling_, k) for k in modeling_.__dict__.keys()})

        with open(converter.diff_file, 'r') as file, open(f"{model_identifier[0]}modeling_{model_identifier[1]}", "w+") as modeling:
            modeling.write(APACHE_LICENCE)
            function_set = {}
            for line in file:
                    if "Converter.register" in line: # TODO use map() to map lines to this
                        # write the code of the original model
                        class_to_use, old_class = re.search(r'Converter\.register\(\"(.*?)\", (.*?)\)', line).groups()
                        model_identifier_camel = re.findall(r'[A-Z][a-z0-9]*', class_to_use)[0]
                        old_model_identifier_camel = re.findall(r'[A-Z][a-z0-9]*', old_class)[0]
                        # import all necessary modules from the path:

                        source_code = inspect.getsource(converter.registered_classes[class_to_use]).replace(old_class, class_to_use)
                        source_code = source_code.replace(old_model_identifier_camel, model_identifier_camel)
                        modeling.write(source_code)
                        modeling.write("\n")

                    elif match:=re.match(r"class (\w+)\((\w+)\):", line):
                        class_name, parent_class = match.groups()
                        pattern = re.compile( r"(\ {4}(?:[\S\s\ \n]*?)(?=\n\ ^[\) ]|\n\n    (?:def|@)|\Z))", re.MULTILINE)
                        parent_class_def = inspect.getsource(eval(parent_class))
                        modeling.write(parent_class_def.split('\n')[0].replace(parent_class,class_name)+"\n")

                        function_name_pattern = r"(?=    def ([\S]*)\()"
                        function_body_pattern = r"(\ {4}(?:[\S\s\ \n]*?)(?=\n\ ^[\) ]|\n\n    (?:def|@)|\Z))"

                        pattern = re.compile(function_body_pattern)
                        matches = pattern.finditer(parent_class_def)
                        parent_function_set = {}
                        for match in matches:
                            full_function = match.group()
                            if "def" in full_function:
                                parent_function_set[full_function.split("def")[1].split("(")[0]] = full_function
                            else:
                                parent_function_set[full_function] = full_function

                        parent_identifier_camel = re.findall(r'[A-Z][a-z0-9]*', parent_class)[0]
                        child_identifier_camel = re.findall(r'[A-Z][a-z0-9]*', class_name)[0]
                        print(f"`{class_name}` -> `{parent_class}`")

                        child_function_set = parent_function_set.copy()
                        class_def = inspect.getsource(eval(class_name))
                        matches = pattern.finditer(class_def)
                        for match in matches:
                            # TODO handle call to super!
                            full_function = match.group()
                            if "def" in full_function:
                                function_name = full_function.split("def")[1].split("(")[0]

                                if (f"super()." in full_function or f"return super()." in full_function) and parent_identifier_camel != child_identifier_camel:
                                    replaced_function = replace_super_calls_in_method(full_function,
                                                                                      parent_function_set.get(function_name,
                                                                                                              ""),
                                                                                      function_name)
                                    child_function_set[function_name] = replaced_function
                                else:
                                    child_function_set[function_name] = full_function
                            else:
                                child_function_set[full_function] = full_function

                        modeling.write("\n".join(child_function_set.values())) # TODO we wrote the code, next lines shall be ignored
                        modeling.write("\n")

                    elif "= ModelConverter(__file__)" in line:
                        pass # don't write the converter to the result file
                    elif line not in "".join(function_set.values()) or line=="\n":
                        modeling.write(line)


def dynamically_import_object(module_path, object_name):
    try:
        module = importlib.import_module(module_path)
        obj = getattr(module, object_name)
        return obj
    except (ImportError, AttributeError) as e:
        print(f"Failed to import object '{object_name}' from module '{module_path}'")
        print(e)


# 3. Apply ruff fix to remove unused imports
# 4. Run a tiny test to import from this new file.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--files_to_parse", default="all", help="A list of `diff_xxxx` files that should be converted to single model file")
    args = parser.parse_args()
    if args.files_to_parse == "all":
        args.files_to_parse = glob.glob("src/transformers/models/**/diff_*.py", recursive=True)
    for file_name in args.files_to_parse:
        print(f"Converting {file_name} to a single model single file format")
        module_path = file_name.replace("/",".").replace(".py","").replace("src.","")
        model_name = MODEL_NAMES_MAPPING[module_path.split('_')[-1]]
        converter = dynamically_import_object(module_path, f"{model_name}Converter")
        create_single_model_file(converter)