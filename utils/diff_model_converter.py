import argparse
import glob
import importlib
import re

import libcst as cst
from check_copies import fix_ruff
from libcst import ClassDef, CSTTransformer, CSTVisitor
from libcst import matchers as m
from libcst.metadata import MetadataWrapper, ParentNodeProvider, PositionProvider, ScopeProvider

from transformers import logging

logger = logging.get_logger(__name__)

def get_module_source_from_name(module_name: str) -> str:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        return f"Module {module_name} not found"

    with open(spec.origin, "r") as file:
        source_code = file.read()

    return source_code


class ClassFinder(CSTVisitor):
    """A visitor class which analyses a module, creating a mapping of dependencies between classes and functions.
    For example if the visited code has
    ```python3
    def init_value(): return 1

    class LlamaModel(PreTrainedModel):
        def __init__(self):
            super().__init__(self)
            self.value = init_value()
    ```
    then the `class_dependency_mapping` should be: `{"LlamaModel":{"PreTrainedModel":{},"init_value":{}}, "init_value":{}}
    """

    METADATA_DEPENDENCIES = (ParentNodeProvider, ScopeProvider, PositionProvider)

    def __init__(self, python_module):
        self.python_module = python_module
        self.classes = {}  # class LlamaAttentino
        self.imports = {}  # from flash_attn import
        self.function_def = {}  # def repeat_kv
        self.assignments = {}  # LLAMA_DOCSTRING
        self.protected_imports = {}  # if is_xxx_available()
        self.class_dependency_mapping = {}  # "LlamaModel":["LlamaDecoderLayer, "LlamaRMSNorm", "LlamaPreTrainedModel"], "LlamaDecoderLayer":["LlamaAttention","Llama"]

    def visit_ClassDef(self, node: ClassDef) -> None:
        self.classes[node.name.value] = node
        for k in node.bases:  # deal with inheritance
            name = self.python_module.code_for_node(k)
            self.class_dependency_mapping.update(
                {
                    node.name.value: set(self.class_dependency_mapping.get(name, {name}))
                    | self.class_dependency_mapping.get(node.name.value, set())
                }
            )

    def visit_SimpleStatementLine(self, node):
        match node:
            case cst.SimpleStatementLine(body=[cst.Assign(targets=[_], value=_)]):
                if m.matches(self.get_metadata(cst.metadata.ParentNodeProvider, node), m.Module()):
                    self.assignments[node.body[0].targets[0].target.value] = node
            case cst.SimpleStatementLine(body=[cst.Import(names=[_])]):
                self.imports[node.body[0].names] = node
            case cst.SimpleStatementLine(body=[cst.ImportFrom(_)]):
                self.imports[node.body[0].names] = node

    def visit_FunctionDef(self, node):
        parent_node = self.get_metadata(cst.metadata.ParentNodeProvider, node)
        if m.matches(parent_node, m.Module()):
            self.function_def[node.name.value] = node

    def leave_If(self, node):
        for stmt in node.body.body:
            if m.matches(stmt, m.SimpleStatementLine(body=[m.ImportFrom() | m.Import()])):
                self.imports[stmt.body[0].names] = node  # match the visit simple statement line to overwrite it

    def leave_Name(self, node):
        if node.value in self.classes.keys() | self.assignments.keys() | self.function_def.keys():
            dad = self.get_metadata(cst.metadata.ScopeProvider, node)
            if not isinstance(dad, cst.metadata.scope_provider.GlobalScope):
                logger.info(f"Name:\t\t{node.value:<45} called in {dad._name_prefix}")
                name = dad._name_prefix.split(".")[0]
                dep = set(self.class_dependency_mapping.get(node.value, set()))
                dep |= set(self.class_dependency_mapping.get(name, {})) | set({node.value})
                self.class_dependency_mapping[name] = dep

    def leave_Arg(self, node):
        if m.matches(node.value, m.Name()):
            dad = self.get_metadata(ParentNodeProvider, node)
            if m.matches(dad, m.ClassDef()) and dad.bases:
                logger.info(f"Arg:\t\t{node.value.value:<45} called in {dad.name.value}")
                name = dad.name.value
                dep = set(self.class_dependency_mapping.get(node.value.value, set()))
                dep |= set(self.class_dependency_mapping.get(name, {})) | set({node.value.value})
                self.class_dependency_mapping[name] = dep

    def leave_Dict(self, node):
        dad = self.get_metadata(cst.metadata.ParentNodeProvider, node)
        if m.matches(dad, m.Assign(targets=[m.AssignTarget()])):
            name = dad.targets[0].target.value
            if name in self.assignments:
                for k in node.elements:
                    if k.value.value in self.classes:
                        dep = set(self.class_dependency_mapping.get(k.value.value, set()))
                        dep |= self.class_dependency_mapping.get(name, set()) | set({k.value.value})
                        self.class_dependency_mapping[name] = dep
                        logger.info(f"Dict:\t\t{k.value.value:<45} called in {name}")

    # Decorator: handle in leave_FunctionDef and leave_ClassDef instead
    def leave_Decorator(self, node):
        if hasattr(node.decorator, "args"):
            for k in node.decorator.args:
                if k.value.value in self.assignments:
                    dad = self.get_metadata(cst.metadata.ParentNodeProvider, node)
                    scope = self.get_metadata(cst.metadata.ScopeProvider, node)
                    if scope._name_prefix != "":
                        name = scope._name_prefix.split(".")[0]
                    else:
                        name = dad.name.value
                    logger.info(f"Decorator:\t{k.value.value:<45} called in {name}")
                    dep = set(self.class_dependency_mapping.get(k.value.value, set()))
                    dep |= self.class_dependency_mapping.get(name, set()) | set({k.value.value})
                    self.class_dependency_mapping[name] = dep

    def leave_Module(self, node):
        self.global_nodes = {**self.assignments, **self.classes, **self.function_def}
        # now sort the class dependency_mapping based on the position of the nodes
        self.class_start_line = {}
        for id, node in self.global_nodes.items():
            self.class_start_line[id] = self.get_metadata(cst.metadata.PositionProvider, node).start.line


class ReplaceNameTransformer(m.MatcherDecoratableTransformer):
    """A transformer that replaces `old_name` with `new_name` in comments, string and any references"""

    def __init__(self, old_name, new_name):
        super().__init__()
        self.new_name = new_name
        self.old_name = old_name
        self.regex = re.compile(re.escape(self.old_name), re.IGNORECASE)

    def preserve_case_replace(self, text):
        def replace(match):
            word = match.group()
            if word.isupper():
                return self.new_name.upper()
            elif word.istitle():
                return self.new_name.title()
            elif word.islower():
                return self.new_name.lower()
            else:
                return self.new_name.title()

        return self.regex.sub(replace, text)

    @m.leave(m.Name() | m.SimpleString() | m.Comment())
    def replace_name(self, original_node, updated_node):
        update = self.preserve_case_replace(updated_node.value)
        return updated_node.with_changes(value=update)


def find_classes_in_file(module, old_id="llama", new_id="gemma"):
    """Helper function to rename and then parse a source file using the ClassFinder"""
    transformer = ReplaceNameTransformer(old_id, new_id)
    new_module = module.visit(transformer)

    wrapper = MetadataWrapper(new_module)

    class_finder = ClassFinder(new_module)
    wrapper.visit(class_finder)
    return class_finder


class SuperTransformer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (ParentNodeProvider,)

    def __init__(self, python_module, original_methods, updated_methods):
        self.python_module = python_module
        self.original_methods = original_methods
        self.updated_methods = updated_methods

    def update_body(self, existing_body, new_statements):
        """
        Helper method to update the body by removing duplicates before adding new statements.
        """
        de_duplicated_new_body = []
        existing_nodes = {
            self.python_module.code_for_node(node).strip() for node in new_statements if isinstance(node, cst.CSTNode)
        }
        for stmt in existing_body:
            if self.python_module.code_for_node(stmt).strip() not in existing_nodes:
                de_duplicated_new_body.append(stmt)
                existing_nodes.add(stmt)
            else:
                logger.info(f"\n{30*'#'}found duplicate{self.python_module.code_for_node(stmt)}{30*'#'}")
        return de_duplicated_new_body

    def replace_super_calls(self, node: cst.IndentedBlock, func_name: str) -> cst.CSTNode:
        new_body = []
        for expr in node.body:
            if m.matches(
                expr,
                m.SimpleStatementLine(
                    body=[
                        m.Expr(
                            value=m.Call(func=m.Attribute(value=m.Call(func=m.Name("super")), attr=m.Name(func_name)))
                        )
                    ]
                ),
            ):
                # Replace the SimpleStatementLine containing super().__init__() with the new body from func_to_body_mapping
                new_body.extend(self.update_body(self.original_methods[func_name].body.body, node.body))
            elif m.matches(
                expr,
                m.SimpleStatementLine(
                    body=[
                        m.Return(
                            value=m.Call(func=m.Attribute(value=m.Call(func=m.Name("super")), attr=m.Name(func_name)))
                        )
                    ]
                ),
            ):
                new_body.extend(self.update_body(self.original_methods[func_name].body.body, node.body))
            else:
                new_body.append(expr)
        return node.with_changes(body=new_body)

    def leave_FunctionDef(self, original_node: cst.Call, updated_node: cst.Call) -> cst.CSTNode:
        if updated_node.name.value in self.updated_methods:
            name = updated_node.name.value
            new_body = self.replace_super_calls(updated_node.body, name)
            return updated_node.with_changes(body=new_body)
        return updated_node

    def leave_Return(self, original_node: cst.Return, updated_node: cst.Return) -> cst.CSTNode:
        if m.matches(updated_node.value, m.Call(func=m.Attribute(attr=m.Name("super")))):
            func_def = self.get_metadata(ParentNodeProvider, original_node)
            if m.matched(func_def, m.FunctionDef()) and func_def.name.value in self.original_methods:
                updated_return_value = updated_node.value.with_changes(
                    args=[
                        cst.Arg(
                            value=cst.Call(func=cst.Name("super"), args=[cst.Arg(value=cst.Name(func_def.name.value))])
                        )
                    ]
                )
                return updated_node.with_changes(value=updated_return_value)
        return updated_node


def replace_call_to_super(class_finder: ClassFinder, updated_node: cst.ClassDef, class_name: str):
    original_node = class_finder.classes[class_name]
    original_methods = {f.name.value: f for f in original_node.body.body if m.matches(f, m.FunctionDef())}

    # Copy methods from original node to replacement node, preserving decorators
    updated_methods = {f.name.value: f for f in updated_node.body.body if m.matches(f, m.FunctionDef())}
    end_meth = []
    for name, func in original_methods.items():
        if name in updated_methods:
            # Replace the method in the replacement class, preserving decorators
            func = func.with_changes(body=updated_methods[name].body)
        end_meth.append(func)

    result_node = original_node.with_changes(body=cst.IndentedBlock(body=end_meth))
    temp_module = cst.Module(body=[result_node])
    new_module = MetadataWrapper(temp_module)
    new_replacement_class = new_module.visit(SuperTransformer(temp_module, original_methods, updated_methods)).body[
        0
    ]  # get the indented block

    return original_node.with_changes(body=new_replacement_class.body)


class DiffConverterTransformer(CSTTransformer):
    METADATA_DEPENDENCIES = (ParentNodeProvider, ScopeProvider, PositionProvider)

    def __init__(self, python_module):
        super().__init__()
        self.python_module = python_module  # we store the original module to use `code_for_node`
        self.transformers_imports = {}      # all the imports made from transformers.models.xxx
        self.imported_mapping = {}          # stores the name of the imported classes, with their source {"LlamaModel":"transformers.model.llama.modeling_llama"}
        self.node_mapping = {}              # stores the name of the nodes that were added to the `new_body`
        self.visited_module = {}            # modules visited like "transformers.models.llama.modeling_llama"
        self.new_body = []                  # store the new body, all global scope nodes should be added here
        self.inserted_deps = []             # nodes inserted via super dependency

    def leave_FunctionDef(self, original_node, node):
        parent_node = self.get_metadata(cst.metadata.ParentNodeProvider, original_node)
        if m.matches(parent_node, m.Module()):
            self.new_body.append(node)
        return node

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        """ When visiting imports from `transformers.models.xxx` we need to:
        1. Get the original source code 
        2. Parse it into an AST Tree
        3. Add this import to `self.transformers_imports` as visited to not parse it twice

        """
        if m.matches(node.module, m.Attribute()):
            import_statement = self.python_module.code_for_node(node.module)
            for imported_ in node.names:
                if re.search(r"transformers\.models\..*\.(modeling|configuration)_.*", import_statement):
                    if import_statement not in self.transformers_imports:
                        source_code = get_module_source_from_name(import_statement)
                        tree = cst.parse_module(source_code)
                        self.transformers_imports[import_statement] = tree
                    imported_class = self.python_module.code_for_node(imported_.name)
                    self.imported_mapping[imported_class] = import_statement

    def leave_SimpleStatementLine(self, original_node: cst.Assign, updated_node: cst.CSTNode):
        if m.matches(original_node, m.SimpleStatementLine(body=[m.Assign()])):
            assign = self.python_module.code_for_node(original_node.body[0])
            node = original_node.body[0]
            if m.matches(node.value, m.Name()) and assign in self.node_mapping:
                return self.node_mapping[assign]

        # remove all relative imports made in the diff file
        full_statement = self.python_module.code_for_node(updated_node.body[0])
        if m.matches(updated_node, m.SimpleStatementLine(body=[m.ImportFrom()])):
            full_statement = self.python_module.code_for_node(updated_node.body[0].module)
            if re.search(r"transformers\.models\..*\.(modeling|configuration)_.*", full_statement):
                return cst.RemoveFromParent()

        parent_node = self.get_metadata(cst.metadata.ParentNodeProvider, original_node)
        if m.matches(parent_node, m.Module()):
            self.new_body.append(updated_node)
        return updated_node

    def leave_ClassDef(self, original_node, updated_node):
        """
        1. Filter the `base` classes of this class
        If they are from `transformers.models.xx` then:
        - take the AST tree of the module it comes from and parse it with a `ClassFinder`.
        - rename all every instance of `old_name` (llama) to `new_name` (gemma)
        2. We insert the modules which the inherited base depends on. This has to be done in 
        the order of the dependencies. If on is already in the new_body (because it's defined in the diff file)
        then we remove it from the new body to add it again in the correct order.
        3. Replace the calls to `super().xxxx` merging parent code
        """
        class_name = original_node.name.value
        bases = [k.value.value for k in original_node.bases if k.value.value in self.imported_mapping]

        for super_class in bases:
            old_name = re.findall(r"[A-Z][a-z0-9]*", super_class)[0].lower()
            if super_class not in self.imported_mapping:
                raise ImportError(
                    f"{super_class} was not imported using `from transformers.models.{old_name}.modeling_{old_name} import {super_class}"
                )

            super_file_name = self.imported_mapping[super_class]  # we need to get the parsed tree
            new_name = re.findall(r"[A-Z][a-z0-9]*", class_name)[0].lower()
            if super_file_name not in self.visited_module:  # only extract classes once
                class_finder = find_classes_in_file(self.transformers_imports[super_file_name], old_name, new_name)
                self.visited_module[super_file_name] = class_finder
                self.node_mapping[class_name] = class_finder.classes[class_name] # here we get the new node form the parent class
            else: # we are re-using the previously parsed data
                class_finder = self.visited_module[super_file_name]

            list_dependencies = {
                dep: class_finder.class_start_line.get(dep, 1000)
                for dep in class_finder.class_dependency_mapping[class_name]
            }

            list_dependencies = sorted(list_dependencies.items(), key=lambda x: x[1])
            for dependency, expected_pos in list_dependencies:
                node = class_finder.global_nodes.get(dependency, None)
                # make sure the class is not re-defined by the diff file
                if node is not None and node not in self.new_body:
                    if dependency not in self.node_mapping:
                        self.new_body.append(node)
                        self.node_mapping[dependency] = node
                    elif dependency not in self.inserted_deps:
                        # make sure the node is written after it's dependencies
                        node = self.node_mapping[dependency]

                        self.new_body.remove(node)
                        self.new_body.append(node)
                        self.inserted_deps.append(dependency)
            updated_node = replace_call_to_super(class_finder, updated_node, class_name)

        self.node_mapping[class_name] = updated_node
        if "Config" in class_name:
            self.config_body = [updated_node]
        else:
            self.new_body.append(updated_node)
        return updated_node

    def leave_If(self, original_node, node):
        parent_node = self.get_metadata(cst.metadata.ParentNodeProvider, original_node)
        if m.matches(parent_node, m.Module()):
            self.new_body.append(node)
        return node

    def leave_Expr(self, original_node: cst.Expr, node: cst.Expr) -> cst.Expr:
        parent_node = self.get_metadata(cst.metadata.ScopeProvider, original_node)
        if m.matches(parent_node, m.Module()):
            self.new_body.append(node)
        return node

    def leave_Module(self, original_node: cst.Assign, node):
        new_body = []
        for visiter in self.visited_module.values():
            new_body += list(visiter.imports.values())
            # TODO for the config we need to sort the dependencies using `class_finder.`
            self.config_body = list(visiter.imports.values()) + self.config_body
        return node.with_changes(body=[*new_body, *self.new_body])


def convert_file(diff_file, cst_transformers=None):
    # Parse the Python file
    with open(diff_file, "r") as file:
        code = file.read()
    module = cst.parse_module(code)
    wrapper = MetadataWrapper(module)
    if cst_transformers is None:
        cst_transformers = DiffConverterTransformer(module)
    new_mod = wrapper.visit(cst_transformers)
    ruffed_code = fix_ruff(new_mod.code)

    with open(diff_file.replace("diff_", "modeling_"), "w") as f:
        f.write(ruffed_code)

    if hasattr(cst_transformers, "config_body"):
        config_module = cst.Module(body=[*cst_transformers.config_body], header=new_mod.header)
        with open(diff_file.replace("diff_", "configuration_"), "w") as f:
            ruffed_code = fix_ruff(config_module.code)
            f.write(ruffed_code)
    
    # TODO optimize by re-using the class_finder
    return cst_transformers

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files_to_parse",
        default="all",
        help="A list of `diff_xxxx` files that should be converted to single model file",
    )
    args = parser.parse_args()
    if args.files_to_parse == "all":
        args.files_to_parse = glob.glob("src/transformers/models/**/diff_*.py", recursive=True)
    for file_name in args.files_to_parse:
        print(f"Converting {file_name} to a single model single file format")
        module_path = file_name.replace("/", ".").replace(".py", "").replace("src.", "")
        converter = convert_file(file_name)
