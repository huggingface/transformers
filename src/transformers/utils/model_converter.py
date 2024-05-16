import libcst as cst
from libcst import matchers as m



module = cst.parse_module("import math")
module.code

# Should we use the scope to figure out if classes are imported and inherited from 
# then go from here, instead of visiting the classes?

# Define a visitor to traverse the tree and find import statements
class ImportVisitor(cst.CSTVisitor):
    def __init__(self, python_module):
        self.transformers_imports = []
        self.python_module = python_module

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        if m.matches(node.module, m.Attribute()):
            print("full import statement:", self.python_module.code_for_node(node))
            for imported_ in node.names:
                print("Modules", self.python_module.code_for_node(imported_.name))
                self.transformers_imports.append((node.module.value, imported_))
                
    def visit_Assign(self,  node: cst.Assign) -> None:
        if m.matches(node.value, m.Name()):
            name =node.value.value 
            if "Llama" in name:
                print(f"Found assignment that should be replaced:{self.python_module.code_for_node(node)}")
            # TODO This is what has to be replaced if it's in 
            # transformers_modeling_imports! Replaced by original source
    
    def visit_ClassDef(self,  node: cst.ClassDef) -> None:
        print(node.name)

    def visit_Import(self, node: cst.Import) -> None:
        for alias in node.names:
            if isinstance(alias.name, cst.Attribute) and alias.name.value == "transformers":
                self.transformers_imports.append((alias.name.attr.value, node))
            elif isinstance(alias.name.value, str) and alias.name.value.startswith("transformers."):
                self.transformers_imports.append((alias.name.value, node))

# Function to get the source code of the imported entity
def get_imported_entity_source(module, entity_name):
    source_code = []
    for stmt in module.body:
        if isinstance(stmt, cst.SimpleStatementLine) and isinstance(stmt.body, cst.FunctionDef):
            if stmt.body.name.value == entity_name:
                source_code.append(cst.MetadataWrapper(module.get_metadata(cst.metadata.PositionProvider)).code_for_node(stmt))
        elif isinstance(stmt, cst.SimpleStatementLine) and isinstance(stmt.body, cst.ClassDef):
            if stmt.body.name.value == entity_name:
                source_code.append(cst.MetadataWrapper(module.get_metadata(cst.metadata.PositionProvider)).code_for_node(stmt))
    return '\n'.join(source_code)

# Parse the Python file
with open("/Users/arthurzucker/Work/transformers/src/transformers/models/gemma/diff_gemma.py", "r") as file:
    code = file.read()
module = cst.parse_module(code)

# Use the visitor to find imports
visitor = ImportVisitor(module)
module.visit(visitor)
exit(0)
# Process the found imports
for imported_name, import_node in visitor.transformers_imports:
    print(f"Imported from transformers: {imported_name}")
    # Extract the source code of the imported entity
    entity_source_code = get_imported_entity_source(module, imported_name)
    print(f"Source code of {imported_name}:")
    print(entity_source_code)