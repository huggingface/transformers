# NOTE issue: https://github.com/huggingface/transformers/issues/36057

import os
import re
import pdb
import ast
import json
from tqdm import tqdm
from collections import defaultdict


class MyVisitor(ast.NodeVisitor):
    def __init__(self):
        self.class_function_types = dict()
        self.types_found = set()
        self.current_class = None
    
    def visit_ClassDef(self, node):
        # print(f"Class name: {node.name}")
        self.current_class = node.name
        self.generic_visit(node)
        
    def visit_FunctionDef(self, node):
        # print(f"    Function name: {node.name}")
        if node.name == "forward":
            for arg in node.args.args:
                if arg.arg == "past_key_values" and arg.annotation:
                    type_annotation = ast.unparse(arg.annotation).strip()
                    self.class_function_types[self.current_class] = type_annotation
                    self.types_found.add(type_annotation)
        self.generic_visit(node)
        
class TypeAnnotationUnifier(ast.NodeTransformer):
    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node.name == 'forward':
            for arg in node.args.args:
                if arg.arg == 'past_key_values':
                    arg.annotation = ast.parse(self.tgt_type).body[0].value
        self.generic_visit(node)
        return node
        
def parse_class_types(fpath):
    with open(fpath, 'r') as f:
        content = f.read()
    class_function_types = {}
    types_found = set()
    class_stack = []
    tree = ast.parse(content)
    current_class = None
        
    visitor = MyVisitor()
    visitor.visit(tree)
    
    if len(visitor.types_found) > 1:
        print(json.dumps(visitor.class_function_types, indent=2))
        pdb.set_trace()
        
def unify_type_annotations(fpath, tgt_type):
    with open(fpath, 'r', encoding='utf-8') as f:
        src_code = f.read()
        
    tree = ast.parse(src_code)
    transformer = TypeAnnotationUnifier()
    modified_tree = transformer.visit(tree)
    
    modified_code = astunparse.unparse(modified_tree)
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(modified_code)

def main():
    base_dir = '/vast/yx3038/repos/transformers/src/transformers/models'
    for model in tqdm(sorted(os.listdir(base_dir))):
        sub_dir = os.path.join(base_dir, model)
        if not os.path.isdir(sub_dir):
            continue
        if f'modeling_{model}.py' not in os.listdir(sub_dir):
            continue
        fpath = os.path.join(sub_dir, f'modeling_{model}.py')
        print(f'parsing {fpath}')
        parse_class_types(fpath)
        
            
if __name__ == '__main__':
    main()