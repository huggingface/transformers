#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
import ast
import difflib
from collections.abc import Mapping
from typing import Any, Callable, Dict, Optional


class InterpretorError(ValueError):
    """
    An error raised when the interpretor cannot evaluate a Python expression, due to syntax error or unsupported
    operations.
    """

    pass


LIST_SAFE_MODULES = ["random", "math", "time", "queue", "itertools", "re", "stat", "statistics", "unicodedata"]


class BreakException(Exception):
    pass


class ContinueException(Exception):
    pass


def get_iterable(obj):
    if isinstance(obj, list):
        return obj
    elif hasattr(obj, "__iter__"):
        return list(obj)
    else:
        raise InterpretorError("Object is not iterable")


def evaluate_unaryop(expression, state, tools):
    operand = evaluate_ast(expression.operand, state, tools)
    if isinstance(expression.op, ast.USub):
        return -operand
    elif isinstance(expression.op, ast.UAdd):
        return operand
    elif isinstance(expression.op, ast.Not):
        return not operand
    elif isinstance(expression.op, ast.Invert):
        return ~operand
    else:
        raise InterpretorError(f"Unary operation {expression.op.__class__.__name__} is not supported.")


def evaluate_lambda(lambda_expression, state, tools):
    args = [arg.arg for arg in lambda_expression.args.args]

    def lambda_func(*values):
        new_state = state.copy()
        for arg, value in zip(args, values):
            new_state[arg] = value
        return evaluate_ast(lambda_expression.body, new_state, tools)

    return lambda_func


def evaluate_while(while_loop, state, tools):
    max_iterations = 1000
    iterations = 0
    while evaluate_ast(while_loop.test, state, tools):
        for node in while_loop.body:
            evaluate_ast(node, state, tools)
        iterations += 1
        if iterations > max_iterations:
            raise InterpretorError(f"Maximum number of {max_iterations} iterations in While loop exceeded")
    return None


def evaluate_function_def(function_def, state, tools):
    def create_function(func_def, state, tools):
        def new_func(*args):
            new_state = state.copy()
            for arg, val in zip(func_def.args.args, args):
                new_state[arg.arg] = val
            result = None
            for node in func_def.body:
                result = evaluate_ast(node, new_state, tools)
            return result

        return new_func

    tools[function_def.name] = create_function(function_def, state, tools)
    return None


def evaluate_augassign(expression: ast.AugAssign, state: Dict[str, Any], tools: Dict[str, Callable]):
    # Extract the target variable name and the operation
    if isinstance(expression.target, ast.Name):
        var_name = expression.target.id
        current_value = state.get(var_name, 0)  # Assuming default of 0 if not in state
        value_to_add = evaluate_ast(expression.value, state, tools)

        # Determine the operation and apply it
        if isinstance(expression.op, ast.Add):
            updated_value = current_value + value_to_add
        elif isinstance(expression.op, ast.Sub):
            updated_value = current_value - value_to_add
        elif isinstance(expression.op, ast.Mult):
            updated_value = current_value * value_to_add
        elif isinstance(expression.op, ast.Div):
            updated_value = current_value / value_to_add
        # Add other operations as needed

        # Update the state
        state[var_name] = updated_value
        return updated_value
    else:
        raise InterpretorError("AugAssign not supported for non-simple variable targets.")


def evaluate_boolop(boolop, state, tools):
    values = [evaluate_ast(val, state, tools) for val in boolop.values]
    op = boolop.op
    if isinstance(op, ast.And):
        return all(values)
    elif isinstance(op, ast.Or):
        return any(values)


def evaluate_binop(binop, state, tools):
    # Recursively evaluate the left and right operands
    left_val = evaluate_ast(binop.left, state, tools)
    right_val = evaluate_ast(binop.right, state, tools)

    # Determine the operation based on the type of the operator in the BinOp
    if isinstance(binop.op, ast.Add):
        return left_val + right_val
    elif isinstance(binop.op, ast.Sub):
        return left_val - right_val
    elif isinstance(binop.op, ast.Mult):
        return left_val * right_val
    elif isinstance(binop.op, ast.Div):
        return left_val / right_val
    elif isinstance(binop.op, ast.Mod):
        return left_val % right_val
    elif isinstance(binop.op, ast.Pow):
        return left_val**right_val
    elif isinstance(binop.op, ast.FloorDiv):
        return left_val // right_val
    elif isinstance(binop.op, ast.BitAnd):
        return left_val & right_val
    elif isinstance(binop.op, ast.BitOr):
        return left_val | right_val
    elif isinstance(binop.op, ast.BitXor):
        return left_val ^ right_val
    elif isinstance(binop.op, ast.LShift):
        return left_val << right_val
    elif isinstance(binop.op, ast.RShift):
        return left_val >> right_val
    else:
        raise NotImplementedError(f"Binary operation {type(binop.op).__name__} is not implemented.")


def evaluate_assign(assign, state, tools):
    var_names = assign.targets
    result = evaluate_ast(assign.value, state, tools)
    if len(var_names) == 1:
        if isinstance(var_names[0], ast.Tuple):
            for i, elem in enumerate(var_names[0].elts):
                state[elem.id] = result[i]
        else:
            state[var_names[0].id] = result
    else:
        if len(result) != len(var_names):
            raise InterpretorError(f"Expected {len(var_names)} values but got {len(result)}.")
        for var_name, r in zip(var_names, result):
            state[var_name.id] = r
    return result


def evaluate_call(call, state, tools):
    if isinstance(call.func, ast.Attribute):
        obj = evaluate_ast(call.func.value, state, tools)
        func_name = call.func.attr
        if not hasattr(obj, func_name):
            raise InterpretorError(f"Object {obj} has no attribute {func_name}")
        func = getattr(obj, func_name)
        args = [evaluate_ast(arg, state, tools) for arg in call.args]
        kwargs = {keyword.arg: evaluate_ast(keyword.value, state, tools) for keyword in call.keywords}
        return func(*args, **kwargs)

    elif isinstance(call.func, ast.Name):
        func_name = call.func.id

        if func_name in state:
            func = state[func_name]
        elif func_name in tools:
            func = tools[func_name]
        else:
            raise InterpretorError(
                f"It is not permitted to evaluate other functions than the provided tools or imported functions (tried to execute {call.func.id})."
            )
        # Todo deal with args
        args = [evaluate_ast(arg, state, tools) for arg in call.args]
        kwargs = {keyword.arg: evaluate_ast(keyword.value, state, tools) for keyword in call.keywords}
        output = func(*args, **kwargs)

        # store logs of print statements
        if func_name == "print":
            state["print_outputs"] += output + "\n"

        return output
    else:
        raise InterpretorError(
            f"It is not permitted to evaluate other functions than the provided tools (tried to execute {call.func})."
        )


def evaluate_subscript(subscript, state, tools):
    index = evaluate_ast(subscript.slice, state, tools)
    value = evaluate_ast(subscript.value, state, tools)
    if isinstance(index, slice):
        return value[index]
    elif isinstance(value, (list, tuple)):
        return value[int(index)]
    elif isinstance(value, str):
        return value[index]
    elif index in value:
        return value[index]
    elif isinstance(index, str) and isinstance(value, Mapping):
        close_matches = difflib.get_close_matches(index, list(value.keys()))
        if len(close_matches) > 0:
            return value[close_matches[0]]
    raise InterpretorError(f"Could not index {value} with '{index}'.")


def evaluate_name(name, state, tools):
    if name.id in state:
        return state[name.id]
    close_matches = difflib.get_close_matches(name.id, list(state.keys()))
    if len(close_matches) > 0:
        return state[close_matches[0]]
    raise InterpretorError(f"The variable `{name.id}` is not defined.")


def evaluate_condition(condition, state, tools):
    left = evaluate_ast(condition.left, state, tools)
    comparators = [evaluate_ast(c, state, tools) for c in condition.comparators]
    ops = [type(op) for op in condition.ops]

    result = left
    for op, comparator in zip(ops, comparators):
        if op == ast.Eq:
            result = result == comparator
        elif op == ast.NotEq:
            result = result != comparator
        elif op == ast.Lt:
            result = result < comparator
        elif op == ast.LtE:
            result = result <= comparator
        elif op == ast.Gt:
            result = result > comparator
        elif op == ast.GtE:
            result = result >= comparator
        elif op == ast.Is:
            result = result is comparator
        elif op == ast.IsNot:
            result = result is not comparator
        elif op == ast.In:
            result = result in comparator
        elif op == ast.NotIn:
            result = result not in comparator
        else:
            raise InterpretorError(f"Operator not supported: {op}")

    return result


def evaluate_if(if_statement, state, tools):
    result = None
    test_result = evaluate_ast(if_statement.test, state, tools)
    if test_result:
        for line in if_statement.body:
            line_result = evaluate_ast(line, state, tools)
            if line_result is not None:
                result = line_result
    else:
        for line in if_statement.orelse:
            line_result = evaluate_ast(line, state, tools)
            if line_result is not None:
                result = line_result
    return result


def evaluate_for(for_loop, state, tools):
    result = None
    iterator = evaluate_ast(for_loop.iter, state, tools)
    for counter in iterator:
        state[for_loop.target.id] = counter
        for node in for_loop.body:
            try:
                line_result = evaluate_ast(node, state, tools)
                if line_result is not None:
                    result = line_result
            except BreakException:
                break
            except ContinueException:
                continue
        else:
            continue
        break
    return result


def evaluate_listcomp(listcomp, state, tools):
    result = []
    vars = {}
    for generator in listcomp.generators:
        var_name = generator.target.id
        iter_value = evaluate_ast(generator.iter, state, tools)
        for value in iter_value:
            vars[var_name] = value
            if all(evaluate_ast(if_clause, {**state, **vars}, tools) for if_clause in generator.ifs):
                elem = evaluate_ast(listcomp.elt, {**state, **vars}, tools)
                result.append(elem)
    return result


def evaluate_ast(expression: ast.AST, state: Dict[str, Any], tools: Dict[str, Callable]):
    """
    Evaluate an abstract syntax tree using the content of the variables stored in a state and only evaluating a given
    set of functions.

    This function will recurse trough the nodes of the tree provided.

    Args:
        expression (`ast.AST`):
            The code to evaluate, as an abastract syntax tree.
        state (`Dict[str, Any]`):
            A dictionary mapping variable names to values. The `state` is updated if need be when the evaluation
            encounters assignements.
        tools (`Dict[str, Callable]`):
            The functions that may be called during the evaluation. Any call to another function will fail with an
            `InterpretorError`.
    """
    if isinstance(expression, ast.Assign):
        # Assignement -> we evaluate the assignement which should update the state
        # We return the variable assigned as it may be used to determine the final result.
        return evaluate_assign(expression, state, tools)
    elif isinstance(expression, ast.AugAssign):
        return evaluate_augassign(expression, state, tools)
    elif isinstance(expression, ast.Call):
        # Function call -> we return the value of the function call
        return evaluate_call(expression, state, tools)
    elif isinstance(expression, ast.Constant):
        # Constant -> just return the value
        return expression.value
    elif isinstance(expression, ast.Tuple):
        return tuple(evaluate_ast(elt, state, tools) for elt in expression.elts)
    elif isinstance(expression, ast.ListComp):
        return evaluate_listcomp(expression, state, tools)
    elif isinstance(expression, ast.UnaryOp):
        return evaluate_unaryop(expression, state, tools)
    elif isinstance(expression, ast.BoolOp):
        # Boolean operation -> evaluate the operation
        return evaluate_boolop(expression, state, tools)
    elif isinstance(expression, ast.Break):
        raise BreakException()
    elif isinstance(expression, ast.Continue):
        raise ContinueException()
    elif isinstance(expression, ast.BinOp):
        # Binary operation -> execute operation
        return evaluate_binop(expression, state, tools)
    elif isinstance(expression, ast.Compare):
        # Comparison -> evaluate the comparison
        return evaluate_condition(expression, state, tools)
    elif isinstance(expression, ast.Return):
        return evaluate_ast(expression.value, state, tools)
    elif isinstance(expression, ast.Lambda):
        return evaluate_lambda(expression, state, tools)
    elif isinstance(expression, ast.FunctionDef):
        return evaluate_function_def(expression, state, tools)
    elif isinstance(expression, ast.Dict):
        # Dict -> evaluate all keys and values
        keys = [evaluate_ast(k, state, tools) for k in expression.keys]
        values = [evaluate_ast(v, state, tools) for v in expression.values]
        return dict(zip(keys, values))
    elif isinstance(expression, ast.Expr):
        # Expression -> evaluate the content
        return evaluate_ast(expression.value, state, tools)
    elif isinstance(expression, ast.For):
        # For loop -> execute the loop
        return evaluate_for(expression, state, tools)
    elif isinstance(expression, ast.FormattedValue):
        # Formatted value (part of f-string) -> evaluate the content and return
        return evaluate_ast(expression.value, state, tools)
    elif isinstance(expression, ast.If):
        # If -> execute the right branch
        return evaluate_if(expression, state, tools)
    elif hasattr(ast, "Index") and isinstance(expression, ast.Index):
        return evaluate_ast(expression.value, state, tools)
    elif isinstance(expression, ast.JoinedStr):
        return "".join([str(evaluate_ast(v, state, tools)) for v in expression.values])
    elif isinstance(expression, ast.List):
        # List -> evaluate all elements
        return [evaluate_ast(elt, state, tools) for elt in expression.elts]
    elif isinstance(expression, ast.Name):
        # Name -> pick up the value in the state
        return evaluate_name(expression, state, tools)
    elif isinstance(expression, ast.Subscript):
        # Subscript -> return the value of the indexing
        return evaluate_subscript(expression, state, tools)
    elif isinstance(expression, ast.IfExp):
        test_val = evaluate_ast(expression.test, state, tools)
        if test_val:
            return evaluate_ast(expression.body, state, tools)
        else:
            return evaluate_ast(expression.orelse, state, tools)
    elif isinstance(expression, ast.Attribute):
        obj = evaluate_ast(expression.value, state, tools)
        return getattr(obj, expression.attr)
    elif isinstance(expression, ast.Slice):
        return slice(
            evaluate_ast(expression.lower, state, tools) if expression.lower is not None else None,
            evaluate_ast(expression.upper, state, tools) if expression.upper is not None else None,
            evaluate_ast(expression.step, state, tools) if expression.step is not None else None,
        )
    elif isinstance(expression, ast.ListComp) or isinstance(expression, ast.GeneratorExp):
        result = []
        vars = {}
        for generator in expression.generators:
            var_name = generator.target.id
            iter_value = evaluate_ast(generator.iter, state, tools)
            for value in iter_value:
                vars[var_name] = value
                if all(evaluate_ast(if_clause, {**state, **vars}, tools) for if_clause in generator.ifs):
                    elem = evaluate_ast(expression.elt, {**state, **vars}, tools)
                    result.append(elem)
        return result
    elif isinstance(expression, ast.DictComp):
        result = {}
        for gen in expression.generators:
            for container in get_iterable(evaluate_ast(gen.iter, state, tools)):
                state[gen.target.id] = container
                key = evaluate_ast(expression.key, state, tools)
                value = evaluate_ast(expression.value, state, tools)
                result[key] = value
        return result
    elif isinstance(expression, ast.Import):
        for alias in expression.names:
            if alias.name in LIST_SAFE_MODULES:
                module = __import__(alias.name)
                state[alias.asname or alias.name] = module
            else:
                raise InterpretorError(f"Import of {alias.name} is not allowed.")
        return None
    elif isinstance(expression, ast.While):
        return evaluate_while(expression, state, tools)
    elif isinstance(expression, ast.ImportFrom):
        if expression.module in LIST_SAFE_MODULES:
            module = __import__(expression.module)
            for alias in expression.names:
                state[alias.asname or alias.name] = getattr(module, alias.name)
        else:
            raise InterpretorError(f"Import from {expression.module} is not allowed.")
        return None
    else:
        # For now we refuse anything else. Let's add things as we need them.
        raise InterpretorError(f"{expression.__class__.__name__} is not supported.")


def evaluate_python_code(code: str, tools: Optional[Dict[str, Callable]] = {}, state=None):
    """
    Evaluate a python expression using the content of the variables stored in a state and only evaluating a given set
    of functions.

    This function will recurse through the nodes of the tree provided.

    Args:
        code (`str`):
            The code to evaluate.
        tools (`Dict[str, Callable]`):
            The functions that may be called during the evaluation. Any call to another function will fail with an
            `InterpretorError`.
        state (`Dict[str, Any]`):
            A dictionary mapping variable names to values. The `state` should contain the initial inputs but will be
            updated by this function to contain all variables as they are evaluated.
            The print outputs will be stored in the state under the key 'print_outputs'.
    """
    try:
        expression = ast.parse(code)
    except SyntaxError as e:
        raise SyntaxError(f"The code generated by the agent is not valid.\n{e}")
    if state is None:
        state = {}
    result = None
    state["print_outputs"] = ""
    for idx, node in enumerate(expression.body):
        try:
            line_result = evaluate_ast(node, state, tools)
        except InterpretorError as e:
            msg = f"You tried to execute the following code:\n{code}\n"
            msg += f"You got these outputs:\n{state['print_outputs']}\n"
            msg += f"Evaluation stopped at line '{node}' because of the following error:\n{e}"
            raise InterpretorError(msg)
        if line_result is not None:
            result = line_result

    return result
