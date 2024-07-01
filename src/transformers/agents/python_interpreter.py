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
import builtins
import difflib
from collections.abc import Mapping
from typing import Any, Callable, Dict, List, Optional


class InterpreterError(ValueError):
    """
    An error raised when the interpretor cannot evaluate a Python expression, due to syntax error or unsupported
    operations.
    """

    pass


ERRORS = {
    name: getattr(builtins, name)
    for name in dir(builtins)
    if isinstance(getattr(builtins, name), type) and issubclass(getattr(builtins, name), BaseException)
}


LIST_SAFE_MODULES = [
    "random",
    "collections",
    "math",
    "time",
    "queue",
    "itertools",
    "re",
    "stat",
    "statistics",
    "unicodedata",
]

PRINT_OUTPUTS = ""


class BreakException(Exception):
    pass


class ContinueException(Exception):
    pass


class ReturnException(Exception):
    def __init__(self, value):
        self.value = value


def get_iterable(obj):
    if isinstance(obj, list):
        return obj
    elif hasattr(obj, "__iter__"):
        return list(obj)
    else:
        raise InterpreterError("Object is not iterable")


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
        raise InterpreterError(f"Unary operation {expression.op.__class__.__name__} is not supported.")


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
            try:
                evaluate_ast(node, state, tools)
            except BreakException:
                return None
            except ContinueException:
                break
        iterations += 1
        if iterations > max_iterations:
            raise InterpreterError(f"Maximum number of {max_iterations} iterations in While loop exceeded")
    return None


def create_function(func_def, state, tools):
    def new_func(*args, **kwargs):
        func_state = state.copy()
        arg_names = [arg.arg for arg in func_def.args.args]
        default_values = [evaluate_ast(d, state, tools) for d in func_def.args.defaults]

        # Apply default values
        defaults = dict(zip(arg_names[-len(default_values) :], default_values))

        # Set positional arguments
        for name, value in zip(arg_names, args):
            func_state[name] = value

        # # Set keyword arguments
        for name, value in kwargs.items():
            func_state[name] = value

        # Handle variable arguments
        if func_def.args.vararg:
            vararg_name = func_def.args.vararg.arg
            func_state[vararg_name] = args

        if func_def.args.kwarg:
            kwarg_name = func_def.args.kwarg.arg
            func_state[kwarg_name] = kwargs

        # Set default values for arguments that were not provided
        for name, value in defaults.items():
            if name not in func_state:
                func_state[name] = value

        # Update function state with self and __class__
        if func_def.args.args and func_def.args.args[0].arg == "self":
            if args:
                func_state["self"] = args[0]
                func_state["__class__"] = args[0].__class__

        result = None
        try:
            for stmt in func_def.body:
                result = evaluate_ast(stmt, func_state, tools)
        except ReturnException as e:
            result = e.value
        return result

    return new_func


def create_class(class_name, class_bases, class_body):
    class_dict = {}
    for key, value in class_body.items():
        class_dict[key] = value
    return type(class_name, tuple(class_bases), class_dict)


def evaluate_function_def(func_def, state, tools):
    tools[func_def.name] = create_function(func_def, state, tools)
    return tools[func_def.name]


def evaluate_class_def(class_def, state, tools):
    class_name = class_def.name
    bases = [evaluate_ast(base, state, tools) for base in class_def.bases]
    class_dict = {}

    for stmt in class_def.body:
        if isinstance(stmt, ast.FunctionDef):
            class_dict[stmt.name] = evaluate_function_def(stmt, state, tools)
        elif isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    class_dict[target.id] = evaluate_ast(stmt.value, state, tools)
                elif isinstance(target, ast.Attribute):
                    class_dict[target.attr] = evaluate_ast(stmt.value, state, tools)
        else:
            raise InterpreterError(f"Unsupported statement in class body: {stmt.__class__.__name__}")

    new_class = type(class_name, tuple(bases), class_dict)
    state[class_name] = new_class
    return new_class


def evaluate_augassign(expression: ast.AugAssign, state: Dict[str, Any], tools: Dict[str, Callable]):
    # Helper function to get current value and set new value based on the target type
    def get_current_value(target):
        if isinstance(target, ast.Name):
            return state.get(target.id, 0)
        elif isinstance(target, ast.Subscript):
            obj = evaluate_ast(target.value, state, tools)
            key = evaluate_ast(target.slice, state, tools)
            return obj[key]
        elif isinstance(target, ast.Attribute):
            obj = evaluate_ast(target.value, state, tools)
            return getattr(obj, target.attr)
        elif isinstance(target, ast.Tuple):
            return tuple(get_current_value(elt) for elt in target.elts)
        elif isinstance(target, ast.List):
            return [get_current_value(elt) for elt in target.elts]
        else:
            raise InterpreterError("AugAssign not supported for {type(target)} targets.")

    current_value = get_current_value(expression.target)
    value_to_add = evaluate_ast(expression.value, state, tools)

    # Determine the operation and apply it
    if isinstance(expression.op, ast.Add):
        if isinstance(current_value, list):
            if not isinstance(value_to_add, list):
                raise InterpreterError(f"Cannot add non-list value {value_to_add} to a list.")
            updated_value = current_value + value_to_add
        else:
            updated_value = current_value + value_to_add
    elif isinstance(expression.op, ast.Sub):
        updated_value = current_value - value_to_add
    elif isinstance(expression.op, ast.Mult):
        updated_value = current_value * value_to_add
    elif isinstance(expression.op, ast.Div):
        updated_value = current_value / value_to_add
    elif isinstance(expression.op, ast.Mod):
        updated_value = current_value % value_to_add
    elif isinstance(expression.op, ast.Pow):
        updated_value = current_value**value_to_add
    elif isinstance(expression.op, ast.FloorDiv):
        updated_value = current_value // value_to_add
    elif isinstance(expression.op, ast.BitAnd):
        updated_value = current_value & value_to_add
    elif isinstance(expression.op, ast.BitOr):
        updated_value = current_value | value_to_add
    elif isinstance(expression.op, ast.BitXor):
        updated_value = current_value ^ value_to_add
    elif isinstance(expression.op, ast.LShift):
        updated_value = current_value << value_to_add
    elif isinstance(expression.op, ast.RShift):
        updated_value = current_value >> value_to_add
    else:
        raise InterpreterError(f"Operation {type(expression.op).__name__} is not supported.")

    # Update the state
    set_value(expression.target, updated_value, state, tools)

    return updated_value


def evaluate_boolop(node, state, tools):
    if isinstance(node.op, ast.And):
        for value in node.values:
            if not evaluate_ast(value, state, tools):
                return False
        return True
    elif isinstance(node.op, ast.Or):
        for value in node.values:
            if evaluate_ast(value, state, tools):
                return True
        return False


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
    result = evaluate_ast(assign.value, state, tools)
    if len(assign.targets) == 1:
        target = assign.targets[0]
        set_value(target, result, state, tools)
    else:
        if len(assign.targets) != len(result):
            raise InterpreterError(f"Assign failed: expected {len(result)} values but got {len(assign.targets)}.")
        for tgt, val in zip(assign.targets, result):
            set_value(tgt, val, state, tools)
    return result


def set_value(target, value, state, tools):
    if isinstance(target, ast.Name):
        if target.id in tools:
            raise InterpreterError(f"Cannot assign to name '{target.id}': doing this would erase the existing tool!")
        state[target.id] = value
    elif isinstance(target, ast.Tuple):
        if not isinstance(value, tuple):
            raise InterpreterError("Cannot unpack non-tuple value")
        if len(target.elts) != len(value):
            raise InterpreterError("Cannot unpack tuple of wrong size")
        for i, elem in enumerate(target.elts):
            set_value(elem, value[i], state, tools)
    elif isinstance(target, ast.Subscript):
        obj = evaluate_ast(target.value, state, tools)
        key = evaluate_ast(target.slice, state, tools)
        obj[key] = value
    elif isinstance(target, ast.Attribute):
        obj = evaluate_ast(target.value, state, tools)
        setattr(obj, target.attr, value)


def evaluate_call(call, state, tools):
    if not (isinstance(call.func, ast.Attribute) or isinstance(call.func, ast.Name)):
        raise InterpreterError(
            f"It is not permitted to evaluate other functions than the provided tools (tried to execute {call.func})."
        )
    if isinstance(call.func, ast.Attribute):
        obj = evaluate_ast(call.func.value, state, tools)
        func_name = call.func.attr
        if not hasattr(obj, func_name):
            raise InterpreterError(f"Object {obj} has no attribute {func_name}")
        func = getattr(obj, func_name)
    elif isinstance(call.func, ast.Name):
        func_name = call.func.id
        if func_name in state:
            func = state[func_name]
        elif func_name in tools:
            func = tools[func_name]
        elif func_name in ERRORS:
            func = ERRORS[func_name]
        else:
            raise InterpreterError(
                f"It is not permitted to evaluate other functions than the provided tools or imported functions (tried to execute {call.func.id})."
            )

    args = [evaluate_ast(arg, state, tools) for arg in call.args]
    kwargs = {keyword.arg: evaluate_ast(keyword.value, state, tools) for keyword in call.keywords}

    if isinstance(func, type) and len(func.__module__.split(".")) > 1:  # Check for user-defined classes
        # Instantiate the class using its constructor
        obj = func.__new__(func)  # Create a new instance of the class
        if hasattr(obj, "__init__"):  # Check if the class has an __init__ method
            obj.__init__(*args, **kwargs)  # Call the __init__ method correctly
        return obj
    else:
        if func_name == "super":
            if not args:
                if "__class__" in state and "self" in state:
                    return super(state["__class__"], state["self"])
                else:
                    raise InterpreterError("super() needs at least one argument")
            cls = args[0]
            if not isinstance(cls, type):
                raise InterpreterError("super() argument 1 must be type")
            if len(args) == 1:
                return super(cls)
            elif len(args) == 2:
                instance = args[1]
                return super(cls, instance)
            else:
                raise InterpreterError("super() takes at most 2 arguments")
        else:
            if func_name == "print":
                output = " ".join(map(str, args))
                global PRINT_OUTPUTS
                PRINT_OUTPUTS += output + "\n"
                return output
            else:  # Assume it's a callable object
                output = func(*args, **kwargs)
                return output


def evaluate_subscript(subscript, state, tools):
    index = evaluate_ast(subscript.slice, state, tools)
    value = evaluate_ast(subscript.value, state, tools)
    if isinstance(index, slice):
        return value[index]
    elif isinstance(value, (list, tuple)):
        # Ensure the index is within bounds
        if not (-len(value) <= index < len(value)):
            raise InterpreterError(f"Index {index} out of bounds for list of length {len(value)}")
        return value[int(index)]
    elif isinstance(value, str):
        # Ensure the index is within bounds
        if not (-len(value) <= index < len(value)):
            raise InterpreterError(f"Index {index} out of bounds for string of length {len(value)}")
        return value[index]
    elif index in value:
        return value[index]
    elif isinstance(index, str) and isinstance(value, Mapping):
        close_matches = difflib.get_close_matches(index, list(value.keys()))
        if len(close_matches) > 0:
            return value[close_matches[0]]
    raise InterpreterError(f"Could not index {value} with '{index}'.")


def evaluate_name(name, state, tools):
    if name.id in state:
        return state[name.id]
    elif name.id in tools:
        return tools[name.id]
    elif name.id in ERRORS:
        return ERRORS[name.id]
    close_matches = difflib.get_close_matches(name.id, list(state.keys()))
    if len(close_matches) > 0:
        return state[close_matches[0]]
    raise InterpreterError(f"The variable `{name.id}` is not defined.")


def evaluate_condition(condition, state, tools):
    left = evaluate_ast(condition.left, state, tools)
    comparators = [evaluate_ast(c, state, tools) for c in condition.comparators]
    ops = [type(op) for op in condition.ops]

    result = True
    current_left = left

    for op, comparator in zip(ops, comparators):
        if op == ast.Eq:
            result = result and (current_left == comparator)
        elif op == ast.NotEq:
            result = result and (current_left != comparator)
        elif op == ast.Lt:
            result = result and (current_left < comparator)
        elif op == ast.LtE:
            result = result and (current_left <= comparator)
        elif op == ast.Gt:
            result = result and (current_left > comparator)
        elif op == ast.GtE:
            result = result and (current_left >= comparator)
        elif op == ast.Is:
            result = result and (current_left is comparator)
        elif op == ast.IsNot:
            result = result and (current_left is not comparator)
        elif op == ast.In:
            result = result and (current_left in comparator)
        elif op == ast.NotIn:
            result = result and (current_left not in comparator)
        else:
            raise InterpreterError(f"Operator not supported: {op}")

        current_left = comparator
        if not result:
            break

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
        if isinstance(for_loop.target, ast.Tuple):
            for i, elem in enumerate(for_loop.target.elts):
                state[elem.id] = counter[i]
        else:
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
    for generator in listcomp.generators:
        iter_value = evaluate_ast(generator.iter, state, tools)
        for value in iter_value:
            new_state = state.copy()
            if isinstance(generator.target, ast.Tuple):
                for idx, elem in enumerate(generator.target.elts):
                    new_state[elem.id] = value[idx]
            else:
                new_state[generator.target.id] = value
            if all(evaluate_ast(if_clause, new_state, tools) for if_clause in generator.ifs):
                result.append(evaluate_ast(listcomp.elt, new_state, tools))
    return result


def evaluate_try(try_node, state, tools):
    try:
        for stmt in try_node.body:
            evaluate_ast(stmt, state, tools)
    except Exception as e:
        matched = False
        for handler in try_node.handlers:
            if handler.type is None or isinstance(e, evaluate_ast(handler.type, state, tools)):
                matched = True
                if handler.name:
                    state[handler.name] = e
                for stmt in handler.body:
                    evaluate_ast(stmt, state, tools)
                break
        if not matched:
            raise e
    else:
        if try_node.orelse:
            for stmt in try_node.orelse:
                evaluate_ast(stmt, state, tools)
    finally:
        if try_node.finalbody:
            for stmt in try_node.finalbody:
                evaluate_ast(stmt, state, tools)


def evaluate_raise(raise_node, state, tools):
    if raise_node.exc is not None:
        exc = evaluate_ast(raise_node.exc, state, tools)
    else:
        exc = None
    if raise_node.cause is not None:
        cause = evaluate_ast(raise_node.cause, state, tools)
    else:
        cause = None
    if exc is not None:
        if cause is not None:
            raise exc from cause
        else:
            raise exc
    else:
        raise InterpreterError("Re-raise is not supported without an active exception")


def evaluate_assert(assert_node, state, tools):
    test_result = evaluate_ast(assert_node.test, state, tools)
    if not test_result:
        if assert_node.msg:
            msg = evaluate_ast(assert_node.msg, state, tools)
            raise AssertionError(msg)
        else:
            # Include the failing condition in the assertion message
            test_code = ast.unparse(assert_node.test)
            raise AssertionError(f"Assertion failed: {test_code}")


def evaluate_with(with_node, state, tools):
    contexts = []
    for item in with_node.items:
        context_expr = evaluate_ast(item.context_expr, state, tools)
        if item.optional_vars:
            state[item.optional_vars.id] = context_expr.__enter__()
            contexts.append(state[item.optional_vars.id])
        else:
            context_var = context_expr.__enter__()
            contexts.append(context_var)

    try:
        for stmt in with_node.body:
            evaluate_ast(stmt, state, tools)
    except Exception as e:
        for context in reversed(contexts):
            context.__exit__(type(e), e, e.__traceback__)
        raise
    else:
        for context in reversed(contexts):
            context.__exit__(None, None, None)


def evaluate_ast(
    expression: ast.AST,
    state: Dict[str, Any],
    tools: Dict[str, Callable],
    authorized_imports: List[str] = LIST_SAFE_MODULES,
):
    """
    Evaluate an abstract syntax tree using the content of the variables stored in a state and only evaluating a given
    set of functions.

    This function will recurse trough the nodes of the tree provided.

    Args:
        expression (`ast.AST`):
            The code to evaluate, as an abstract syntax tree.
        state (`Dict[str, Any]`):
            A dictionary mapping variable names to values. The `state` is updated if need be when the evaluation
            encounters assignements.
        tools (`Dict[str, Callable]`):
            The functions that may be called during the evaluation. Any call to another function will fail with an
            `InterpreterError`.
        authorized_imports (`List[str]`):
            The list of modules that can be imported by the code. By default, only a few safe modules are allowed.
            Add more at your own risk!
    """
    if isinstance(expression, ast.Assign):
        # Assignement -> we evaluate the assignment which should update the state
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
            if alias.name in authorized_imports:
                module = __import__(alias.name)
                state[alias.asname or alias.name] = module
            else:
                raise InterpreterError(f"Import of {alias.name} is not allowed.")
        return None
    elif isinstance(expression, ast.While):
        return evaluate_while(expression, state, tools)
    elif isinstance(expression, ast.ImportFrom):
        if expression.module in authorized_imports:
            module = __import__(expression.module)
            for alias in expression.names:
                state[alias.asname or alias.name] = getattr(module, alias.name)
        else:
            raise InterpreterError(f"Import from {expression.module} is not allowed.")
        return None
    elif isinstance(expression, ast.ClassDef):
        return evaluate_class_def(expression, state, tools)
    elif isinstance(expression, ast.Try):
        return evaluate_try(expression, state, tools)
    elif isinstance(expression, ast.Raise):
        return evaluate_raise(expression, state, tools)
    elif isinstance(expression, ast.Assert):
        return evaluate_assert(expression, state, tools)
    elif isinstance(expression, ast.With):
        return evaluate_with(expression, state, tools)
    elif isinstance(expression, ast.Set):
        return {evaluate_ast(elt, state, tools) for elt in expression.elts}
    elif isinstance(expression, ast.Return):
        raise ReturnException(evaluate_ast(expression.value, state, tools) if expression.value else None)
    else:
        # For now we refuse anything else. Let's add things as we need them.
        raise InterpreterError(f"{expression.__class__.__name__} is not supported.")


def evaluate_python_code(
    code: str, tools: Optional[Dict[str, Callable]] = {}, state=None, authorized_imports: List[str] = LIST_SAFE_MODULES
):
    """
    Evaluate a python expression using the content of the variables stored in a state and only evaluating a given set
    of functions.

    This function will recurse through the nodes of the tree provided.

    Args:
        code (`str`):
            The code to evaluate.
        tools (`Dict[str, Callable]`):
            The functions that may be called during the evaluation. Any call to another function will fail with an
            `InterpreterError`.
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
    global PRINT_OUTPUTS
    PRINT_OUTPUTS = ""
    for node in expression.body:
        try:
            result = evaluate_ast(node, state, tools, authorized_imports)
        except InterpreterError as e:
            msg = f"Evaluation stopped at line '{ast.get_source_segment(code, node)}' because of the following error:\n{e}"
            if len(PRINT_OUTPUTS) > 0:
                msg += f"Executing code yielded these outputs:\n{PRINT_OUTPUTS}\n====\n"
            raise InterpreterError(msg)
        finally:
            state["print_outputs"] = PRINT_OUTPUTS

    return result
