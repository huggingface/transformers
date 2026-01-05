#!/usr/bin/env python3
"""
CST-based Test Expectations Updater - Version 2.0 (Complete Redesign)

================================================================================
DESIGN PRINCIPLES
================================================================================

1. CST FOR ALL POSITION DETECTION
   - Parse the test file ONCE with CST
   - Use CST to find exact character positions for all replacements
   - No string searching, no regex

2. SIMPLE STRING OPERATIONS FOR REPLACEMENT
   - All replacements are pure string operations
   - Work on in-memory content
   - No AST, no regex, no CST during replacement

3. SINGLE FILE WRITE
   - Collect all replacement positions and values
   - Apply all replacements in reverse order (high to low position)
   - Write to file once at the end

================================================================================
WORKFLOW
================================================================================

Phase 1: Parse captured_info.txt
  - Extract all assertion failures
  - Identify which argument to update (EXPECT* prefix priority)
  - Get new value from the OTHER argument

Phase 2: CST Analysis (ONE PARSE)
  - Parse the test file with CST once
  - For each assertion line, trace back to variable definition
  - Extract EXACT positions (start_line, start_col, end_line, end_col)
  - Handle all patterns: torch.tensor, Expectations, lists, dicts, strings

Phase 3: Compute Replacements
  - For each position, compute the replacement text
  - Handle indentation, multi-line formatting
  - Use ONLY string operations (no CST, no AST, no regex)

Phase 4: Apply All Replacements
  - Sort by position (highest first to avoid offset issues)
  - Apply each replacement as pure string operation
  - Write file once

================================================================================
"""

import argparse
import libcst as cst
from libcst.metadata import PositionProvider
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass


@dataclass
class CapturedBlock:
    """A single assertion failure block from captured_info.txt"""
    test_name: str
    test_file: str
    test_context: str
    assertion_line: int
    assertion_code: str
    arguments: List[Tuple[str, str, str]]  # (name, expression, value)


@dataclass
class UpdateTask:
    """Information about what to update and where"""
    variable_name: str  # e.g., "EXPECTED_OUTPUT" or "torch.tensor([...])"
    new_value: str  # The new value from captured_info
    assertion_line: int  # Where the assertion is
    is_inline: bool  # True if it's an inline expression


@dataclass
class ReplacementPosition:
    """Exact position to replace in the file"""
    start_line: int  # 1-indexed
    start_col: int  # 0-indexed
    end_line: int  # 1-indexed
    end_col: int  # 0-indexed
    pattern_type: str  # "torch.tensor", "Expectations", "list", "string", etc.
    indent_level: int  # Base indentation


@dataclass
class Replacement:
    """A computed replacement ready to apply"""
    position: ReplacementPosition
    new_text: str  # The exact text to insert


# ============================================================================
# PHASE 1: PARSE CAPTURED_INFO.TXT
# ============================================================================

def parse_captured_info(filepath: str) -> List[CapturedBlock]:
    """
    Parse captured_info.txt and extract all assertion failure blocks.
    
    Returns a list of CapturedBlock objects, each containing:
    - test name and file
    - assertion line and code
    - all arguments with their names, expressions, and values
    """
    with open(filepath) as f:
        lines = f.readlines()
    
    blocks = []
    current_block = None
    i = 0
    current_arg_name = None
    collecting_value = False
    value_lines = []
    
    while i < len(lines):
        line = lines[i].rstrip()
        
        if line.startswith('test:'):
            # Start new block - next non-empty line has the test name
            if current_block and current_block.test_name:
                blocks.append(current_block)
            
            # Find test name (skip blank lines)
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            
            if i < len(lines):
                test_name = lines[i].strip()
                # Format: tests/models/MODEL/test_file.py::Class::test_method
                test_file = test_name.split('::')[0] if '::' in test_name else ''
                
                current_block = CapturedBlock(
                    test_name=test_name,
                    test_file=test_file,
                    test_context='',
                    assertion_line=0,
                    assertion_code='',
                    arguments=[]
                )
                current_arg_name = None
                collecting_value = False
                value_lines = []
            i += 1
            continue
        
        elif current_block:
            if line.startswith('test context:'):
                # Format: test context: /path/file.py:LINE
                # Next lines have the code, extract line number from first line
                context_line = line[13:].strip()
                current_block.test_context = context_line
                
                # Extract line number for assertion
                if ':' in context_line:
                    parts = context_line.split(':')
                    try:
                        current_block.assertion_line = int(parts[-1])
                    except:
                        pass
                
                i += 1
                continue
            
            elif line.startswith('argument name:'):
                # Save previous argument if collecting value
                if collecting_value and current_arg_name:
                    value_str = '\n'.join(value_lines)
                    # Find and update the argument
                    for idx, (name, expr, _) in enumerate(current_block.arguments):
                        if name == current_arg_name:
                            current_block.arguments[idx] = (name, expr, value_str)
                            break
                
                # Start new argument
                arg_name = line[14:].strip()
                # Remove backticks
                arg_name = arg_name.strip('`')
                current_block.arguments.append((arg_name, '', ''))
                current_arg_name = arg_name
                collecting_value = False
                value_lines = []
                i += 1
                continue
            
            elif line.startswith('argument expression:') and current_block.arguments:
                expr = line[20:].strip()
                # Remove backticks
                expr = expr.strip('`')
                # Update last argument
                name, _, val = current_block.arguments[-1]
                current_block.arguments[-1] = (name, expr, val)
                i += 1
                continue
            
            elif line.startswith('argument value:'):
                # Start collecting multi-line value
                collecting_value = True
                value_lines = []
                i += 1
                continue
            
            elif collecting_value:
                # Check if this is a separator (new section)
                if line.startswith('---') or line.startswith('test:') or line.startswith('argument name:'):
                    # Save collected value
                    if current_arg_name and value_lines:
                        value_str = '\n'.join(value_lines)
                        for idx, (name, expr, _) in enumerate(current_block.arguments):
                            if name == current_arg_name:
                                current_block.arguments[idx] = (name, expr, value_str)
                                break
                    collecting_value = False
                    value_lines = []
                    # Don't increment i - reprocess this line
                    continue
                else:
                    # Add to value
                    value_lines.append(line)
                    i += 1
                    continue
        
        i += 1
    
    # Save last block
    if current_block and current_block.test_name:
        # Save any pending value
        if collecting_value and current_arg_name and value_lines:
            value_str = '\n'.join(value_lines)
            for idx, (name, expr, _) in enumerate(current_block.arguments):
                if name == current_arg_name:
                    current_block.arguments[idx] = (name, expr, value_str)
                    break
        blocks.append(current_block)
    
    return blocks


def select_target_and_value(block: CapturedBlock) -> Tuple[Optional[str], Optional[str]]:
    """
    Select which argument to update and which value to use.
    
    Returns: (target_expression, new_value)
    - target_expression: The expression to find and update
    - new_value: The value from the OTHER argument
    
    Strategy (priority order):
    1. EXPECT* prefix in expression - highest priority
    2. Literal expressions (torch.tensor, list, dict, string literals)
    3. 'actual' parameter name (for assert_close confusion)
    4. 'expected' parameter name
    5. First argument (fallback)
    """
    if len(block.arguments) < 2:
        return None, None
    
    # Focus on first 2 arguments
    args = block.arguments[:2]
    
    # Check for EXPECT* prefix (case-insensitive)
    for i, (name, expr, val) in enumerate(args):
        if expr.upper().startswith('EXPECT'):
            # This is the target - use value from OTHER argument
            other_idx = 1 - i
            return expr, args[other_idx][2]
    
    # Check for literal expressions
    literals = []
    for i, (name, expr, val) in enumerate(args):
        if is_literal_expression(expr):
            literals.append(i)
    
    if literals:
        # Use first literal as target
        target_idx = literals[0]
        other_idx = 1 - target_idx
        return args[target_idx][1], args[other_idx][2]
    
    # Check for 'actual' parameter name (assert_close confusion)
    for i, (name, expr, val) in enumerate(args):
        if name == 'actual':
            other_idx = 1 - i
            return expr, args[other_idx][2]
    
    # Check for 'expected' parameter name
    for i, (name, expr, val) in enumerate(args):
        if 'expect' in name.lower():
            other_idx = 1 - i
            return expr, args[other_idx][2]
    
    # Fallback: first argument
    return args[0][1], args[1][2]


def is_literal_expression(expr: str) -> bool:
    """Check if expression is a literal (inline constant)"""
    expr = expr.strip()
    return (expr.startswith('torch.tensor(') or
            expr.startswith('Expectations(') or
            expr.startswith('[') or
            expr.startswith('{') or
            expr.startswith('"') or
            expr.startswith("'"))


def should_skip_block(block: CapturedBlock) -> Tuple[bool, str]:
    """
    Determine if block should be skipped (cross-file helper function).
    
    Returns: (should_skip, reason)
    """
    # Extract context file from "test context:" line
    # Format: /path/to/file.py:LINE or path/to/file.py:LINE
    if not block.test_context:
        return False, ""
    
    context_parts = block.test_context.split(':')
    if len(context_parts) < 2:
        return False, ""
    
    context_file = context_parts[0]
    
    # Normalize paths for comparison
    # test_file: tests/models/MODEL/test_file.py
    # context_file might be:
    #   - /transformers/tests/models/MODEL/test_file.py (absolute)
    #   - models/MODEL/test_file.py (relative, missing tests/)
    #   - tests/models/MODEL/test_file.py (correct relative)
    
    # Extract the path starting from 'tests/'
    if 'tests/' in context_file:
        # Find 'tests/' and take everything from there
        idx = context_file.find('tests/')
        context_file = context_file[idx:]
    elif context_file.startswith('models/'):
        # Missing 'tests/' prefix
        context_file = 'tests/' + context_file
    
    if context_file != block.test_file:
        return True, f"Cross-file: test={block.test_file}, context={context_file}"
    
    return False, ""


# ============================================================================
# PHASE 2: CST ANALYSIS (SINGLE PARSE)
# ============================================================================

class TestFileAnalyzer(cst.CSTVisitor):
    """
    Single-pass CST visitor that finds all variable definitions and matches them to tasks.
    
    This visitor:
    1. Tracks which test method we're in
    2. Collects ALL assignments with their positions
    3. After visiting a method, matches tasks to assignments (backward trace)
    4. Extracts exact value positions for matched assignments
    """
    
    METADATA_DEPENDENCIES = (PositionProvider,)
    
    def __init__(self, tasks: List[UpdateTask]):
        super().__init__()
        self.tasks = tasks
        self.results: Dict[str, ReplacementPosition] = {}
        
        # Track current method context
        self.current_method: Optional[cst.FunctionDef] = None
        self.current_method_name: Optional[str] = None
        self.current_method_start_line: int = 0
        self.current_method_end_line: int = 0
        
        # Collect assignments: {method_name: [(var_name, line, value_node, assign_node, assign_pos), ...]}
        self.method_assignments: Dict[str, List[Tuple]] = {}
        
        # Track method line ranges: {method_name: (start_line, end_line)}
        self.method_ranges: Dict[str, Tuple[int, int]] = {}
    
    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        """Track which test method we're in"""
        try:
            pos = self.get_metadata(PositionProvider, node)
            self.current_method = node
            self.current_method_name = node.name.value
            self.current_method_start_line = pos.start.line
            # We'll set end_line in leave_FunctionDef
            
            # Initialize assignment list for this method
            if self.current_method_name not in self.method_assignments:
                self.method_assignments[self.current_method_name] = []
        except KeyError:
            pass
    
    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
        """Process tasks when leaving a test method"""
        if node == self.current_method:
            try:
                pos = self.get_metadata(PositionProvider, node)
                self.current_method_end_line = pos.end.line
                
                # Store method range
                self.method_ranges[self.current_method_name] = (
                    self.current_method_start_line,
                    self.current_method_end_line
                )
                
                # Match tasks that belong to THIS method
                self._match_tasks_for_method(self.current_method_name)
                
            except KeyError:
                pass
            
            # Reset method context
            self.current_method = None
            self.current_method_name = None
            self.current_method_start_line = 0
            self.current_method_end_line = 0
    
    def _match_tasks_for_method(self, method_name: str) -> None:
        """
        Match tasks to assignments for a specific method.
        
        Only processes tasks where the assertion line falls within this method's line range.
        """
        if method_name not in self.method_ranges:
            return
        
        start_line, end_line = self.method_ranges[method_name]
        assignments = self.method_assignments[method_name]
        
        # Find tasks that belong to this method
        method_tasks = [
            task for task in self.tasks
            if not task.is_inline and start_line <= task.assertion_line <= end_line
        ]
        
        if not method_tasks:
            return
        
        for task in method_tasks:
            # Find assignments of this variable that occur BEFORE the assertion
            candidates = [
                (var_name, line, value_node, assign_node, assign_pos)
                for var_name, line, value_node, assign_node, assign_pos in assignments
                if var_name == task.variable_name and line < task.assertion_line
            ]
            
            if not candidates:
                continue
            
            # Take the CLOSEST assignment before the assertion (backward trace!)
            var_name, line, value_node, assign_node, assign_pos = max(
                candidates,
                key=lambda x: x[1]  # Sort by line number
            )
            
            # Check if this is a DERIVED value from a method call
            # Example: EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
            # We need to update EXPECTED_TEXTS, not EXPECTED_TEXT
            if isinstance(value_node, cst.Call):
                if isinstance(value_node.func, cst.Attribute):
                    # This is ANY method call: something.method()
                    # Trace back to the source variable
                    if isinstance(value_node.func.value, cst.Name):
                        source_var = value_node.func.value.value
                        
                        # Find source variable assignment
                        source_candidates = [
                            (v, l, val, node, pos)
                            for v, l, val, node, pos in assignments
                            if v == source_var and l < line  # Before the derived assignment
                        ]
                        
                        if source_candidates:
                            # Use the SOURCE variable instead!
                            var_name, line, value_node, assign_node, assign_pos = max(
                                source_candidates,
                                key=lambda x: x[1]
                            )
                            
                            # Update the task to reflect the actual variable we're updating
                            task.variable_name = source_var
            
            # Extract the exact position of the VALUE to replace
            value_pos = self._extract_value_position(value_node, assign_pos)
            if value_pos:
                self.results[task.variable_name] = value_pos
    
    def visit_Assign(self, node: cst.Assign) -> None:
        """Collect all variable assignments in the current method"""
        if not self.current_method:
            return
        
        try:
            pos = self.get_metadata(PositionProvider, node)
            
            # Get variable name
            if not isinstance(node.targets[0].target, cst.Name):
                return
            var_name = node.targets[0].target.value
            
            # Store this assignment
            self.method_assignments[self.current_method_name].append(
                (var_name, pos.start.line, node.value, node, pos)
            )
        
        except KeyError:
            pass
    
    def visit_Call(self, node: cst.Call) -> None:
        """Find inline expressions in assertion calls"""
        if not self.current_method:
            return
        
        try:
            pos = self.get_metadata(PositionProvider, node)
            
            # Check if this call is on an assertion line we care about
            for task in self.tasks:
                if not task.is_inline:
                    continue
                
                if pos.start.line != task.assertion_line:
                    continue
                
                # This is an assertion call with an inline expression
                # Find the argument that matches our target
                for arg in node.args:
                    arg_info = self._check_inline_argument(arg, task.variable_name)
                    if arg_info:
                        self.results[task.variable_name] = arg_info
                        return
        
        except KeyError:
            pass
    
    def _extract_value_position(
        self,
        value_node: cst.BaseExpression,
        assign_pos: Any
    ) -> Optional[ReplacementPosition]:
        """
        Extract exact position info from a value node.
        
        This is the KEY function that handles all patterns:
        - torch.tensor([...]) -> extract position of [...] only
        - Expectations({...}) -> extract position of {...} only  
        - [...] lists -> extract full position
        - {...} dicts -> extract full position
        - "..." strings -> extract full position
        - 1.3 numbers -> extract full position
        
        Args:
            value_node: The RHS of the assignment
            assign_pos: Position metadata of the assignment node
        
        Returns:
            ReplacementPosition with exact start/end positions
        """
        try:
            # Unwrap chained calls first (e.g., tensor(...).to(...).float())
            unwrapped = value_node
            while isinstance(unwrapped, cst.Call) and isinstance(unwrapped.func, cst.Attribute):
                if isinstance(unwrapped.func.value, cst.Call):
                    unwrapped = unwrapped.func.value
                else:
                    break
            
            # Handle torch.tensor(data, ...)
            if isinstance(unwrapped, cst.Call):
                if self._is_torch_tensor_call(unwrapped):
                    # Get first argument (the data)
                    if unwrapped.args:
                        data_node = unwrapped.args[0].value
                        data_pos = self.get_metadata(PositionProvider, data_node)
                        
                        return ReplacementPosition(
                            start_line=data_pos.start.line,
                            start_col=data_pos.start.column,
                            end_line=data_pos.end.line,
                            end_col=data_pos.end.column,
                            pattern_type="torch.tensor",
                            indent_level=assign_pos.start.column
                        )
                
                # Handle Expectations(device_dict)
                elif self._is_expectations_call(unwrapped):
                    if unwrapped.args:
                        dict_node = unwrapped.args[0].value
                        
                        # For Expectations, we need to find the specific device key
                        # that matches our target device
                        # For now, extract the whole dict position
                        dict_pos = self.get_metadata(PositionProvider, dict_node)
                        
                        return ReplacementPosition(
                            start_line=dict_pos.start.line,
                            start_col=dict_pos.start.column,
                            end_line=dict_pos.end.line,
                            end_col=dict_pos.end.column,
                            pattern_type="Expectations",
                            indent_level=assign_pos.start.column
                        )
            
            # Handle plain values: lists, dicts, strings, numbers
            if isinstance(unwrapped, cst.List):
                value_pos = self.get_metadata(PositionProvider, unwrapped)
                return ReplacementPosition(
                    start_line=value_pos.start.line,
                    start_col=value_pos.start.column,
                    end_line=value_pos.end.line,
                    end_col=value_pos.end.column,
                    pattern_type="list",
                    indent_level=assign_pos.start.column
                )
            
            elif isinstance(unwrapped, cst.Dict):
                value_pos = self.get_metadata(PositionProvider, unwrapped)
                return ReplacementPosition(
                    start_line=value_pos.start.line,
                    start_col=value_pos.start.column,
                    end_line=value_pos.end.line,
                    end_col=value_pos.end.column,
                    pattern_type="dict",
                    indent_level=assign_pos.start.column
                )
            
            elif isinstance(unwrapped, (cst.SimpleString, cst.ConcatenatedString)):
                value_pos = self.get_metadata(PositionProvider, unwrapped)
                return ReplacementPosition(
                    start_line=value_pos.start.line,
                    start_col=value_pos.start.column,
                    end_line=value_pos.end.line,
                    end_col=value_pos.end.column,
                    pattern_type="string",
                    indent_level=assign_pos.start.column
                )
            
            elif isinstance(unwrapped, (cst.Integer, cst.Float)):
                value_pos = self.get_metadata(PositionProvider, unwrapped)
                return ReplacementPosition(
                    start_line=value_pos.start.line,
                    start_col=value_pos.start.column,
                    end_line=value_pos.end.line,
                    end_col=value_pos.end.column,
                    pattern_type="number",
                    indent_level=assign_pos.start.column
                )
            
            elif isinstance(unwrapped, cst.Tuple):
                value_pos = self.get_metadata(PositionProvider, unwrapped)
                return ReplacementPosition(
                    start_line=value_pos.start.line,
                    start_col=value_pos.start.column,
                    end_line=value_pos.end.line,
                    end_col=value_pos.end.column,
                    pattern_type="tuple",
                    indent_level=assign_pos.start.column
                )
        
        except KeyError:
            pass
        
        return None
    
    def _check_inline_argument(
        self,
        arg: cst.Arg,
        target_expr: str
    ) -> Optional[ReplacementPosition]:
        """Check if this argument matches the target inline expression"""
        try:
            # Check for torch.tensor inline
            if 'torch.tensor(' in target_expr:
                if isinstance(arg.value, cst.Call) and self._is_torch_tensor_call(arg.value):
                    if arg.value.args:
                        data_node = arg.value.args[0].value
                        data_pos = self.get_metadata(PositionProvider, data_node)
                        arg_pos = self.get_metadata(PositionProvider, arg.value)
                        
                        return ReplacementPosition(
                            start_line=data_pos.start.line,
                            start_col=data_pos.start.column,
                            end_line=data_pos.end.line,
                            end_col=data_pos.end.column,
                            pattern_type="torch.tensor_inline",
                            indent_level=arg_pos.start.column
                        )
            
            # Check for string literal inline
            elif target_expr.startswith('"') or target_expr.startswith("'"):
                if isinstance(arg.value, (cst.SimpleString, cst.ConcatenatedString)):
                    arg_pos = self.get_metadata(PositionProvider, arg.value)
                    return ReplacementPosition(
                        start_line=arg_pos.start.line,
                        start_col=arg_pos.start.column,
                        end_line=arg_pos.end.line,
                        end_col=arg_pos.end.column,
                        pattern_type="string_inline",
                        indent_level=arg_pos.start.column
                    )
            
            # Check for list literal inline
            elif target_expr.startswith('['):
                if isinstance(arg.value, cst.List):
                    arg_pos = self.get_metadata(PositionProvider, arg.value)
                    return ReplacementPosition(
                        start_line=arg_pos.start.line,
                        start_col=arg_pos.start.column,
                        end_line=arg_pos.end.line,
                        end_col=arg_pos.end.column,
                        pattern_type="list_inline",
                        indent_level=arg_pos.start.column
                    )
        
        except KeyError:
            pass
        
        return None
    
    def _is_torch_tensor_call(self, node: cst.Call) -> bool:
        """Check if this is a torch.tensor(...) call"""
        if isinstance(node.func, cst.Attribute):
            if isinstance(node.func.value, cst.Name):
                return (node.func.value.value == "torch" and
                       node.func.attr.value == "tensor")
        return False
    
    def _is_expectations_call(self, node: cst.Call) -> bool:
        """Check if this is an Expectations(...) call"""
        if isinstance(node.func, cst.Name):
            return node.func.value == "Expectations"
        return False


def analyze_test_file(filepath: str, tasks: List[UpdateTask]) -> Dict[str, ReplacementPosition]:
    """
    Analyze test file with CST to find exact positions for all updates.
    
    This function:
    1. Parses the file ONCE with CST
    2. Visits all nodes to find variable definitions and inline expressions
    3. Returns exact positions for all requested updates
    
    Args:
        filepath: Path to test file
        tasks: List of UpdateTask objects (what to find)
    
    Returns:
        Dictionary mapping variable_name -> ReplacementPosition
    """
    with open(filepath) as f:
        code = f.read()
    
    tree = cst.parse_module(code)
    wrapper = cst.metadata.MetadataWrapper(tree)
    
    analyzer = TestFileAnalyzer(tasks)
    wrapper.visit(analyzer)
    
    return analyzer.results


# ============================================================================
# PHASE 3: COMPUTE REPLACEMENTS (STRING OPERATIONS ONLY)
# ============================================================================

def find_matching_device_key_in_expectations(dict_text: str, device: tuple) -> tuple:
    """
    Find the matching device key in an Expectations dict using fallback logic.
    
    Args:
        dict_text: The dict content as text
        device: Target device tuple, e.g. ("cuda", (8, 6))
    
    Returns:
        Matching key tuple or None
    
    Fallback order for device=("cuda", (8, 6)):
    1. ("cuda", (8, 6))
    2. ("cuda", 8) 
    3. ("cuda", None)
    4. (None, None)
    """
    device_str, version = device
    
    # Build fallback candidates
    candidates = []
    if version:
        if isinstance(version, tuple):
            # ("cuda", (8, 6))
            candidates.append(f'("{device_str}", {version})')
            # ("cuda", 8) - first element of version tuple
            candidates.append(f'("{device_str}", {version[0]})')
        else:
            # ("cuda", 8)
            candidates.append(f'("{device_str}", {version})')
    
    # ("cuda", None)
    candidates.append(f'("{device_str}", None)')
    # (None, None)
    candidates.append('(None, None)')
    
    # Find which candidate exists in the dict
    for candidate in candidates:
        if candidate in dict_text:
            return candidate
    
    return None


def update_expectations_dict(dict_text: str, new_value: str, device: tuple) -> str:
    """
    Update a specific device key in an Expectations dict.
    
    Args:
        dict_text: The current dict as text
        new_value: The new value to set
        device: Device tuple for key matching
    
    Returns:
        Updated dict text
    """
    # Find matching key
    matching_key = find_matching_device_key_in_expectations(dict_text, device)
    
    if not matching_key:
        # No matching key found - return dict with just our device key
        device_str, version = device
        if version and isinstance(version, tuple):
            key_str = f'("{device_str}", {version})'
        elif version:
            key_str = f'("{device_str}", {version})'
        else:
            key_str = f'("{device_str}", None)'
        
        return f'{{{key_str}: {new_value}}}'
    
    # Find the value for this key
    # Pattern: key: value,  or  key: value}
    key_pos = dict_text.find(matching_key)
    if key_pos == -1:
        return dict_text
    
    # Find the colon after the key
    colon_pos = dict_text.find(':', key_pos)
    if colon_pos == -1:
        return dict_text
    
    # Find where the value ends (either comma or closing brace)
    # Need to handle nested brackets
    value_start = colon_pos + 1
    
    # Skip whitespace
    while value_start < len(dict_text) and dict_text[value_start] in ' \n\t':
        value_start += 1
    
    # Find value end by counting brackets
    bracket_count = 0
    in_string = False
    string_char = None
    i = value_start
    
    while i < len(dict_text):
        char = dict_text[i]
        
        # Handle strings
        if char in '"\'':
            if not in_string:
                in_string = True
                string_char = char
            elif char == string_char and (i == 0 or dict_text[i-1] != '\\'):
                in_string = False
                string_char = None
        
        if not in_string:
            if char in '[{(':
                bracket_count += 1
            elif char in ']}':
                bracket_count -= 1
                if bracket_count < 0:
                    # Reached end of dict
                    break
            elif char == ',' and bracket_count == 0:
                # Reached end of this entry
                break
        
        i += 1
    
    value_end = i
    
    # Replace the value
    before = dict_text[:value_start]
    after = dict_text[value_end:]
    
    return before + new_value + after


def compute_replacement_text(new_value: str, position: ReplacementPosition, is_multiline: bool, device: tuple = ("cuda", (8, 6)), original_text: str = None) -> str:
    """
    Compute the exact text to replace at the given position.
    
    This uses ONLY string operations - no CST, no AST, no regex.
    
    Args:
        new_value: The new value from captured_info (may include formatting)
        position: The exact position to replace
        is_multiline: Whether the new value spans multiple lines
        device: Device tuple for Expectations matching
        original_text: Original file text (needed for Expectations)
    
    Returns:
        The formatted text ready to insert
    """
    # Strip the value
    new_value = new_value.strip()
    value_lines = new_value.split('\n')
    
    # Special handling for Expectations pattern
    if position.pattern_type == "Expectations":
        if original_text:
            # Extract the current dict text
            lines = original_text.split('\n')
            start_line_idx = position.start_line - 1
            end_line_idx = position.end_line - 1
            
            
            if start_line_idx == end_line_idx:
                dict_text = lines[start_line_idx][position.start_col:position.end_col]
            else:
                # Multi-line dict
                dict_lines = []
                dict_lines.append(lines[start_line_idx][position.start_col:])
                for i in range(start_line_idx + 1, end_line_idx):
                    dict_lines.append(lines[i])
                dict_lines.append(lines[end_line_idx][:position.end_col])
                dict_text = '\n'.join(dict_lines)
            
            # Update the specific device key
            updated_dict = update_expectations_dict(dict_text, new_value, device)
            return updated_dict
        else:
            # Fallback: create new dict with device key
            device_str, version = device
            if version and isinstance(version, tuple):
                key_str = f'("{device_str}", {version})'
            elif version:
                key_str = f'("{device_str}", {version})'
            else:
                key_str = f'("{device_str}", None)'
            
            if len(value_lines) == 1:
                return f'{{{key_str}: {new_value}}}'
            else:
                result = ['{']
                result.append(f'    {key_str}: {value_lines[0]}')
                for line in value_lines[1:]:
                    result.append(f'        {line.strip()}')
                result.append('}')
                return '\n'.join(result)
    
    if len(value_lines) == 1:
        # Single-line replacement - just return the value
        return new_value
    
    # Multi-line replacement - need to handle indentation
    
    if position.pattern_type in ["torch.tensor", "torch.tensor_inline"]:
        # For torch.tensor, the first line is opening bracket
        # Middle lines need indent_level + 4
        # Last line needs indent_level
        
        result_lines = []
        result_lines.append(value_lines[0])  # Opening bracket line
        
        # Middle lines
        inner_indent = ' ' * (position.indent_level + 4)
        for line in value_lines[1:-1]:
            content = line.strip()
            result_lines.append(inner_indent + content)
        
        # Closing bracket line
        close_indent = ' ' * position.indent_level
        result_lines.append(close_indent + value_lines[-1].strip())
        
        return '\n'.join(result_lines)
    
    else:
        # For other patterns, preserve the formatting from captured_info
        # but adjust base indentation
        result_lines = []
        base_indent = ' ' * position.indent_level
        
        for line in value_lines:
            # Strip and re-indent
            content = line.strip()
            if content:
                result_lines.append(base_indent + content)
            else:
                result_lines.append('')
        
        return '\n'.join(result_lines)


# ============================================================================
# PHASE 4: APPLY REPLACEMENTS (IN-MEMORY, SINGLE WRITE)
# ============================================================================

def apply_replacements(filepath: str, replacements: List[Replacement], dry_run: bool = True) -> bool:
    """
    Apply all replacements to the file.
    
    Algorithm:
    1. Read file into memory
    2. Sort replacements by position (highest first)
    3. Apply each replacement as string operation
    4. Write file once
    
    The key insight: by processing from highest position to lowest,
    each replacement doesn't affect the positions of subsequent replacements.
    
    Args:
        filepath: Path to file
        replacements: List of Replacement objects
        dry_run: If True, don't actually write the file
    
    Returns:
        True if successful
    """
    # Read file
    with open(filepath) as f:
        lines = f.readlines()
    
    # Sort replacements by position (highest line first, then highest column)
    # This ensures we process from end to beginning
    replacements_sorted = sorted(
        replacements,
        key=lambda r: (r.position.start_line, r.position.start_col),
        reverse=True
    )
    
    # Apply each replacement
    for replacement in replacements_sorted:
        pos = replacement.position
        
        # Convert to 0-indexed
        start_idx = pos.start_line - 1
        end_idx = pos.end_line - 1
        
        if start_idx == end_idx:
            # Single-line replacement
            line = lines[start_idx]
            new_line = (line[:pos.start_col] + 
                       replacement.new_text + 
                       line[pos.end_col:])
            lines[start_idx] = new_line
        
        else:
            # Multi-line replacement
            # Take prefix from first line, suffix from last line
            prefix = lines[start_idx][:pos.start_col]
            suffix = lines[end_idx][pos.end_col:]
            
            # Combine
            new_content = prefix + replacement.new_text + suffix
            
            # Split into lines (need to preserve newlines for writelines)
            new_lines = new_content.split('\n')
            # Add back newlines (except for the last line which gets suffix's newline)
            new_lines_with_newlines = [line + '\n' for line in new_lines[:-1]]
            # Last line keeps its original ending
            if new_lines:
                new_lines_with_newlines.append(new_lines[-1])
            
            # Replace the span
            lines[start_idx:end_idx + 1] = new_lines_with_newlines
    
    # Write file (if not dry-run)
    if not dry_run:
        with open(filepath, 'w') as f:
            f.writelines(lines)
    
    return True


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def process_captured_info(captured_file: str, apply: bool = False, device: tuple = ("cuda", (8, 6))):
    """
    Main entry point that orchestrates the entire update process.
    
    Args:
        captured_file: Path to captured_info.txt
        apply: Whether to apply changes or dry-run
        device: Device tuple for Expectations matching, e.g. ("cuda", (8, 6))
    
    Workflow:
    1. Parse captured_info.txt
    2. Group tasks by file
    3. For each file:
       a. Analyze with CST (single parse)
       b. Compute replacements
       c. Apply replacements
    """
    print("=" * 80)
    print("CST-based Expectations Updater v2.0")
    print("=" * 80)
    print()
    
    # Phase 1: Parse captured_info.txt
    print("[1] Parsing captured_info.txt...")
    blocks = parse_captured_info(captured_file)
    print(f"    Found {len(blocks)} assertion failure blocks")
    
    # Convert blocks to tasks
    tasks_by_file: Dict[str, List[UpdateTask]] = {}
    skipped = 0
    
    for block in blocks:
        # Check if should skip
        should_skip, reason = should_skip_block(block)
        if should_skip:
            print(f"    Skip: {block.test_name}")
            print(f"          {reason}")
            skipped += 1
            continue
        
        # Select target and value
        target_expr, new_value = select_target_and_value(block)
        if not target_expr or not new_value:
            print(f"    Skip: Could not select target for {block.test_name}")
            skipped += 1
            continue
        
        # Create task
        task = UpdateTask(
            variable_name=target_expr,
            new_value=new_value,
            assertion_line=block.assertion_line,
            is_inline=is_literal_expression(target_expr)
        )
        
        # Group by file
        if block.test_file not in tasks_by_file:
            tasks_by_file[block.test_file] = []
        tasks_by_file[block.test_file].append(task)
    
    print(f"    Created {sum(len(t) for t in tasks_by_file.values())} update tasks")
    print(f"    Skipped {skipped} blocks")
    print(f"    Grouped into {len(tasks_by_file)} file(s)")
    print()
    
    # Phase 2-4: Process each file
    print("[2] Processing files...")
    
    for filepath, tasks in tasks_by_file.items():
        print(f"\n  File: {filepath}")
        print(f"  Tasks: {len(tasks)}")
        
        # Phase 2: CST Analysis
        print("  [2a] Analyzing with CST...")
        positions = analyze_test_file(filepath, tasks)
        print(f"       Found positions for {len(positions)}/{len(tasks)} tasks")
        
        # Read file for Expectations processing
        with open(filepath) as f:
            original_text = f.read()
        
        # Phase 3: Compute Replacements
        print("  [2b] Computing replacements...")
        replacements = []
        
        for task in tasks:
            if task.variable_name not in positions:
                print(f"       ✗ No position found for {task.variable_name}")
                continue
            
            pos = positions[task.variable_name]
            is_multi = '\n' in task.new_value
            
            new_text = compute_replacement_text(
                task.new_value, 
                pos, 
                is_multi,
                device=device,
                original_text=original_text
            )
            
            replacement = Replacement(
                position=pos,
                new_text=new_text
            )
            replacements.append(replacement)
            
            print(f"       ✓ {task.variable_name} at line {pos.start_line}")
        
        # Phase 4: Apply Replacements
        print("  [2c] Applying replacements...")
        if apply:
            success = apply_replacements(filepath, replacements, dry_run=False)
            if success:
                print(f"       ✅ Updated {len(replacements)} locations")
            else:
                print(f"       ✗ Failed to apply replacements")
        else:
            print(f"       💡 DRY RUN - Would update {len(replacements)} locations")
    
    print()
    print("=" * 80)
    print("✅ Complete!")
    print("=" * 80)
    
    mode = "APPLY" if apply else "DRY RUN"
    print(f"\nMode: {mode}")
    print(f"Files processed: {len(tasks_by_file)}")
    print(f"Total tasks: {sum(len(t) for t in tasks_by_file.values())}")


def main():
    parser = argparse.ArgumentParser(description="Update test expectations")
    parser.add_argument("captured_info", help="Path to captured_info.txt")
    parser.add_argument("--apply", action="store_true", help="Apply changes (default is dry run)")
    parser.add_argument("--device", default="cuda:8.6", 
                       help="Device for Expectations matching (format: cuda:8.6, cuda:8, cuda, etc.)")
    
    args = parser.parse_args()
    
    # Parse device argument
    device_parts = args.device.split(':')
    device_str = device_parts[0]
    if len(device_parts) > 1:
        version_str = device_parts[1]
        if '.' in version_str:
            # cuda:8.6 -> ("cuda", (8, 6))
            major, minor = version_str.split('.')
            device = (device_str, (int(major), int(minor)))
        else:
            # cuda:8 -> ("cuda", 8)
            device = (device_str, int(version_str))
    else:
        # cuda -> ("cuda", None)
        device = (device_str, None)
    
    process_captured_info(args.captured_info, args.apply, device)


if __name__ == "__main__":
    main()
