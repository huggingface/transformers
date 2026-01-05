#!/usr/bin/env python3
"""
CST-based Test Expectations Updater - Version 2.0

Automatically updates test expectations from captured runtime output, handling
device-specific values for different hardware configurations (CUDA, ROCm, XPU, etc.).

================================================================================
DESIGN PRINCIPLES
================================================================================

1. PURE CST FOR ALL POSITION DETECTION
   - Parse each test file ONCE with LibCST
   - Use CST metadata (PositionProvider) for exact character positions
   - For Expectations: CST drills into Dict.elements to find matching device key
   - No string searching, no regex, no ad-hoc text parsing

2. SIMPLE STRING OPERATIONS FOR REPLACEMENT TEXT
   - All formatting uses basic Python: .strip(), .split(), ' ' * indent
   - No AST manipulation, no regex, no complex parsing
   - Clean, readable, maintainable code

3. ELEGANT CHARACTER-BASED REPLACEMENT
   - Convert CST (line, col) positions to character offsets
   - Sort replacements by position (ascending)
   - Apply all replacements in one pass with replace_substrings()
   - Single file read, single file write
   - No position tracking, no reverse processing

================================================================================
WORKFLOW
================================================================================

Phase 1: Parse captured_info.txt
  • Extract all assertion failures with their context
  • Identify target variable (EXPECT* prefix has priority)
  • Extract new value from the OTHER argument

Phase 2: CST Analysis (PURE CST)
  • Parse test file once with LibCST + PositionProvider
  • Track method boundaries and collect ALL assignments
  • Backward trace: Find variable definition before assertion
  • Handle derived values: EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
  • For Expectations: Use CST to find matching device key in Dict
  • Extract EXACT value positions (start/end line and column)

Phase 3: Compute Replacements (SIMPLE STRINGS)
  • Format new value with proper indentation
  • Handle single-line vs multi-line
  • Pattern-specific formatting (torch.tensor gets special treatment)
  • Pure string operations: no regex, no complex logic

Phase 4: Apply Replacements (ELEGANT ALGORITHM)
  • Read file as single string
  • Convert all (line, col) to character positions
  • Sort by character position (ascending)
  • Apply all with replace_substrings() in one pass
  • Write file once

================================================================================
USAGE
================================================================================

    # Dry run (preview)
    python3 auto_update.py captured/gemma3/captured_info.txt
    
    # Apply changes
    python3 auto_update.py captured/gemma3/captured_info.txt --apply
    
    # Specify device for Expectations
    python3 auto_update.py captured/gemma3/captured_info.txt --device cuda:8.6 --apply

================================================================================
"""

import argparse
import libcst as cst
from libcst.metadata import PositionProvider
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass


@dataclass
class CapturedBlock:
    """
    Represents a single assertion failure block from captured_info.txt.
    
    Each block contains information about one test failure, including the
    test location, assertion code, and the actual vs expected argument values.
    
    Attributes:
        test_name: Full qualified test name (format: path::Class::method)
                   Example: "tests/models/clip/test_modeling_clip.py::CLIPTest::test_forward"
        test_file: Path to the test file relative to repository root
                   Example: "tests/models/clip/test_modeling_clip.py"
        test_context: Context line showing where the assertion failed
                      Format: "path:LINE_NUMBER"
        assertion_line: Line number where the assertion is located (1-indexed)
        assertion_code: The actual assertion code that failed
                        Example: "self.assertEqual(output, EXPECTED)"
        arguments: List of (name, expression, value) tuples for assertion arguments
                   Example: [("list1", "output", "[1, 2, 3]"),
                            ("list2", "EXPECTED", "[0, 0, 0]")]
    """
    test_name: str
    test_file: str
    test_context: str
    assertion_line: int
    assertion_code: str
    arguments: List[Tuple[str, str, str]]


@dataclass
class UpdateTask:
    """
    Information about what variable to update and where.
    
    Created after parsing a CapturedBlock and selecting which variable
    should be updated (typically the one with EXPECT* prefix).
    
    Attributes:
        variable_name: Name of the variable to update
                       Example: "EXPECTED_OUTPUT" or "expected_slice"
                       For inline literals: The expression itself like "torch.tensor([1,2])"
        new_value: The new value from captured_info (actual runtime output)
                   This is the raw string from the captured_info.txt file
        assertion_line: Line number where the assertion that failed is located
                        Used to scope the backward search for variable definition
        is_inline: True if this is an inline literal in the assertion
                   Example: assert_close(output, torch.tensor([1, 2]))
                   False if it's a variable: assert_close(output, EXPECTED)
    """
    variable_name: str
    new_value: str
    assertion_line: int
    is_inline: bool


@dataclass
class ReplacementPosition:
    """
    Exact character position of a value to replace in the test file.
    
    All positions are extracted using LibCST's PositionProvider metadata.
    These positions point to ONLY the value part, not the variable assignment.
    
    Examples:
        - For 'EXPECTED = torch.tensor([[1, 2]])', position points to '[[1, 2]]'
        - For 'EXPECTED = Expectations({("cuda", (8,6)): [1,2]})', position points to '[1,2]'
        - For 'EXPECTED = [1, 2, 3]', position points to '[1, 2, 3]'
    
    Attributes:
        start_line: Starting line number (1-indexed, as reported by CST)
        start_col: Starting column number (0-indexed, as reported by CST)
        end_line: Ending line number (1-indexed)
        end_col: Ending column number (0-indexed, exclusive)
        pattern_type: Type of pattern for formatting logic
                      Values: "torch.tensor", "Expectations", "list", "dict", 
                              "string", "number", "tuple", "torch.tensor_inline",
                              "string_inline", "list_inline"
        indent_level: Base indentation level (number of spaces) where the opening
                      bracket/brace SHOULD be positioned in multi-line format
        is_function_call: True if value is inside a function call (torch.tensor, Expectations, etc.)
                          False for direct assignments (plain list, dict, etc.)
        has_multiline_structure: True if original has opening/closing on separate lines
                                 For function calls: whether ( and ) are on different lines
                                 For direct assignments: whether opening bracket is on assignment line
        is_data_inline: True if data/value starts on same line as function call opening (
                        Only relevant for function calls
        assignment_indent: Column position of the assignment statement (for reference)
    """
    start_line: int
    start_col: int
    end_line: int
    end_col: int
    pattern_type: str
    indent_level: int
    is_function_call: bool = False
    has_multiline_structure: bool = False
    is_data_inline: bool = False
    assignment_indent: int = 0


@dataclass
class Replacement:
    """
    A computed replacement ready to be applied to the file.
    
    Contains the exact position (from ReplacementPosition) and the
    formatted replacement text (computed with proper indentation).
    
    Attributes:
        position: ReplacementPosition object with exact coordinates
        new_text: Formatted replacement text, ready to insert at position
                  This text has proper indentation and line breaks applied
    """
    position: ReplacementPosition
    new_text: str


# ============================================================================
# PHASE 1: PARSE CAPTURED_INFO.TXT
# ============================================================================

def parse_captured_info(filepath: str) -> List[CapturedBlock]:
    """
    Parse captured_info.txt and extract all assertion failure blocks.
    
    The captured_info.txt file contains detailed information about test failures,
    including the test name, assertion location, and the actual vs expected values.
    This function parses that format into structured CapturedBlock objects.
    
    File Format Example:
        test:
        
        tests/models/model/test.py::Class::method
        
        --------------------------------------------------------------------------------
        
        test context: /path/test.py:LINE
        
        assertion_code_here
        
        --------------------------------------------------------------------------------
        
        argument name: `arg1`
        argument expression: `expr1`
        
        argument value:
        
        value1
        
        --------------------------------------------------------------------------------
        
        argument name: `arg2`
        argument expression: `expr2`
        
        argument value:
        
        value2
    
    Args:
        filepath: Path to the captured_info.txt file
    
    Returns:
        List of CapturedBlock objects, one for each assertion failure.
        Each block contains:
        - test name and file path
        - assertion line number and code
        - all arguments with their names, expressions, and values
    
    State Machine:
        The parser uses a state machine to handle multi-line values:
        - Collecting test name after "test:" marker
        - Extracting line number from "test context:"
        - Collecting argument name and expression
        - Accumulating multi-line argument values
        - Detecting section boundaries (separators)
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
    Select which argument to update and which value to use from an assertion failure.
    
    When an assertion fails, we have two arguments: one is the actual output,
    the other is the expected value. We need to determine which is which and
    select the expected value variable as the target to update.
    
    Selection Strategy (Priority Order):
        1. EXPECT* prefix (highest priority)
           - If any expression starts with "EXPECT" (case-insensitive)
           - This is typically the expected value variable
           - Update this using the value from the OTHER argument
           
        2. Literal expressions
           - torch.tensor([...]), [...], {...}, "...", etc.
           - These are inline constants that should be updated
           
        3. 'actual' parameter name
           - For assert_close(actual, expected) style
           - Sometimes the naming is backwards
           
        4. 'expected' parameter name
           - Look for "expect" in the parameter name
           
        5. First argument (fallback)
           - When nothing else matches, assume first arg is the target
    
    Args:
        block: CapturedBlock containing assertion failure information
    
    Returns:
        Tuple of (target_expression, new_value):
        - target_expression: The variable name or expression to find and update
                             Example: "EXPECTED_OUTPUT"
        - new_value: The runtime value from the OTHER argument
                     This is what the target should be updated to
        
        Returns (None, None) if:
        - Block has fewer than 2 arguments
        - Arguments are malformed
    
    Example:
        Given arguments:
        [("list1", "output_text", '["actual"]'),
         ("list2", "EXPECTED_TEXT", '["old_expected"]')]
        
        Returns: ("EXPECTED_TEXT", '["actual"]')
        Because "EXPECTED_TEXT" has EXPECT* prefix, so we update it
        with the value from "output_text"
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
    """
    Check if expression is a literal (inline constant value).
    
    Literal expressions are values written directly in the code, as opposed
    to variables that hold values. These typically appear inline in assertions.
    
    Supported Literals:
        - torch.tensor([...]) or torch.tensor([[...]])
        - Expectations({...})
        - [...] - lists
        - {...} - dictionaries
        - "..." or '...' - strings
    
    Args:
        expr: The expression string to check
    
    Returns:
        True if expression is a recognized literal pattern
        False if it's a variable name or unsupported pattern
    
    Examples:
        >>> is_literal_expression("torch.tensor([1, 2])")
        True
        >>> is_literal_expression("[1, 2, 3]")
        True
        >>> is_literal_expression("EXPECTED_OUTPUT")
        False
        >>> is_literal_expression("output_tensor")
        False
    """
    expr = expr.strip()
    return (expr.startswith('torch.tensor(') or
            expr.startswith('Expectations(') or
            expr.startswith('[') or
            expr.startswith('{') or
            expr.startswith('"') or
            expr.startswith("'"))


def should_skip_block(block: CapturedBlock) -> Tuple[bool, str]:
    """
    Determine if an assertion failure block should be skipped.
    
    Blocks are skipped when the assertion is in a different file than where
    the test is defined. This happens when tests call helper functions that
    contain assertions in other files.
    
    Cross-File Detection:
        The test_file comes from the test name: "tests/models/model/test.py::..."
        The test_context comes from the assertion location: "/path/.../file.py:LINE"
        
        If these point to different files, it's a cross-file assertion that
        we should skip (updating would require finding the helper function).
    
    Args:
        block: CapturedBlock to check
    
    Returns:
        Tuple of (should_skip, reason):
        - should_skip: True if block should be skipped
        - reason: Human-readable explanation of why (empty if not skipped)
    
    Path Normalization:
        Handles various path formats:
        - /full/path/tests/models/model/test.py → tests/models/model/test.py
        - models/model/test.py → tests/models/model/test.py
        - tests/models/model/test.py → tests/models/model/test.py
    
    Example:
        Block with:
        - test_file: "tests/models/clip/test_modeling_clip.py"
        - test_context: "/transformers/tests/models/clip/test_helper.py:42"
        
        Returns: (True, "Cross-file: test=tests/models/clip/test_modeling_clip.py, 
                        context=tests/models/clip/test_helper.py")
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
    CST visitor that finds exact positions of values to update in test files.
    
    This is the CORE of the position detection system. It performs a single-pass
    traversal of the test file's CST to:
    1. Track which test method we're currently in
    2. Collect ALL variable assignments within each method
    3. Match update tasks to their corresponding assignments (backward trace)
    4. Handle derived values (e.g., X = Y.get_expectation())
    5. Extract EXACT character positions of values to replace
    
    The visitor uses LibCST's PositionProvider metadata to get precise line and
    column numbers for all code elements, with no string searching or regex.
    
    Key Design Decisions:
        - Method-scoped analysis: Only look for assignments within the same test method
        - Backward tracing: Find the CLOSEST assignment BEFORE the assertion
        - Derived value detection: If value comes from method call, trace to source
        - Device-aware Expectations: Drill into Dict.elements to find matching key
        - Pure CST: No string operations, all position data from CST metadata
    
    Workflow:
        1. visit_FunctionDef: Enter a test method, note its boundaries
        2. visit_Assign: Collect all assignments (var_name, line, value_node, ...)
        3. leave_FunctionDef: Match tasks to assignments, extract positions
        4. visit_Call: Handle inline literals in assertions
    
    Attributes:
        tasks: List of UpdateTask objects (what to find)
        device: Target device tuple for Expectations matching
               Example: ("cuda", (8, 6))
        results: Output dictionary mapping variable_name → ReplacementPosition
        
        current_method: Currently visited FunctionDef node
        current_method_name: Name of current test method
        current_method_start_line: Starting line of current method
        current_method_end_line: Ending line of current method
        
        method_assignments: Dict[method_name → List[(var, line, value_node, ...)]]
                            Collected assignments per method
        method_ranges: Dict[method_name → (start_line, end_line)]
                       Boundary information per method
    
    Derived Value Handling Example:
        ```python
        EXPECTED_TEXTS = Expectations({...})        # Line 100
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()  # Line 110 - DERIVED
        assert output == EXPECTED_TEXT               # Line 120 - ASSERTION
        ```
        
        Process:
        1. Task targets "EXPECTED_TEXT" at line 120
        2. Find assignment at line 110: EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        3. Detect it's a method call → derived value
        4. Trace back to EXPECTED_TEXTS at line 100
        5. Update task to target EXPECTED_TEXTS instead
        6. Extract position from EXPECTED_TEXTS value
    
    Expectations Device Matching Example:
        ```python
        EXPECTED = Expectations({
            ("cuda", (8, 6)): [value1],  ← Target for device=("cuda", (8, 6))
            ("cuda", (9, 0)): [value2],
        })
        ```
        
        Process:
        1. Detect Expectations(...) call
        2. Get dict_node from args[0].value
        3. Iterate dict_node.elements (DictElement objects)
        4. Parse each key with _parse_cst_tuple_key()
        5. Match key to device with _device_matches() and fallback logic
        6. Extract position of matching value_node using PositionProvider
    """
    
    METADATA_DEPENDENCIES = (PositionProvider,)
    
    def __init__(self, tasks: List[UpdateTask], device: tuple = ("cuda", (8, 6))):
        super().__init__()
        self.tasks = tasks
        self.device = device
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
        Extract exact character position of the VALUE to replace from an assignment.
        
        This is THE CORE position extraction function. For each type of value,
        it extracts the position of ONLY the data part that should be replaced,
        not the wrapper or variable name.
        
        Pattern-Specific Extraction:
        
            1. torch.tensor([data]) → Extract position of [data] only
               Unwraps: tensor(...).to(device).float() chains
               Gets: First argument (the data)
               
            2. Expectations({dict}) → Extract position of MATCHING VALUE only
               Gets: Dictionary argument
               Finds: Matching device key using _find_matching_dict_element()
               Extracts: Position of that key's value
               
            3. Lists, dicts, strings, numbers → Extract full position
               No special handling, use position of entire value
        
        CST Unwrapping:
            For chained method calls like tensor(...).to(...).float(),
            unwraps to find the original tensor(...) call before extracting
            the data argument.
        
        Args:
            value_node: CST node representing the RHS of an assignment
                        Example: For "X = torch.tensor([1, 2])",
                                value_node is the entire RHS expression
            assign_pos: Position metadata of the assignment statement
                       Used to get indent_level for formatting
        
        Returns:
            ReplacementPosition object with:
            - Exact line/column of value to replace
            - Pattern type for formatting logic
            - Indent level for multi-line formatting
            
            Returns None if:
            - Position extraction fails
            - For Expectations: No matching device key found
            - Unsupported pattern type
        
        Example 1 - torch.tensor:
            Code: EXPECTED = torch.tensor([[1, 2], [3, 4]]).to(device)
            Result:
                start_line=10, start_col=26  (start of [[1, 2]])
                end_line=10, end_col=42      (end of ]])
                pattern_type="torch.tensor"
        
        Example 2 - Expectations:
            Code: EXPECTED = Expectations({("cuda", (8,6)): [1,2], ...})
            Device: ("cuda", (8, 6))
            Result:
                start_line=15, start_col=45  (start of [1,2])
                end_line=15, end_col=50      (end of ])
                pattern_type="Expectations"
        
        Example 3 - Plain list:
            Code: EXPECTED = [1, 2, 3]
            Result:
                start_line=20, start_col=13  (start of [1, 2, 3])
                end_line=20, end_col=22      (end of ])
                pattern_type="list"
        
        Note:
            Uses LibCST's PositionProvider.get_metadata() for ALL positions.
            No string operations, no calculations, just CST metadata queries.
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
                        
                        # Get the call node position (for torch.tensor( ... ))
                        call_pos = self.get_metadata(PositionProvider, unwrapped)
                        
                        # Check if this is a multi-line structure:
                        # If the opening ( and closing ) are on different lines
                        has_multiline_structure = call_pos.start.line != call_pos.end.line
                        
                        # Check if data is inline with the opening (
                        # This matters for adding leading newline when converting to multi-line
                        is_data_inline = data_pos.start.line == call_pos.start.line
                        
                        # Compute indent_level:
                        # - If original has multi-line structure, use data_pos column
                        # - If original is single-line, use assignment column + 4
                        if has_multiline_structure:
                            # Data position tells us where content should be indented
                            indent_level = data_pos.start.column
                        else:
                            # For single-line → multi-line conversion, indent is assignment + 4
                            indent_level = assign_pos.start.column + 4
                        
                        return ReplacementPosition(
                            start_line=data_pos.start.line,
                            start_col=data_pos.start.column,
                            end_line=data_pos.end.line,
                            end_col=data_pos.end.column,
                            pattern_type="torch.tensor",
                            indent_level=indent_level,
                            is_function_call=True,
                            has_multiline_structure=has_multiline_structure,
                            is_data_inline=is_data_inline,
                            assignment_indent=assign_pos.start.column
                        )
                
                # Handle Expectations(device_dict)
                elif self._is_expectations_call(unwrapped):
                    if unwrapped.args:
                        dict_node = unwrapped.args[0].value
                        
                        # Use CST to find the matching device key and get its value position
                        matching_element = self._find_matching_dict_element(dict_node)
                        
                        if matching_element:
                            # Extract the exact position of the VALUE only
                            value_node = matching_element.value
                            value_pos = self.get_metadata(PositionProvider, value_node)
                            
                            # Get the call node position
                            call_pos = self.get_metadata(PositionProvider, unwrapped)
                            
                            # Check if this is a multi-line structure
                            has_multiline_structure = call_pos.start.line != call_pos.end.line
                            
                            # Check if value is inline with the opening (
                            is_data_inline = value_pos.start.line == call_pos.start.line
                            
                            # Compute indent_level
                            if has_multiline_structure:
                                indent_level = value_pos.start.column
                            else:
                                indent_level = assign_pos.start.column + 4
                            
                            return ReplacementPosition(
                                start_line=value_pos.start.line,
                                start_col=value_pos.start.column,
                                end_line=value_pos.end.line,
                                end_col=value_pos.end.column,
                                pattern_type="Expectations",
                                indent_level=indent_level,
                                is_function_call=True,
                                has_multiline_structure=has_multiline_structure,
                                is_data_inline=is_data_inline,
                                assignment_indent=assign_pos.start.column
                            )
                        else:
                            # No matching key found - this shouldn't happen but handle gracefully
                            # Return None so this task will be skipped
                            return None
            
            # Handle plain values: lists, dicts, strings, numbers
            if isinstance(unwrapped, cst.List):
                value_pos = self.get_metadata(PositionProvider, unwrapped)
                
                # For plain lists, opening [ stays on assignment line
                # Check if list is already multi-line
                has_multiline_structure = value_pos.start.line != value_pos.end.line
                
                # Indent level is where content inside [ should be
                if has_multiline_structure:
                    # Multi-line: use position of first element if available
                    # For now, use value_pos column (opening [) + some default
                    indent_level = value_pos.start.column + 4
                else:
                    # Single-line or converting to multi-line
                    indent_level = value_pos.start.column + 4
                
                return ReplacementPosition(
                    start_line=value_pos.start.line,
                    start_col=value_pos.start.column,
                    end_line=value_pos.end.line,
                    end_col=value_pos.end.column,
                    pattern_type="list",
                    indent_level=indent_level,
                    is_function_call=False,
                    has_multiline_structure=has_multiline_structure,
                    is_data_inline=True,  # Opening [ is always on assignment line
                    assignment_indent=assign_pos.start.column
                )
            
            elif isinstance(unwrapped, cst.Dict):
                value_pos = self.get_metadata(PositionProvider, unwrapped)
                has_multiline_structure = value_pos.start.line != value_pos.end.line
                indent_level = value_pos.start.column + 4
                
                return ReplacementPosition(
                    start_line=value_pos.start.line,
                    start_col=value_pos.start.column,
                    end_line=value_pos.end.line,
                    end_col=value_pos.end.column,
                    pattern_type="dict",
                    indent_level=indent_level,
                    is_function_call=False,
                    has_multiline_structure=has_multiline_structure,
                    is_data_inline=True,
                    assignment_indent=assign_pos.start.column
                )
            
            elif isinstance(unwrapped, (cst.SimpleString, cst.ConcatenatedString)):
                value_pos = self.get_metadata(PositionProvider, unwrapped)
                has_multiline_structure = value_pos.start.line != value_pos.end.line
                
                return ReplacementPosition(
                    start_line=value_pos.start.line,
                    start_col=value_pos.start.column,
                    end_line=value_pos.end.line,
                    end_col=value_pos.end.column,
                    pattern_type="string",
                    indent_level=assign_pos.start.column,
                    is_function_call=False,
                    has_multiline_structure=has_multiline_structure,
                    is_data_inline=True,
                    assignment_indent=assign_pos.start.column
                )
            
            elif isinstance(unwrapped, (cst.Integer, cst.Float)):
                value_pos = self.get_metadata(PositionProvider, unwrapped)
                
                return ReplacementPosition(
                    start_line=value_pos.start.line,
                    start_col=value_pos.start.column,
                    end_line=value_pos.end.line,
                    end_col=value_pos.end.column,
                    pattern_type="number",
                    indent_level=assign_pos.start.column,
                    is_function_call=False,
                    has_multiline_structure=False,
                    is_data_inline=True,
                    assignment_indent=assign_pos.start.column
                )
            
            elif isinstance(unwrapped, cst.Tuple):
                value_pos = self.get_metadata(PositionProvider, unwrapped)
                has_multiline_structure = value_pos.start.line != value_pos.end.line
                indent_level = value_pos.start.column + 4
                
                return ReplacementPosition(
                    start_line=value_pos.start.line,
                    start_col=value_pos.start.column,
                    end_line=value_pos.end.line,
                    end_col=value_pos.end.column,
                    pattern_type="tuple",
                    indent_level=indent_level,
                    is_function_call=False,
                    has_multiline_structure=has_multiline_structure,
                    is_data_inline=True,
                    assignment_indent=assign_pos.start.column
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
    
    def _parse_cst_tuple_key(self, node: cst.BaseExpression) -> Optional[tuple]:
        """
        Parse a CST tuple node to extract device information from Expectations dict keys.
        
        Expectations dictionaries use tuple keys to specify device configurations:
            {("cuda", (8, 6)): value1, ("rocm", (9, 4)): value2, ...}
        
        This method uses PURE CST inspection to extract these tuples, handling:
        - Device strings: "cuda", "rocm", "xpu", None
        - Version specifications: (8, 6), 8, None
        
        CST Structure:
            Tuple(
                elements=[
                    Element(value=SimpleString(value='"cuda"')),
                    Element(value=Tuple(elements=[...]) or Integer(...) or Name("None"))
                ]
            )
        
        Args:
            node: CST BaseExpression node to parse (should be a Tuple)
        
        Returns:
            Parsed tuple in format (device_str, version) where:
            - device_str: String like "cuda" or None
            - version: Tuple like (8, 6), Integer like 8, or None
            
            Returns None if:
            - Node is not a Tuple
            - Tuple doesn't have exactly 2 elements
            - Elements can't be parsed
        
        Examples:
            >>> # CST node for ("cuda", (8, 6))
            >>> result = _parse_cst_tuple_key(cst_node)
            >>> print(result)
            ("cuda", (8, 6))
            
            >>> # CST node for ("rocm", 9)
            >>> result = _parse_cst_tuple_key(cst_node)
            >>> print(result)
            ("rocm", 9)
            
            >>> # CST node for (None, None)
            >>> result = _parse_cst_tuple_key(cst_node)
            >>> print(result)
            (None, None)
        
        Note:
            Uses ONLY CST node inspection. No string parsing, no regex.
            Accesses CST nodes like SimpleString.value, Integer.value, Name.value.
        """
        if not isinstance(node, cst.Tuple):
            return None
        
        elements = node.elements
        if len(elements) != 2:
            return None
        
        # Parse first element (device string)
        first_elem = elements[0].value
        if isinstance(first_elem, cst.SimpleString):
            device_str = first_elem.value.strip('"\'')
        elif isinstance(first_elem, cst.Name) and first_elem.value == "None":
            device_str = None
        else:
            return None
        
        # Parse second element (version - can be tuple, int, or None)
        second_elem = elements[1].value
        
        if isinstance(second_elem, cst.Name) and second_elem.value == "None":
            version = None
        elif isinstance(second_elem, cst.Integer):
            version = int(second_elem.value)
        elif isinstance(second_elem, cst.Tuple):
            # Parse nested tuple like (8, 6)
            version_elements = second_elem.elements
            if len(version_elements) == 2:
                try:
                    major = int(version_elements[0].value.value)
                    minor = int(version_elements[1].value.value)
                    version = (major, minor)
                except:
                    return None
            else:
                return None
        else:
            return None
        
        return (device_str, version)
    
    def _device_matches(self, key_device: tuple, target_device: tuple) -> int:
        """
        Check if a dict key device matches the target device using fallback logic.
        
        Expectations dictionaries may not have exact device matches, so we use
        a priority-based fallback system to find the "closest" match.
        
        Matching Priority (lower number = better match):
            0. Exact match: key and target are identical
               Example: ("cuda", (8, 6)) matches ("cuda", (8, 6))
            
            1. Major version match: device matches, major version matches
               Example: ("cuda", 8) matches target ("cuda", (8, 6))
               Use case: Dict has ("cuda", 8) but we want ("cuda", (8, 6))
            
            2. Device-only match: device matches, version is None
               Example: ("cuda", None) matches target ("cuda", (8, 6))
               Use case: Dict has generic ("cuda", None) entry
            
            3. Wildcard match: (None, None) matches anything
               Use case: Dict has a fallback entry for all devices
            
            -1: No match at all
        
        Args:
            key_device: The device tuple from a dict key
                        Format: (device_str, version)
            target_device: The device we're trying to match
                          Format: (device_str, version)
        
        Returns:
            Priority integer (0-3 for match, -1 for no match)
            Lower number means better match
        
        Example Matching Process:
            Target: ("cuda", (8, 6))
            Dict keys: [("cuda", (9, 0)), ("cuda", 8), ("cuda", None)]
            
            Matches:
            - ("cuda", (9, 0)): -1 (no match, wrong version)
            - ("cuda", 8): 1 (major version match)
            - ("cuda", None): 2 (device-only match)
            
            Best match: ("cuda", 8) with priority 1
        
        Usage Pattern:
            ```python
            best_match = None
            best_priority = 999
            
            for key in dict_keys:
                priority = self._device_matches(key, target)
                if priority >= 0 and priority < best_priority:
                    best_match = key
                    best_priority = priority
            ```
        """
        key_str, key_ver = key_device
        target_str, target_ver = target_device
        
        # Exact match
        if key_str == target_str and key_ver == target_ver:
            return 0
        
        # Major version match: ("cuda", 8) matches ("cuda", (8, 6))
        if key_str == target_str and isinstance(target_ver, tuple) and key_ver == target_ver[0]:
            return 1
        
        # Device match: ("cuda", None) matches ("cuda", (8, 6))
        if key_str == target_str and key_ver is None:
            return 2
        
        # Wildcard match: (None, None) matches anything
        if key_str is None and key_ver is None:
            return 3
        
        return -1
    
    def _find_matching_dict_element(self, dict_node: cst.Dict) -> Optional[cst.DictElement]:
        """
        Find the DictElement in an Expectations dict that best matches target device.
        
        This is the KEY method for device-aware Expectations handling. It uses
        PURE CST to iterate through dictionary elements, parse their keys, and
        find the best matching one using the fallback logic.
        
        CST Structure:
            Dict(
                elements=[
                    DictElement(key=Tuple(...), value=List(...)),
                    DictElement(key=Tuple(...), value=List(...)),
                    ...
                ]
            )
        
        Process:
            1. Iterate through dict_node.elements
            2. For each DictElement:
               a. Extract key with _parse_cst_tuple_key()
               b. Check match with _device_matches()
               c. Track best match (lowest priority)
            3. Return DictElement with best matching key
        
        Args:
            dict_node: CST Dict node from Expectations({...}) argument
        
        Returns:
            DictElement object with best matching key, or None if:
            - dict_node is not a Dict
            - No keys match (all return priority -1)
            - All keys fail to parse
        
        Example:
            ```python
            # Target device: ("cuda", (8, 6))
            # Dict:
            # {
            #     ("xpu", 3): [value1],           # No match (-1)
            #     ("cuda", (9, 0)): [value2],     # No match (-1)
            #     ("cuda", 8): [value3],          # Match (priority 1)
            #     ("cuda", None): [value4],       # Match (priority 2)
            # }
            
            element = self._find_matching_dict_element(dict_node)
            # Returns: DictElement with key ("cuda", 8)
            # Because it has the best priority (1 < 2)
            
            # Then extract the value:
            value_node = element.value  # → CST node for [value3]
            ```
        
        Why This Works:
            By returning the entire DictElement (not just the key), we can
            immediately access element.value to get the exact CST node of
            the value, which we then use with PositionProvider to get its
            exact character position in the file.
        
        Note:
            This method uses NO string operations. Everything is CST-based:
            - dict_node.elements: CST API
            - _parse_cst_tuple_key(): CST node inspection
            - _device_matches(): Pure comparison logic
        """
        if not isinstance(dict_node, cst.Dict):
            return None
        
        best_match = None
        best_priority = 999
        
        for element in dict_node.elements:
            if isinstance(element, cst.DictElement):
                key_tuple = self._parse_cst_tuple_key(element.key)
                if key_tuple:
                    priority = self._device_matches(key_tuple, self.device)
                    if priority >= 0 and priority < best_priority:
                        best_match = element
                        best_priority = priority
        
        return best_match




def analyze_test_file(filepath: str, tasks: List[UpdateTask], device: tuple = ("cuda", (8, 6))) -> Dict[str, ReplacementPosition]:
    """
    Perform CST analysis on test file to find exact positions for all updates.
    
    This is PHASE 2 - the CST analysis phase. It performs a single-pass traversal
    of the test file to find where each target variable is defined and extract
    the exact character position of its value.
    
    Process:
        1. Read test file
        2. Parse with LibCST to build CST
        3. Wrap with MetadataWrapper to enable PositionProvider
        4. Create TestFileAnalyzer visitor with tasks and device
        5. Visit all nodes (TestFileAnalyzer does the work)
        6. Return results dictionary
    
    What TestFileAnalyzer Does:
        - Tracks method boundaries
        - Collects all assignments
        - Matches tasks to assignments (backward trace)
        - Handles derived values (.get_expectation())
        - For Expectations: Finds matching device key
        - Extracts exact value positions with PositionProvider
    
    Args:
        filepath: Path to test file (e.g., "tests/models/clip/test_modeling_clip.py")
        tasks: List of UpdateTask objects specifying what to update
               Each task has: variable_name, new_value, assertion_line, is_inline
        device: Device tuple for Expectations matching
                Format: (device_str, version)
                Example: ("cuda", (8, 6))
                Used to select which Expectations key to update
    
    Returns:
        Dictionary mapping variable_name → ReplacementPosition
        
        For each successfully found variable, provides:
        - Exact start/end line and column of VALUE to replace
        - Pattern type for formatting logic
        - Indent level for multi-line formatting
        
        Variables not found are simply not in the dictionary.
    
    Example:
        Tasks:
            [UpdateTask(variable_name="EXPECTED_OUTPUT", ...),
             UpdateTask(variable_name="expected_ids", ...)]
        
        Returns:
            {
                "EXPECTED_OUTPUT": ReplacementPosition(
                    start_line=100, start_col=25,
                    end_line=100, end_col=35,
                    pattern_type="list",
                    indent_level=12
                ),
                "expected_ids": ReplacementPosition(
                    start_line=150, start_col=20,
                    end_line=150, end_col=30,
                    pattern_type="list",
                    indent_level=8
                )
            }
    
    Performance:
        - Parse time: O(n) where n = file size
        - Visit time: O(m) where m = number of nodes
        - Typical: 50-100ms for a 50KB test file
    
    Note:
        Uses LibCST, NOT the built-in ast module.
        LibCST preserves all formatting and provides exact positions.
    """
    with open(filepath) as f:
        code = f.read()
    
    tree = cst.parse_module(code)
    wrapper = cst.metadata.MetadataWrapper(tree)
    
    analyzer = TestFileAnalyzer(tasks, device)
    wrapper.visit(analyzer)
    
    return analyzer.results


# ============================================================================
# PHASE 3: COMPUTE REPLACEMENTS (STRING OPERATIONS ONLY)
# ============================================================================


def compute_replacement_text(new_value: str, position: ReplacementPosition, is_multiline: bool) -> str:
    """
    Format the new value with proper indentation for insertion into the file.
    
    This is PHASE 3 of the pipeline. Takes the raw value from captured_info.txt
    and formats it to match the indentation and structure of the target location.
    
    Uses ONLY simple string operations:
    - .strip() to remove leading/trailing whitespace
    - .split('\n') to break into lines
    - ' ' * indent to create indentation
    - String concatenation to build result
    
    NO regex, NO bracket counting, NO complex parsing!
    
    Formatting Rules:
    
        Single-Line Values:
            Return as-is after stripping whitespace
            Example: "[1, 2, 3]" → "[1, 2, 3]"
        
        Multi-Line torch.tensor:
            Special formatting with inner indentation
            
            Input:
                [[1, 2],
                 [3, 4]]
            
            Output (indent_level=8):
                [[1, 2],
                        [3, 4]]  ← Inner lines indented by indent+4
                ]                 ← Closing bracket at indent level
        
        Multi-Line Other Patterns:
            Apply base indentation to all non-empty lines
            
            Input:
                [
                "line1",
                "line2"
                ]
            
            Output (indent_level=8):
                        [
                        "line1",
                        "line2"
                        ]
    
    Args:
        new_value: Raw value string from captured_info.txt
                   May contain multiple lines and variable whitespace
        position: ReplacementPosition with indent_level and pattern_type
        is_multiline: True if new_value contains newlines (unused, computed from new_value)
    
    Returns:
        Formatted text string ready to insert at the position
        Includes proper indentation and line breaks
    
    Examples:
        >>> pos = ReplacementPosition(..., pattern_type="list", indent_level=8)
        >>> compute_replacement_text("[1, 2, 3]", pos, False)
        "[1, 2, 3]"
        
        >>> pos = ReplacementPosition(..., pattern_type="torch.tensor", indent_level=8)
        >>> compute_replacement_text("[[1, 2],\n [3, 4]]", pos, True)
        "[[1, 2],\n            [3, 4]]\n        ]"
    
    Note:
        This function has NO KNOWLEDGE of the original file content.
        It only formats based on indent_level and pattern_type.
        The original file positions are handled in Phase 4 (apply_replacements).
    """
    # Strip the value
    new_value = new_value.strip()
    value_lines = new_value.split('\n')
    
    # Single-line replacement - just return the value
    if len(value_lines) == 1:
        return new_value
    
    # Multi-line replacement
    # Need to handle different combinations of structure and format
    
    if position.is_function_call:
        # Function call (torch.tensor, Expectations, etc.)
        # Format: function(\n    data\n)
        
        if position.is_data_inline:
            # Data was inline with opening (, needs leading newline
            # Example: tensor([[old]]) → tensor(\n    [[new]]\n)
            base_indent = ' ' * position.indent_level
            inner_indent = ' ' * (position.indent_level + 4)
            
            result_lines = []
            # Add leading newline + opening bracket
            result_lines.append('\n' + base_indent + value_lines[0])
            
            # Middle lines
            for line in value_lines[1:-1]:
                content = line.strip()
                result_lines.append(inner_indent + content)
            
            # Closing bracket  
            if len(value_lines) > 1:
                result_lines.append(base_indent + value_lines[-1].strip())
            
            return '\n'.join(result_lines)
        else:
            # Data already on separate line, no leading newline needed
            # Example: tensor(\n    [[old]]\n) → tensor(\n    [[new]]\n)
            base_indent = ' ' * position.indent_level
            inner_indent = ' ' * (position.indent_level + 4)
            
            result_lines = []
            # Opening bracket (no leading newline)
            result_lines.append(value_lines[0])
            
            # Middle lines
            for line in value_lines[1:-1]:
                content = line.strip()
                result_lines.append(inner_indent + content)
            
            # Closing bracket
            if len(value_lines) > 1:
                result_lines.append(base_indent + value_lines[-1].strip())
            
            return '\n'.join(result_lines)
    
    else:
        # Direct assignment (plain list, dict, tuple)
        # Opening bracket stays on assignment line
        # Example: var = [\n    items\n]
        
        content_indent = ' ' * position.indent_level
        
        result_lines = []
        # Opening bracket (no leading newline, stays on assignment line)
        result_lines.append(value_lines[0])
        
        # Content lines
        for line in value_lines[1:-1]:
            content = line.strip()
            if content:
                result_lines.append(content_indent + content)
        
        # Closing bracket
        if len(value_lines) > 1:
            # Closing bracket aligns with opening bracket
            # Use assignment_indent, not content_indent
            close_indent = ' ' * position.assignment_indent
            result_lines.append(close_indent + value_lines[-1].strip())
        
        return '\n'.join(result_lines)


# ============================================================================
# PHASE 4: APPLY REPLACEMENTS (IN-MEMORY, SINGLE WRITE)
# ============================================================================

def line_col_to_char_pos(text: str, line: int, col: int) -> int:
    """
    Convert CST (line, col) position to character offset in text string.
    
    LibCST's PositionProvider gives us positions as (line, column) coordinates.
    For the elegant replace_substrings() algorithm, we need character offsets.
    
    This function performs the conversion by:
    1. Counting characters in all lines before the target line
    2. Adding 1 for each newline character
    3. Adding the column offset within the target line
    
    Position Formats:
        CST: (line=1-indexed, col=0-indexed)
        Character offset: 0-indexed from start of file
    
    Args:
        text: The complete file content as a string
        line: Line number (1-indexed, as returned by CST)
        col: Column number (0-indexed, as returned by CST)
    
    Returns:
        Character position (0-indexed) in the text string
    
    Example:
        Text:
            "Line 1\n"     # 7 chars (6 + newline)
            "Line 2\n"     # 7 chars
            "Line 3"       # 6 chars
        
        Conversions:
        - (1, 0) → 0   (start of "Line 1")
        - (1, 5) → 5   (char '1' in "Line 1")
        - (2, 0) → 7   (start of "Line 2")
        - (2, 3) → 10  (char 'e' in "Line 2")
        - (3, 0) → 14  (start of "Line 3")
    
    Edge Cases:
        - Line beyond file end: Will count to end of text
        - Column beyond line end: May point past line end
        
    Note:
        Assumes Unix-style line endings (\n).
        Works correctly as long as file was read consistently.
    """
    lines = text.split('\n')
    char_pos = 0
    
    # Add lengths of all lines before target line
    for i in range(line - 1):
        if i < len(lines):
            char_pos += len(lines[i]) + 1  # +1 for newline
    
    # Add column offset
    char_pos += col
    
    return char_pos


def replace_substrings(s: str, replacements: List[Tuple[int, int, str]]) -> str:
    """
    Apply multiple replacements to a string in one pass (THE ELEGANT ALGORITHM).
    
    This is the CORE of Phase 4. It replaces multiple substrings in a single
    forward pass through the text, without any position tracking or adjustments.
    
    Algorithm:
        1. Start at position 0
        2. For each replacement (start, end, new_text):
           a. Copy everything from last_end to start (unchanged part)
           b. Insert new_text (the replacement)
           c. Update last_end = end (skip the old text)
        3. Copy everything from last_end to end of string
        4. Join all parts
    
    Why This Is Elegant:
        - Single forward pass (O(n) where n = string length)
        - No position tracking needed
        - No reverse processing needed
        - Works because replacements are SORTED BY START POSITION
        - Each replacement operates on ORIGINAL positions
    
    Visual Example:
        Original: "Hello world, this is a test"
                   ^     ^           ^    ^
                   0     6          17   23
        
        Replacements (sorted by start):
            (0, 5, "Hi")       - Replace "Hello"
            (6, 11, "Python")  - Replace "world"
            (23, 27, "demo")   - Replace "test"
        
        Process:
            result = []
            result.append(s[0:0])        = ""
            result.append("Hi")
            result.append(s[5:6])        = " "
            result.append("Python")
            result.append(s[11:23])      = ", this is a "
            result.append("demo")
            result.append(s[27:])        = ""
            
        Result: "Hi Python, this is a demo"
    
    Args:
        s: Original string (entire file content)
        replacements: List of (start, end, replacement_string) tuples
                      MUST be sorted by start position (ascending)
                      start: inclusive character position
                      end: exclusive character position
                      replacement_string: text to insert
    
    Returns:
        New string with all replacements applied
    
    Requirements:
        - Replacements MUST be sorted by start position
        - Replacements MUST NOT overlap (start[i+1] >= end[i])
        - Positions are character offsets (0-indexed)
    
    Performance:
        Time: O(n + m) where n = len(s), m = len(replacements)
        Space: O(n) for result string
    
    Note:
        This function is why we convert CST (line, col) to character positions.
        Working with character offsets makes the algorithm simple and efficient.
    """
    result = []
    last_end = 0
    
    for start, end, r in replacements:
        result.append(s[last_end:start])
        result.append(r)
        last_end = end
    
    result.append(s[last_end:])
    
    return ''.join(result)


def apply_replacements(filepath: str, replacements: List[Replacement], dry_run: bool = True) -> bool:
    """
    Apply all replacements to a file using the elegant character-based algorithm.
    
    This is PHASE 4 - the final step that actually modifies the file.
    
    Workflow:
        1. Read entire file as single string (one read)
        2. Convert all CST (line, col) positions to character offsets
        3. Sort replacements by character position (ascending)
        4. Apply all replacements in one pass with replace_substrings()
        5. Write entire file back (one write)
    
    Why This Approach:
        ✓ Simple: No position tracking or adjustments
        ✓ Efficient: Single read, single write, O(n) replacement
        ✓ Elegant: Clean, readable code
        ✓ Correct: All replacements use original positions
        
        vs. Old Approach:
        ✗ Complex: Line array manipulation, position adjustments
        ✗ Fragile: Reverse processing, newline handling
        ✗ Slow: Multiple list operations
    
    Character Position Conversion:
        CST gives: (line=10, col=20)
        We need: char_pos=523
        
        Conversion: Count all characters before line 10, plus 20
        (See line_col_to_char_pos for details)
    
    Args:
        filepath: Path to test file to modify
        replacements: List of Replacement objects, each containing:
                      - position: ReplacementPosition with (line, col) coords
                      - new_text: Formatted text to insert
        dry_run: If True, perform all operations but don't write file
                 Useful for previewing changes
    
    Returns:
        True if successful, False if any step fails
    
    Example:
        Before:
            File content (50 lines, 2000 chars)
            Line 10: "EXPECTED = [0, 0, 0]"
            Line 25: "OTHER = [1, 1, 1]"
        
        Replacements:
            [Replacement(pos @ line 10, "[2, 2, 2]"),
             Replacement(pos @ line 25, "[3, 3, 3]")]
        
        Process:
            1. Read: "..." (2000 chars)
            2. Convert: [(523, 532, "[2, 2, 2]"), (1245, 1254, "[3, 3, 3]")]
            3. Sort: Already sorted (523 < 1245)
            4. Replace: replace_substrings() → new content
            5. Write: new content to file
        
        After:
            Line 10: "EXPECTED = [2, 2, 2]"  ← Updated
            Line 25: "OTHER = [3, 3, 3]"     ← Updated
    
    Error Handling:
        - File read errors: Return False
        - Position conversion errors: May produce invalid positions
        - Write errors: File remains unchanged (if dry_run=False)
    
    Note:
        In dry_run mode, all operations except file write are performed.
        This ensures dry_run accurately reflects what --apply would do.
    """
    # Read file as single string
    with open(filepath) as f:
        original_text = f.read()
    
    # Convert positions to character positions and prepare replacement tuples
    char_replacements = []
    
    for replacement in replacements:
        pos = replacement.position
        
        # Convert start position
        start_char = line_col_to_char_pos(original_text, pos.start_line, pos.start_col)
        
        # Convert end position
        end_char = line_col_to_char_pos(original_text, pos.end_line, pos.end_col)
        
        char_replacements.append((start_char, end_char, replacement.new_text))
    
    # Sort by start position (ascending) - required by replace_substrings
    char_replacements.sort(key=lambda x: x[0])
    
    # Apply all replacements using the elegant function
    updated_text = replace_substrings(original_text, char_replacements)
    
    # Write file (if not dry-run)
    if not dry_run:
        with open(filepath, 'w') as f:
            f.write(updated_text)
    
    return True


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def process_captured_info(captured_file: str, apply: bool = False, device: tuple = ("cuda", (8, 6))):
    """
    Main entry point that orchestrates the entire 4-phase update process.
    
    This function implements the complete workflow:
        Phase 1: Parse captured_info.txt → Extract assertion failures
        Phase 2: CST analysis → Find exact value positions
        Phase 3: Compute replacements → Format new values
        Phase 4: Apply replacements → Update files
    
    Workflow Details:
    
        Phase 1: Parse captured_info.txt
            • Read and parse captured_info.txt
            • Extract all assertion failure blocks
            • For each block:
              - Determine target variable (EXPECT* prefix priority)
              - Extract new value from the other argument
              - Create UpdateTask
            • Skip cross-file assertions
            • Group tasks by test file
        
        Phase 2: CST Analysis (per file)
            • Parse test file once with LibCST
            • Use TestFileAnalyzer to find all positions
            • Handle backward tracing and derived values
            • For Expectations: Match device keys
            • Output: Dict[variable_name → ReplacementPosition]
        
        Phase 3: Compute Replacements (per task)
            • Format new value with proper indentation
            • Handle single-line vs multi-line
            • Pattern-specific formatting rules
            • Output: List[Replacement]
        
        Phase 4: Apply Replacements (per file)
            • Convert CST positions to character offsets
            • Sort by position
            • Apply all with replace_substrings()
            • Write file (if not dry_run)
    
    Args:
        captured_file: Path to captured_info.txt file
                       Example: "captured/gemma3/captured_info.txt"
        apply: Whether to apply changes
               False (default): Dry-run, preview changes
               True: Actually modify files
        device: Device tuple for Expectations matching
                Format: (device_str, version)
                Examples:
                - ("cuda", (8, 6)): CUDA compute capability 8.6
                - ("cuda", 9): CUDA major version 9
                - ("cuda", None): Any CUDA version
                - ("rocm", (9, 4)): ROCm 9.4
    
    Output (to stdout):
        Progress information for each phase:
        - [1] Parsing results: blocks found, tasks created, skipped blocks
        - [2] Processing per file:
          - [2a] CST analysis: positions found
          - [2b] Compute replacements: which variables
          - [2c] Apply results: how many updated (or dry-run message)
        - Summary: mode, files processed, total tasks
    
    Return Value:
        None (prints output to stdout)
    
    Example Output:
        ================================================================================
        CST-based Expectations Updater v2.0
        ================================================================================
        
        [1] Parsing captured_info.txt...
            Found 2 assertion failure blocks
            Created 1 update tasks
            Skipped 1 blocks
            Grouped into 1 file(s)
        
        [2] Processing files...
        
          File: tests/models/gemma3/test_modeling_gemma3.py
          Tasks: 1
          [2a] Analyzing with CST...
               Found positions for 1/1 tasks
          [2b] Computing replacements...
               ✓ EXPECTED_TEXTS at line 727
          [2c] Applying replacements...
               💡 DRY RUN - Would update 1 locations
        
        ================================================================================
        ✅ Complete!
        ================================================================================
        
        Mode: DRY RUN
        Files processed: 1
        Total tasks: 1
    
    Error Handling:
        • File not found: Print error, continue
        • Parse errors: Skip block, report
        • CST errors: Skip variable, report
        • No positions found: Report, continue
        • All errors are non-fatal: Process what's possible
    
    Best Practices:
        1. Always run without --apply first (dry-run)
        2. Review output before applying
        3. Use version control (git)
        4. Test on single model before batch processing
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
        positions = analyze_test_file(filepath, tasks, device)
        print(f"       Found positions for {len(positions)}/{len(tasks)} tasks")
        
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
                is_multi
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
