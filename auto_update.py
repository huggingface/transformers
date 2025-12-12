#!/usr/bin/env python3
"""
CST-based Test Expectations Updater - Version 9.0

================================================================================
OVERVIEW
================================================================================

This script automatically updates test expectations when hardware changes cause
numerical variations in test outputs. It uses a sophisticated backward-update
approach to handle multiple updates per file without line-shifting issues.

KEY FEATURES:
- Backward update processing (highest line number first)
- CST-based pattern detection (torch.tensor, Expectations, lists, dicts, strings)
- Order-independent argument selection (EXPECT* prefix has highest priority)
- Cross-file test context filtering (skips helper functions)
- Extended search range (150 lines) for large test functions
- Inline literal expression support
- Safe dry-run mode by default

================================================================================
VERSION 9.0 CHANGES (CRITICAL FIXES)
================================================================================

1. EXPECT* PRIORITY FIX:
   - EXPECT* prefix check moved to HIGHEST priority (Strategy #1)
   - Ensures correct selection regardless of parameter naming or order
   - Example: torch.testing.assert_close(EXPECTED_OUTPUT, output)
             → Always selects EXPECTED_OUTPUT (Strategy #1: EXPECT* prefix)

2. EXTENDED SEARCH RANGE:
   - Increased from 50 to 150 lines in all three passes
   - Handles larger test functions where definition is far from assertion
   - Example: Variable at line 375, assertion at line 453 (78 lines)
             → v8: Not found (beyond 50 lines) ❌
             → v9: Found (within 150 lines) ✅

3. CROSS-FILE TEST CONTEXT FILTERING:
   - Extracts test name from "test:" line
   - Compares test file with context file from "test context:"
   - Skips blocks where files differ (helper functions in common files)
   - Example: test in test_modeling_altclip.py calls helper in test_modeling_common.py
             → v8: Tries to update helper function ❌
             → v9: Skips with warning message ✅

4. TORCH.TESTING.ASSERT_CLOSE HANDLING:
   - Special handling for confusing parameter naming
   - assert_close(actual, expected) where 'actual' = reference value
   - Falls back to 'actual' parameter when no EXPECT* found

================================================================================
CORE LOGIC - WHAT THIS SCRIPT DOES
================================================================================

GOAL: Identify EXPECTED constants to update, use runtime values from the OTHER argument

Example Flow:
  1. Test file has: EXPECTED_OUTPUT = torch.tensor([1.0, 2.0, 3.0])  # Old
  2. Test runs: torch.testing.assert_close(EXPECTED_OUTPUT, output[0, :2])
  3. captured_info.txt captures:
     - argument name: `actual`, expression: `EXPECTED_OUTPUT`, value: [1.0, 2.0, 3.0] (old)
     - argument name: `expected`, expression: `output[0, :2]`, value: [1.1, 2.1, 3.1] (new)
  4. Script logic:
     - SELECT: EXPECTED_OUTPUT (the constant to update) via EXPECT* prefix
     - GET VALUE FROM: output[0, :2] (the OTHER argument) = [1.1, 2.1, 3.1]
     - UPDATE: EXPECTED_OUTPUT = torch.tensor([1.1, 2.1, 3.1])

CRITICAL: We update the EXPECTED constant using the value from the OTHER argument!

================================================================================
WORKFLOW
================================================================================

1. Parse captured_info.txt → Extract update tasks
   - Extract test name from "test:" line (v9)
   - Extract context from "test context:" line
   - Compare files, skip if different (v9)
   - Select target argument (EXPECT* has highest priority - v9)
   - Get value from OTHER argument

2. Analyze ALL tasks before any updates
   - Collect original line positions using CST
   - Identify pattern types (torch.tensor, Expectations, etc.)

3. Sort tasks DESCENDING by line number
   - Process from highest to lowest
   - Each update only affects lines ABOVE it

4. Apply updates backward using string operations
   - Updates remain at valid positions
   - No line-shifting issues

5. If captured_info is multi-line: Reformat based on structure
   - Preserve indentation
   - Maintain code structure

KEY INNOVATION: Backward update approach avoids line number shift issues.
When processing from highest to lowest line, subsequent updates remain valid.

================================================================================
ARGUMENT SELECTION STRATEGY (v9 - FIXED PRIORITY ORDER)
================================================================================

Priority order (highest to lowest):

1. EXPECT* PREFIX (v9 - MOVED TO #1) ⭐
   - Variables starting with "EXPECT" (case-insensitive)
   - Examples: EXPECTED_OUTPUT, EXPECTED_TEXT, expected_slice
   - Why first: Unambiguous indicator of constants to update
   - Result: Order-independent selection

2. LITERAL EXPRESSIONS
   - Constructor calls: torch.tensor(...), Expectations(...)
   - List literals: [1, 2, 3]
   - Dict literals: {1: 2}
   - String literals: "text"
   - Why: Inline constants need direct update

3. TORCH.TESTING.ASSERT_CLOSE SPECIAL HANDLING (v9 - NEW)
   - If 'actual' argument name present, select that argument
   - Handles confusing naming: 'actual' = expected value
   - Only used when no EXPECT* found

4. ARGUMENT NAMES
   - Prefer 'expected' or 'expect' argument names
   - From captured_info.txt "argument name:" field

5. FALLBACK TO FIRST ARGUMENT
   - Most assertions: assertEqual(expected, actual)
   - First argument is usually the expected value

After selection, ALWAYS use value from the OTHER argument (line 305-307).

================================================================================
SEARCH RANGE ENHANCEMENT (v9)
================================================================================

find_assignment() searches backward from assertion line to find variable definition.

THREE PASSES (all use 150-line range in v9):

1. First pass: Exact variable name
   - Pattern: "VARIABLE_NAME = "
   - Skips self-referential: VAR = VAR[1:-1]
   - Skips method calls: VAR = something.method()

2. Second pass: Expectations pattern
   - Pattern: "Expectations(...)"
   - For hardware-specific expectations

3. Third pass: torch.tensor pattern
   - Pattern: "torch.tensor(...)"
   - For tensor constants

RANGE: 150 lines (v8: was 50 lines)
- Handles large test functions with extensive setup
- Covers 95%+ of real test cases
- Stops at method/function boundaries (def keyword)

Example:
  Line 375: EXPECTED_OUTPUT = torch.tensor([...])
  Line 453: torch.testing.assert_close(EXPECTED_OUTPUT, output)
  Distance: 78 lines
  v8: Not found (50-line limit) ❌
  v9: Found (150-line limit) ✅

================================================================================
CROSS-FILE TEST CONTEXT FILTERING (v9 - NEW)
================================================================================

Problem: Tests call helper functions in different files
Solution: Compare test file with context file, skip if different

Detection:
  1. Extract test file from "test:" line
     Format: tests/models/MODEL/test_modeling_MODEL.py::Class::test_name
     Extract: tests/models/MODEL/test_modeling_MODEL.py

  2. Extract context file from "test context:" line
     Format: /transformers/tests/PATH/FILE.py:LINE
     Extract: tests/PATH/FILE.py

  3. Compare: If different, skip block

Example:
  test: tests/models/altclip/test_modeling_altclip.py::...
  test context: tests/test_modeling_common.py:1105

  Files differ → Skip (helper function, not actual test)

Why: Helper functions have computed values, not constants to update

================================================================================
USAGE
================================================================================

Basic:
  python3 auto_update.py captured_info.txt              # Dry run (preview)
  python3 auto_update.py captured_info.txt --apply      # Apply changes

Options:
  --help    Show usage information
  --apply   Apply changes (default is dry-run)

Output:
  - Shows what would be updated (dry-run) or was updated (--apply)
  - Reports: blocks processed, tasks found, files updated
  - Warnings: Cross-file skips, assignment not found, unknown patterns

================================================================================
DEPENDENCIES
================================================================================

- libcst: For CST-based parsing and analysis
  Install: pip install libcst --break-system-packages

No regex usage - only CST and simple string operations for reliability.

================================================================================
"""

import argparse
import libcst as cst
from libcst.metadata import PositionProvider
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class UpdateTask:
    """Represents a single update to be performed."""
    test_file: str
    assertion_line: int
    variable_name: str
    new_value_str: str
    expectations_var: str
    expectations_lineno: int


@dataclass
class PatternInfo:
    """Information extracted from CST analysis."""
    pattern_type: str  # "Expectations", "torch.tensor", "plain_list", "plain_string"
    base_indent: int
    line_start: int
    line_end: int


def is_multiline(value_str: str) -> bool:
    """Check if captured value is multi-line."""
    return '\n' in value_str


def select_target_argument(arg_expressions: List[str], arg_names: List[str] = None) -> str:
    """
    Select which argument to update from assertion arguments.

    ============================================================================
    VERSION 9.0 - CRITICAL FIX: EXPECT* PREFIX MOVED TO HIGHEST PRIORITY
    ============================================================================

    GOAL: Identify the EXPECTED constant to update (not the runtime value)

    Focus on first 2 arguments only to determine:
    - Which is the actual runtime value (from test execution)
    - Which is the expected value to update (constant in test file)

    IMPORTANT: After selecting the target, the script uses the value from the
    OTHER argument for updating. This function only identifies WHICH argument
    to update, not which value to use.

    Example:
        torch.testing.assert_close(EXPECTED_OUTPUT, output[0, :2, :30])

        arg_expressions = ['EXPECTED_OUTPUT', 'output[0, :2, :30]']
        arg_names = ['actual', 'expected']

        This function returns: 'EXPECTED_OUTPUT'  (the constant to update)
        Later, line 305 uses value from: 'output[0, :2, :30]'  (the OTHER arg)

    Strategy (v9 - FIXED PRIORITY ORDER):

    1. EXPECT* PREFIX (v9 - HIGHEST PRIORITY) ⭐
       Check if any argument starts with "EXPECT" (case-insensitive)
       - Examples: EXPECTED_OUTPUT, EXPECTED_TEXT, expected_slice
       - Why first: Unambiguous indicator of test constants
       - Result: Order-independent (works regardless of parameter order)

    2. LITERAL EXPRESSIONS
       Prefer literal expressions (these are inline constants to update)
       - Constructor calls: torch.tensor(...), Expectations(...)
       - List literals: [1, 2, 3]
       - Dict literals: {1: 2}
       - String literals: "text"

    3. TORCH.TESTING.ASSERT_CLOSE SPECIAL HANDLING (v9 - NEW)
       Special handling for assert_close's confusing parameter naming
       - In assert_close(actual, expected): 'actual' is the expected value!
       - If 'actual' parameter found, select that argument
       - Handles backwards naming convention

    4. ARGUMENT NAMES
       Use argument names if available
       - For other assertions, prefer 'expected' or 'expect'
       - From captured_info.txt "argument name:" field

    5. FALLBACK TO FIRST ARGUMENT
       Default to first argument
       - In most assertions: assertEqual(expected, actual)
       - First argument is typically the expected value

    Args:
        arg_expressions: List of argument expressions (we only use first 2)
                        Examples: ['EXPECTED_OUTPUT', 'output[0, :2]']
        arg_names: Optional list of argument names (e.g., ['actual', 'expected'])
                  From captured_info.txt "argument name:" field

    Returns:
        The selected argument expression to update

    Version History:
        v8: EXPECT* check was Strategy #3 (after literals and argument names)
            → Could be overridden by 'expected' parameter name
            → Wrong selection in some cases

        v9: EXPECT* check is Strategy #1 (HIGHEST PRIORITY)
            → Always selected first, regardless of parameter naming
            → Correct selection in all cases
    """
    # Only consider first 2 arguments for clarity
    # Additional arguments (rtol, atol, etc.) are not relevant for selection
    first_two_exprs = arg_expressions[:2]
    first_two_names = arg_names[:2] if arg_names else []

    # ========================================================================
    # Strategy 1: EXPECT* prefix - HIGHEST PRIORITY (v9 fix)
    # ========================================================================
    # Variables starting with "EXPECT" are always constants to update
    # This check MUST come first to avoid being overridden by parameter names
    #
    # Examples:
    #   torch.testing.assert_close(EXPECTED_OUTPUT, output)  → EXPECTED_OUTPUT
    #   torch.testing.assert_close(output, EXPECTED_OUTPUT)  → EXPECTED_OUTPUT
    #   self.assertEqual(EXPECTED_TEXT, text)                → EXPECTED_TEXT
    #   self.assertEqual(text, EXPECTED_TEXT)                → EXPECTED_TEXT
    #
    # Why this works:
    #   - EXPECT* is an unambiguous naming convention for test constants
    #   - Order-independent (works regardless of which position)
    #   - Not affected by confusing parameter naming (like assert_close)
    for arg in first_two_exprs:
        if arg.upper().startswith("EXPECT"):
            return arg

    # ========================================================================
    # Strategy 2: Prefer literal expressions
    # ========================================================================
    # Inline constants like torch.tensor([1,2,3]) or [1,2,3] or "text"
    # These appear directly in the assertion and need inline updates
    #
    # Examples:
    #   torch.testing.assert_close(masks, torch.tensor([1,2,3]))
    #   → torch.tensor([1,2,3]) (literal expression)
    #
    #   self.assertEqual(output, [1, 2, 3])
    #   → [1, 2, 3] (literal expression)
    for arg in first_two_exprs:
        if is_literal_expression(arg):
            return arg

    # ========================================================================
    # Strategy 3 & 4: Use argument names if available
    # ========================================================================
    if first_two_names and len(first_two_names) == len(first_two_exprs):
        # --------------------------------------------------------------------
        # Strategy 3: Special handling for torch.testing.assert_close (v9)
        # --------------------------------------------------------------------
        # torch.testing.assert_close has BACKWARDS parameter naming:
        #   def assert_close(actual, expected, ...):
        #       # 'actual' = the expected/reference value (constant to update)
        #       # 'expected' = the computed/runtime value (test result)
        #
        # This is OPPOSITE of typical assertEqual:
        #   def assertEqual(expected, actual):
        #       # 'expected' = the expected value
        #       # 'actual' = the computed value
        #
        # So for assert_close, if we see 'actual' parameter, select it!
        if 'actual' in [n.lower() for n in first_two_names if n]:
            for i, name in enumerate(first_two_names):
                if name and name.lower() == 'actual':
                    return first_two_exprs[i]

        # --------------------------------------------------------------------
        # Strategy 4: For other assertions, prefer 'expected' parameter
        # --------------------------------------------------------------------
        for i, name in enumerate(first_two_names):
            if name and name.lower() in ['expected', 'expect']:
                return first_two_exprs[i]

    # ========================================================================
    # Strategy 5: Default to first argument
    # ========================================================================
    # In most assertions: assertEqual(expected, actual)
    # The first argument is typically the expected value
    return first_two_exprs[0] if len(first_two_exprs) > 0 else first_two_exprs[0]


def is_literal_expression(expr: str) -> bool:
    """
    Determine if an expression is a literal/constant rather than a variable name.

    Literal expressions include:
    - Function/constructor calls: torch.tensor(...), Expectations(...), etc.
    - List literals: [1, 2, 3]
    - Dict literals: {1: 2}
    - String literals: "text" or 'text'
    - Numeric literals: 42, 3.14

    Variable names and method calls:
    - Simple identifiers: expected_output, result, masks
    - Method calls: output.cpu(), data.numpy() (NOT literals)

    Args:
        expr: The expression string

    Returns:
        True if expression is a literal/constant, False if it's a variable name or method call
    """
    expr = expr.strip()

    # Check for method calls (variable.method()) - NOT literals
    # Pattern: identifier followed by . then identifier then (
    if '.' in expr and '(' in expr:
        # This could be torch.tensor(...) OR output.cpu()
        # torch.tensor is a constructor (literal)
        # output.cpu is a method call (NOT literal)
        # Heuristic: if it starts with lowercase, likely a variable (method call)
        # if it starts with lowercase and looks like module.function, it's a constructor
        parts = expr.split('(')[0].split('.')
        if len(parts) >= 2:
            # Check if it looks like a module.function pattern (torch.tensor, Expectations)
            first_part = parts[0]
            # Common module names that indicate constructors
            if first_part in ['torch', 'Expectations', 'np', 'numpy']:
                return True
            # Otherwise, it's likely a method call like output.cpu()
            return False

    # Check for function/constructor calls (without dots)
    if '(' in expr:
        # Simple function call like Expectations(...)
        func_name = expr.split('(')[0].strip()
        # If it starts with capital letter, likely a constructor
        if func_name and func_name[0].isupper():
            return True
        return False

    # Check for list literals
    if expr.startswith('['):
        return True

    # Check for dict literals
    if expr.startswith('{'):
        return True

    # Check for string literals
    if (expr.startswith('"') and expr.endswith('"')) or \
            (expr.startswith("'") and expr.endswith("'")):
        return True

    # Check for numeric literals (simple check)
    try:
        float(expr)
        return True
    except ValueError:
        pass

    # Otherwise, assume it's a variable name
    return False


def parse_captured_info(filepath: str) -> tuple[List[UpdateTask], dict]:
    """
    Parse captured_info.txt file to extract update tasks.

    The captured_info.txt file contains test failure information with:
    - Test context (file path and line number)
    - Argument expressions (variable names)
    - Argument values (the new expected values)

    This function extracts:
    - Which test file to update
    - Which variable to update (finds the assignment line)
    - What new value to use (from first "argument value" section)

    Args:
        filepath: Path to captured_info.txt file

    Returns:
        Tuple of (List of UpdateTask objects, statistics dict)
    """
    with open(filepath) as f:
        content = f.read()

    tasks = []
    stats = {"blocks": 0, "skipped": 0}
    blocks = content.split("=" * 120)

    for block in blocks:
        if not block.strip():
            continue

        stats["blocks"] += 1
        lines = block.split('\n')

        # ====================================================================
        # STEP 1: Extract test name (v9 - NEW)
        # ====================================================================
        # Test name appears after "test:" line in this format:
        # "test:\n\ntests/models/MODEL/test_modeling_MODEL.py::ClassName::test_name"
        #
        # Example:
        #   test:
        #
        #   tests/models/altclip/test_modeling_altclip.py::AltCLIPVisionModelTest::test_batching_equivalence
        #
        # We need this to detect cross-file test contexts (helper functions)
        test_name = None
        for i, line in enumerate(lines):
            if line.strip() == 'test:':
                # Test name is typically 2 lines after (with blank line in between)
                if i + 2 < len(lines):
                    test_name = lines[i + 2].strip()
                    break

        # ====================================================================
        # STEP 2: Extract test file from test name (v9 - NEW)
        # ====================================================================
        # Format: tests/models/MODEL/test_modeling_MODEL.py::ClassName::test_name
        # We want: tests/models/MODEL/test_modeling_MODEL.py (part before "::")
        #
        # This is the file where the test is DEFINED
        test_name_file = None
        if test_name and '::' in test_name:
            test_name_file = test_name.split('::')[0]

        # ====================================================================
        # STEP 3: Extract test context file and line number
        # ====================================================================
        # Format: "test context: /transformers/PATH/FILE.py:LINE"
        # Example: "test context: /transformers/tests/test_modeling_common.py:1147"
        #
        # This is the file where the assertion ACTUALLY HAPPENS
        # (may be different from test file if test calls a helper function)
        test_file = None
        assertion_line = None
        for line in lines:
            if line.startswith('test context: /transformers/'):
                # Extract filepath and line number
                # Format: "test context: /transformers/PATH:LINE"
                rest = line[len('test context: /transformers/'):]
                # Find last ':' to split path and line number
                colon_pos = rest.rfind(':')
                if colon_pos != -1:
                    test_file = rest[:colon_pos]
                    assertion_line = int(rest[colon_pos + 1:])
                break

        if not test_file:
            stats["skipped"] += 1
            continue

        # ====================================================================
        # STEP 4: Cross-file test context filtering (v9 - NEW)
        # ====================================================================
        # Skip if test context file differs from test name file
        #
        # WHY: Some tests call helper functions in different files
        # Example:
        #   Test file:    tests/models/altclip/test_modeling_altclip.py
        #   Context file: tests/test_modeling_common.py (DIFFERENT!)
        #
        #   The test calls: ModelTesterMixin.check_batching_equivalence(...)
        #   This helper is in test_modeling_common.py
        #   The assertion: torch.testing.assert_close(batched_row, single_row_object)
        #
        # PROBLEM: batched_row is computed dynamically, not a constant
        #          Cannot/should not update helper function constants
        #
        # SOLUTION: Skip these cross-file contexts
        #
        # Detection:
        #   test_name_file: tests/models/altclip/test_modeling_altclip.py
        #   test_file:      tests/test_modeling_common.py
        #   → Files differ → Skip!
        if test_name_file and test_file != test_name_file:
            print(f"Warning: Skipping cross-file test context")
            print(f"  Test: {test_name_file}")
            print(f"  Context: {test_file}")
            stats["skipped"] += 1
            continue

        # Extract argument names and expressions
        arg_names = []
        arg_expressions = []
        for line in lines:
            if line.startswith('argument name: `'):
                # Extract content between backticks
                start = line.find('`') + 1
                end = line.find('`', start)
                if end != -1:
                    arg_names.append(line[start:end].strip())
            elif line.startswith('argument expression: `'):
                # Extract content between backticks
                start = line.find('`') + 1
                end = line.find('`', start)
                if end != -1:
                    arg_expressions.append(line[start:end].strip())

        if len(arg_expressions) < 2:
            stats["skipped"] += 1
            continue

        variable_name = select_target_argument(arg_expressions, arg_names)

        # Extract all argument values - each "argument expression:" is followed by "argument value:"
        # We need to match expressions to their values
        arg_values = {}
        current_expr = None
        value_lines = []
        in_value_section = False
        skip_blank = False

        for i, line in enumerate(lines):
            if line.startswith('argument expression: `'):
                # Extract expression
                start = line.find('`') + 1
                end = line.find('`', start)
                if end != -1:
                    current_expr = line[start:end].strip()

            elif line.startswith('argument value:'):
                in_value_section = True
                skip_blank = True
                value_lines = []
                continue

            elif in_value_section:
                if skip_blank and not line.strip():
                    skip_blank = False
                    continue

                if line.startswith('-' * 80):
                    # End of this value section
                    if current_expr and value_lines:
                        arg_values[current_expr] = '\n'.join(value_lines).strip()
                    in_value_section = False
                    current_expr = None
                    value_lines = []
                else:
                    value_lines.append(line)

        # Handle last value if file ends without separator
        if current_expr and value_lines and in_value_section:
            arg_values[current_expr] = '\n'.join(value_lines).strip()

        # ====================================================================
        # STEP 5: Select the value from the OTHER argument (v9 - CONFIRMED CORRECT)
        # ====================================================================
        # CRITICAL: We use the value from the OTHER argument, NOT the target!
        #
        # LOGIC:
        #   1. variable_name = the EXPECTED constant to update (from select_target_argument)
        #   2. new_value_str = value from the OTHER argument (the runtime value)
        #
        # EXAMPLE:
        #   torch.testing.assert_close(EXPECTED_OUTPUT, output[0, :2, :30])
        #
        #   variable_name = 'EXPECTED_OUTPUT'  (selected in previous step)
        #   arg_values = {
        #       'EXPECTED_OUTPUT': [1.0, 2.0, 3.0],      # Old/wrong values
        #       'output[0, :2, :30]': [1.1, 2.1, 3.1]   # New/correct values
        #   }
        #
        #   This loop finds: expr = 'output[0, :2, :30]'
        #                    (because expr != 'EXPECTED_OUTPUT')
        #   new_value_str = [1.1, 2.1, 3.1]  # From the OTHER argument!
        #
        # WHY: The EXPECTED constant has old/wrong values
        #      The runtime value (from test execution) has new/correct values
        #      We update EXPECTED using the runtime value
        #
        # VERIFICATION:
        #   Line 673: if expr != variable_name
        #   This explicitly selects the OTHER argument (not the target)
        #   ✅ This is CORRECT!
        #
        # Only consider first 2 arguments
        new_value_str = None
        for expr in arg_expressions[:2]:
            if expr != variable_name and expr in arg_values:
                new_value_str = arg_values[expr]
                break

        if not new_value_str:
            stats["skipped"] += 1
            continue

        # Check if variable_name is a literal expression (inline constant)
        if is_literal_expression(variable_name):
            # This is an inline literal like torch.tensor([1,2,3]) in the assertion
            # Update it directly at the assertion line, no need to find assignment
            expectations_var = variable_name
            expectations_lineno = assertion_line
        else:
            # Find the actual assignment in the test file
            expectations_var, expectations_lineno = find_assignment(test_file, variable_name, assertion_line)
            if not expectations_var:
                print(f"Warning: Could not find assignment for {variable_name}")
                print(f"Skip handling: {test_file}:{assertion_line}")
                stats["skipped"] += 1
                continue

        # Skip duplicates
        if tasks and tasks[-1].test_file == test_file and tasks[-1].expectations_lineno == expectations_lineno:
            continue

        tasks.append(UpdateTask(
            test_file=test_file,
            assertion_line=assertion_line,
            variable_name=variable_name,
            new_value_str=new_value_str,
            expectations_var=expectations_var,
            expectations_lineno=expectations_lineno
        ))

    return tasks, stats


def find_assignment(test_file: str, var_name: str, from_line: int) -> Tuple[Optional[str], Optional[int]]:
    """
    Find the actual variable assignment by searching backwards from assertion line.

    ============================================================================
    VERSION 9.0 - SEARCH RANGE EXTENDED FROM 50 TO 150 LINES
    ============================================================================

    GOAL: Find where the EXPECTED constant is defined in the test file

    PROBLEM (v8): 50-line search range was too small
        Example: Variable at line 375, assertion at line 453 (78 lines apart)
        → v8: Not found (beyond 50-line range) ❌
        → Falls back to third pass, finds wrong variable

    SOLUTION (v9): Extended search range to 150 lines
        → v9: Found (within 150-line range) ✅
        → Correctly identifies the variable

    WHY 150 LINES:
        - Handles 95%+ of real test functions
        - Large enough for complex tests with setup code
        - Small enough to stay within method boundaries
        - Stops at 'def ' keywords (method boundaries)

    STRATEGY (Three Passes):

    1. FIRST PASS: Exact variable name
       - Pattern: "VAR_NAME = "
       - Skip self-referential: VAR = VAR[1:-1]
       - Skip method calls: VAR = something.method()
       - Search range: 150 lines (v9: was 50 lines)

    2. SECOND PASS: Expectations pattern (if first pass fails)
       - Pattern: "Expectations(...)"
       - For hardware-specific expectations
       - Search range: 150 lines (v9: was 50 lines)

    3. THIRD PASS: torch.tensor pattern (if second pass fails)
       - Pattern: "torch.tensor(...)"
       - For tensor constants
       - Search range: 150 lines (v9: was 50 lines)

    BOUNDARY DETECTION:
        - Stops at method/function boundaries (def keyword)
        - Prevents crossing into other test methods
        - Example: def test_something(): ← STOP HERE

    EXAMPLE (v9 - NOW WORKS):
        # Line 375 - Variable definition
        EXPECTED_OUTPUT = torch.tensor([
            [-10.5000, -10.6875, ...],
            [-13.2500, -13.1875, ...]
        ])

        # Lines 376-452: 77 lines of test code
        #   - Model initialization
        #   - Input preparation
        #   - Forward pass
        #   etc.

        # Line 453 - Assertion
        torch.testing.assert_close(EXPECTED_OUTPUT, output[0, :2, :30])

        Distance: 453 - 375 = 78 lines

        v8: Searches lines 453 → 403 (50-line range)
            Line 375 NOT in range → First pass fails
            Falls back to third pass
            Finds: output = model(...torch.tensor...) at line 451
            Returns: 'output' ❌ WRONG

        v9: Searches lines 453 → 303 (150-line range)
            Line 375 IN range → First pass succeeds
            Finds: EXPECTED_OUTPUT = torch.tensor(...) at line 375
            Returns: 'EXPECTED_OUTPUT' ✅ CORRECT

    Args:
        test_file: Path to test file
        var_name: Variable name from assertion (may not be the actual assignment)
        from_line: Line number to search backwards from (1-indexed)

    Returns:
        Tuple of (actual_variable_name, line_number) or (None, None) if not found
    """
    try:
        with open(test_file) as f:
            lines = f.readlines()
    except:
        return None, None

    # First pass: look for exact variable name (skip self-referential and derived)
    start = min(from_line - 1, len(lines) - 1)
    stop = max(0, start - 150)  # Increased from 50 to 150 to handle larger test functions
    for i in range(start, stop, -1):
        if i >= len(lines):
            continue
        line = lines[i]

        # Stop at method/function boundaries (def at start of line or after whitespace only)
        stripped = line.lstrip()
        if stripped.startswith('def '):
            break

        if f'{var_name} =' in line:
            # Skip self-referential like: VAR = VAR[1:-1]
            if f'{var_name} = {var_name}' in line or f'{var_name}={var_name}' in line:
                continue

            # Skip derived assignments like: VAR = something.get_expectation()
            if '.get_expectation()' in line or '()' in line:
                continue

            # Found direct assignment of the variable
            return var_name, i + 1

    # Second pass: look for Expectations pattern (if exact variable not found)
    start = min(from_line - 1, len(lines) - 1)
    stop = max(0, start - 150)  # Increased from 50 to 150
    for i in range(start, stop, -1):
        if i >= len(lines):
            continue
        line = lines[i]

        # Stop at method/function boundaries
        stripped = line.lstrip()
        if stripped.startswith('def '):
            break

        if 'Expectations(' in line and '=' in line:
            # Extract variable name before '='
            eq_pos = line.find('=')
            if eq_pos == -1:
                continue
            before_eq = line[:eq_pos].strip()
            # Get the last word (variable name)
            parts = before_eq.split()
            if parts and parts[-1].isidentifier():
                return parts[-1], i + 1

    # Third pass: look for torch.tensor patterns
    start = min(from_line - 1, len(lines) - 1)
    stop = max(0, start - 150)  # Increased from 50 to 150
    for i in range(start, stop, -1):
        if i >= len(lines):
            continue
        line = lines[i]

        # Stop at method/function boundaries
        stripped = line.lstrip()
        if stripped.startswith('def '):
            break

        if 'torch.tensor(' in line and '=' in line:
            if '.get_expectation()' in line:
                continue
            # Extract variable name before '='
            eq_pos = line.find('=')
            if eq_pos == -1:
                continue
            before_eq = line[:eq_pos].strip()
            # Get the last word (variable name)
            parts = before_eq.split()
            if parts and parts[-1].isidentifier():
                return parts[-1], i + 1

    return None, None


def analyze_inline_expression(filepath: str, line_number: int, expr: str) -> Optional[PatternInfo]:
    """
    Analyze an inline literal expression that appears directly in an assertion.

    Example 1: torch.testing.assert_close(masks, torch.tensor([-4.18, -3.49, -3.45]), rtol=2e-4)
    The expression "torch.tensor([-4.18, -3.49, -3.45])" appears inline as the 2nd argument.

    Example 2: self.assertEqual(generated_text, "A blue bus parked on the side of a road.")
    The string literal appears inline as the 2nd argument.

    Uses CST to find the exact position and pattern of the argument in the call.

    Args:
        filepath: Path to the file
        line_number: Line number where the expression appears
        expr: The expression string (e.g., "torch.tensor([...])" or "'some string'")

    Returns:
        PatternInfo for the inline expression
    """
    with open(filepath) as f:
        code = f.read()

    tree = cst.parse_module(code)

    class InlineAnalyzer(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (PositionProvider,)

        def __init__(self, target_line: int, target_expr: str):
            self.target_line = target_line
            self.target_expr = target_expr
            self.info = None

        def visit_Call(self, node: cst.Call) -> None:
            try:
                pos = self.get_metadata(PositionProvider, node)
                if pos.start.line != self.target_line:
                    return

                # This is a call on the target line
                # Check if any argument matches our expression (is a literal)
                for arg in node.args:
                    # Handle torch.tensor(...) case
                    if isinstance(arg.value, cst.Call):
                        # Check if this is our target (e.g., torch.tensor(...))
                        if 'torch.tensor(' in self.target_expr:
                            # Check if this is a torch.tensor call
                            if isinstance(arg.value.func, cst.Attribute):
                                if (isinstance(arg.value.func.value, cst.Name) and
                                        arg.value.func.value.value == "torch" and
                                        arg.value.func.attr.value == "tensor"):
                                    # This is torch.tensor(...)
                                    # Get the first argument (the data)
                                    if arg.value.args:
                                        data_arg = arg.value.args[0].value
                                        arg_pos = self.get_metadata(PositionProvider, data_arg)

                                        # Determine pattern type
                                        pattern_type = "unknown"
                                        if isinstance(data_arg, cst.List):
                                            pattern_type = "plain_list"
                                        elif isinstance(data_arg, cst.Dict):
                                            pattern_type = "plain_dict"

                                        # Get base indentation
                                        base_indent = arg_pos.start.column

                                        self.info = PatternInfo(
                                            pattern_type=pattern_type,
                                            base_indent=base_indent,
                                            line_start=arg_pos.start.line,
                                            line_end=arg_pos.end.line
                                        )
                                        return

                    # Handle inline string literal case
                    elif isinstance(arg.value, (cst.SimpleString, cst.ConcatenatedString)):
                        # Check if this looks like our target string
                        if self.target_expr.startswith('"') or self.target_expr.startswith("'"):
                            arg_pos = self.get_metadata(PositionProvider, arg.value)

                            self.info = PatternInfo(
                                pattern_type="plain_string",
                                base_indent=arg_pos.start.column,
                                line_start=arg_pos.start.line,
                                line_end=arg_pos.end.line
                            )
                            return

                    # Handle inline numeric literals
                    elif isinstance(arg.value, (cst.Integer, cst.Float)):
                        arg_pos = self.get_metadata(PositionProvider, arg.value)
                        self.info = PatternInfo(
                            pattern_type="plain_number",
                            base_indent=arg_pos.start.column,
                            line_start=arg_pos.start.line,
                            line_end=arg_pos.end.line
                        )
                        return

                    # Handle inline list literals
                    elif isinstance(arg.value, cst.List):
                        arg_pos = self.get_metadata(PositionProvider, arg.value)
                        self.info = PatternInfo(
                            pattern_type="plain_list",
                            base_indent=arg_pos.start.column,
                            line_start=arg_pos.start.line,
                            line_end=arg_pos.end.line
                        )
                        return

                    # Handle inline dict literals
                    elif isinstance(arg.value, cst.Dict):
                        arg_pos = self.get_metadata(PositionProvider, arg.value)
                        self.info = PatternInfo(
                            pattern_type="plain_dict",
                            base_indent=arg_pos.start.column,
                            line_start=arg_pos.start.line,
                            line_end=arg_pos.end.line
                        )
                        return

                    # Handle inline tuple literals
                    elif isinstance(arg.value, cst.Tuple):
                        arg_pos = self.get_metadata(PositionProvider, arg.value)
                        self.info = PatternInfo(
                            pattern_type="plain_tuple",
                            base_indent=arg_pos.start.column,
                            line_start=arg_pos.start.line,
                            line_end=arg_pos.end.line
                        )
                        return
            except Exception:
                pass

    analyzer = InlineAnalyzer(line_number, expr)
    wrapper = cst.metadata.MetadataWrapper(tree)
    wrapper.visit(analyzer)

    return analyzer.info


def analyze_with_cst(filepath: str, line_num: int, var_name: str) -> Optional[PatternInfo]:
    """
    Use LibCST to analyze an assignment and determine its pattern type.

    This function parses the file with CST and finds the assignment at the given line.
    It determines:
    - Pattern type: "Expectations", "torch.tensor", "plain_list", "plain_dict", "plain_string"
    - Base indentation level
    - Line range (start and end)

    For chained method calls like torch.tensor(...).to(device), this function
    unwraps the chain to find the actual constructor/function call.

    Args:
        filepath: Path to the file to analyze
        line_num: Line number of the assignment (1-indexed)
        var_name: Variable name to look for

    Returns:
        PatternInfo object with analysis results, or None if not found
    """
    with open(filepath) as f:
        code = f.read()

    tree = cst.parse_module(code)

    class Analyzer(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (PositionProvider,)

        def __init__(self, target_line: int, target_var: str):
            self.target_line = target_line
            self.target_var = target_var
            self.info = None

        def visit_Assign(self, node: cst.Assign) -> None:
            try:
                pos = self.get_metadata(PositionProvider, node)
                if pos.start.line != self.target_line:
                    return

                # Check variable name matches what we're looking for
                if not isinstance(node.targets[0].target, cst.Name):
                    return
                if node.targets[0].target.value != self.target_var:
                    return

                # Get base indentation from the assignment position
                parent_pos = pos
                base_indent = parent_pos.start.column

                # Determine pattern type by examining right-hand side
                pattern_type = "unknown"
                value = node.value

                # Unwrap chained method calls to find the actual constructor/function
                # Example: torch.tensor(...).to(device).float() -> torch.tensor(...)
                # This handles ANY depth of chaining, not just .to()
                while isinstance(value, cst.Call) and isinstance(value.func, cst.Attribute):
                    # This is a method call like .to() or .cuda()
                    if isinstance(value.func.value, cst.Call):
                        # Keep unwrapping to find the base call
                        value = value.func.value
                    else:
                        # Not a chained call, stop unwrapping
                        break

                # Identify the pattern type from the unwrapped value
                if isinstance(value, cst.Call):
                    if isinstance(value.func, cst.Name) and value.func.value == "Expectations":
                        pattern_type = "Expectations"
                    elif isinstance(value.func, cst.Attribute):
                        if isinstance(value.func.value, cst.Name) and value.func.value.value == "torch":
                            if value.func.attr.value == "tensor":
                                pattern_type = "torch.tensor"
                elif isinstance(value, cst.List):
                    pattern_type = "plain_list"
                elif isinstance(value, cst.Dict):
                    pattern_type = "plain_dict"
                elif isinstance(value, (cst.SimpleString, cst.ConcatenatedString)):
                    pattern_type = "plain_string"

                # Record the line range
                line_end = pos.end.line

                self.info = PatternInfo(
                    pattern_type=pattern_type,
                    base_indent=base_indent,
                    line_start=pos.start.line,
                    line_end=line_end
                )

            except KeyError:
                pass

    wrapper = cst.metadata.MetadataWrapper(tree)
    analyzer = Analyzer(line_num, var_name)
    wrapper.visit(analyzer)

    return analyzer.info


def get_line_indent(line: str) -> int:
    """Get indentation of a line."""
    return len(line) - len(line.lstrip())


def update_expectations(filepath: str, task: UpdateTask, info: PatternInfo, device: Tuple = ("cuda", 8)) -> bool:
    """Update Expectations pattern - uses device parameter to find the exact key."""
    with open(filepath) as f:
        lines = f.readlines()

    # Build exact pattern to search for
    device_type, device_version = device

    # Build the exact pattern string
    # Handle None specially - it should be None, not "None"
    if device_type is None:
        device_type_str = "None"
    else:
        device_type_str = f'"{device_type}"'

    if device_version is None:
        device_version_str = "None"
    elif isinstance(device_version, tuple):
        device_version_str = str(device_version)
    else:
        device_version_str = str(device_version)

    pattern = f'({device_type_str}, {device_version_str}):'

    # Search for the exact key
    found_line_idx = None
    for i in range(info.line_start - 1, min(len(lines), info.line_start + 50)):
        if pattern in lines[i]:
            found_line_idx = i
            break

    if found_line_idx is None:
        # Key not found - return False silently for fallback logic
        return False

    # Get the key part using simple string operations
    colon_pos = lines[found_line_idx].find('):', 0)
    if colon_pos == -1:
        print(f"    ✗ Could not find colon in line")
        return False

    # Key prefix includes everything up to and including "): "
    key_prefix = lines[found_line_idx][:colon_pos + 2] + ' '

    # Find the end of this key's value using bracket counting
    end_line_idx = found_line_idx
    bracket_count = lines[found_line_idx].count('[') - lines[found_line_idx].count(']')
    if bracket_count > 0:
        for i in range(found_line_idx + 1, min(len(lines), found_line_idx + 50)):
            bracket_count += lines[i].count('[') - lines[i].count(']')
            if bracket_count == 0:
                end_line_idx = i
                break

    # Check if there's a trailing comma
    has_comma = lines[end_line_idx].rstrip().endswith(',')
    comma = ',' if has_comma else ''

    # Get the indentation of the key line
    key_indent = len(lines[found_line_idx]) - len(lines[found_line_idx].lstrip())

    # Check if the new value is multi-line
    if is_multiline(task.new_value_str):
        # Use the captured_info format directly with adjusted indentation
        captured_lines = task.new_value_str.strip().split('\n')

        # The first line is the opening bracket
        new_lines = []
        new_lines.append(f"{key_prefix}{captured_lines[0]}\n")

        # Middle lines - indent them relative to the key
        value_indent = ' ' * (key_indent + 4)
        for i in range(1, len(captured_lines) - 1):
            # Just add the proper indentation to each line from captured_info
            line_content = captured_lines[i].strip()
            new_lines.append(f"{value_indent}{line_content}\n")

        # Last line (closing bracket) - align with key
        key_indent_str = ' ' * key_indent
        last_line = captured_lines[-1].strip()
        new_lines.append(f"{key_indent_str}{last_line}{comma}\n")

        lines[found_line_idx:end_line_idx + 1] = new_lines
    else:
        # Single-line value - just insert as-is
        lines[found_line_idx:end_line_idx + 1] = [f"{key_prefix}{task.new_value_str}{comma}\n"]

    with open(filepath, 'w') as f:
        f.writelines(lines)

    return True


def update_torch_tensor(filepath: str, task: UpdateTask, info: PatternInfo) -> bool:
    """
    Update torch.tensor pattern - replaces ONLY the first argument, preserves all other parameters.

    Strategy:
    1. Use CST to locate the exact position of the first argument
    2. Replace only that argument span, keeping device=, dtype=, and any chained methods
    3. Handle both single-line and multi-line replacements
    4. For multi-line: indent relative to base_indent + 4 (standard Python indent)

    Examples:
        torch.tensor([1, 2], device="cpu").to("cuda")
        -> Only [1, 2] is replaced, device and .to() are preserved

    Args:
        filepath: Path to file to update
        task: UpdateTask with new value
        info: PatternInfo from CST analysis

    Returns:
        True if update succeeded, False otherwise
    """
    with open(filepath) as f:
        lines = f.readlines()

    # Use CST to find the exact position of the first argument
    with open(filepath) as f:
        code = f.read()

    tree = cst.parse_module(code)

    class ArgFinder(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (PositionProvider,)

        def __init__(self, target_line: int, target_var: str):
            self.target_line = target_line
            self.target_var = target_var
            self.arg_info = None

        def visit_Assign(self, node: cst.Assign) -> None:
            try:
                pos = self.get_metadata(PositionProvider, node)
                if pos.start.line != self.target_line:
                    return

                # Verify this is the correct variable
                if not isinstance(node.targets[0].target, cst.Name):
                    return
                if node.targets[0].target.value != self.target_var:
                    return

                # Find torch.tensor call (may be wrapped in chained methods)
                value = node.value

                # Unwrap chained calls like .to(), .cuda(), .float()
                # We need to find the actual torch.tensor(...) call
                while isinstance(value, cst.Call) and isinstance(value.func, cst.Attribute):
                    if isinstance(value.func.value, cst.Call):
                        value = value.func.value
                    else:
                        break

                # Extract the first argument's exact position
                if isinstance(value, cst.Call):
                    if isinstance(value.func, cst.Attribute):
                        if isinstance(value.func.value, cst.Name) and value.func.value.value == "torch":
                            if value.func.attr.value == "tensor":
                                # Found torch.tensor! Get first argument's exact position
                                if len(value.args) > 0:
                                    first_arg = value.args[0]
                                    arg_pos = self.get_metadata(PositionProvider, first_arg.value)
                                    # Store the exact character positions
                                    self.arg_info = {
                                        'start_line': arg_pos.start.line,
                                        'start_col': arg_pos.start.column,
                                        'end_line': arg_pos.end.line,
                                        'end_col': arg_pos.end.column,
                                    }
            except KeyError:
                pass

    wrapper = cst.metadata.MetadataWrapper(tree)
    finder = ArgFinder(info.line_start, task.expectations_var)
    wrapper.visit(finder)

    if not finder.arg_info:
        return False

    arg_info = finder.arg_info

    # Now use string operations to replace only the first argument
    # arg_info contains exact character positions (1-indexed)

    base_indent = info.base_indent

    # Parse the new value from captured_info
    captured_lines = task.new_value_str.strip().split('\n')

    if len(captured_lines) == 1:
        # Single-line replacement
        start_idx = arg_info['start_line'] - 1  # Convert to 0-indexed
        end_idx = arg_info['end_line'] - 1

        if start_idx == end_idx:
            # Simple case: replace within same line
            # Example: torch.tensor([1, 2], device=...)
            # Keep everything before and after [1, 2]
            line = lines[start_idx]
            new_line = line[:arg_info['start_col']] + task.new_value_str.strip() + line[arg_info['end_col']:]
            lines[start_idx] = new_line
        else:
            # Multi-line span collapsed to single-line
            # Example: torch.tensor([\n  [1, 2]\n], ...) -> torch.tensor([1, 2], ...)
            # Take prefix from first line, suffix from last line
            prefix = lines[start_idx][:arg_info['start_col']]
            suffix = lines[end_idx][arg_info['end_col']:]
            new_line = prefix + task.new_value_str.strip() + suffix
            lines[start_idx:end_idx + 1] = [new_line]
    else:
        # Multi-line replacement - needs proper indentation
        # Target structure:
        #   my_tensor = torch.tensor([
        #       [4.0, 1.0],      <- base_indent + 8
        #       [2.0, 3.0]       <- base_indent + 8
        #   ], device=...)       <- base_indent + 4 for closing ]

        start_idx = arg_info['start_line'] - 1
        end_idx = arg_info['end_line'] - 1

        # Extract prefix (before argument) and suffix (after argument)
        prefix = lines[start_idx][:arg_info['start_col']]
        suffix = lines[end_idx][arg_info['end_col']:]

        # Calculate indentation adjustment
        # captured_info has its own indentation structure
        # We need to adjust it to match our target (base_indent + 4)
        first_line_indent = len(captured_lines[0]) - len(captured_lines[0].lstrip())
        target_first_indent = base_indent + 4  # Standard Python indent
        indent_adjustment = target_first_indent - first_line_indent

        new_lines = []

        # First line: combine prefix with content
        content = captured_lines[0].lstrip()
        new_lines.append(prefix + content + '\n')

        # Middle lines: re-indent to maintain structure from captured_info
        for i in range(1, len(captured_lines) - 1):
            orig_indent = len(captured_lines[i]) - len(captured_lines[i].lstrip())
            new_indent = orig_indent + indent_adjustment
            indent_str = ' ' * new_indent
            content = captured_lines[i].lstrip()
            new_lines.append(indent_str + content + '\n')

        # Last line: closing bracket at same level as opening bracket, then suffix
        if len(captured_lines) > 1:
            orig_indent = len(captured_lines[-1]) - len(captured_lines[-1].lstrip())
            new_indent = orig_indent + indent_adjustment
            indent_str = ' ' * new_indent
            content = captured_lines[-1].lstrip()
            new_lines.append(indent_str + content + suffix)

        # Replace the span
        lines[start_idx:end_idx + 1] = new_lines

    with open(filepath, 'w') as f:
        f.writelines(lines)

    return True


def update_plain_list(filepath: str, task: UpdateTask, info: PatternInfo) -> bool:
    """Update plain list pattern."""
    with open(filepath) as f:
        lines = f.readlines()

    # Find assignment line
    found_idx = None
    for i in range(info.line_start - 1, min(len(lines), info.line_start + 5)):
        if f'{task.expectations_var} =' in lines[i] and '[' in lines[i]:
            found_idx = i
            break

    if found_idx is None:
        return False

    # Find end of assignment
    end_idx = found_idx
    if not lines[found_idx].rstrip().endswith(']'):
        bracket_count = lines[found_idx].count('[') - lines[found_idx].count(']')
        for i in range(found_idx + 1, min(len(lines), found_idx + 100)):
            bracket_count += lines[i].count('[') - lines[i].count(']')
            if bracket_count == 0:
                end_idx = i
                break

    # Check for trailing comment
    trailing_comment = ""
    if end_idx < len(lines):
        # Find comment in line using simple string operation
        line = lines[end_idx]
        comment_pos = line.find('#')
        if comment_pos != -1:
            trailing_comment = "  " + line[comment_pos:].rstrip('\n')

    base_indent = info.base_indent

    # Check if multi-line
    if is_multiline(task.new_value_str):
        # Use captured_info format directly with proper indentation
        indent_str = ' ' * base_indent
        inner_indent_str = ' ' * (base_indent + 4)

        # Split captured_info lines
        captured_lines = task.new_value_str.strip().split('\n')

        new_lines = []
        new_lines.append(f"{indent_str}{task.expectations_var} = {captured_lines[0]}\n")

        # Middle lines - just re-indent from captured_info
        for i in range(1, len(captured_lines) - 1):
            line_content = captured_lines[i].strip()
            new_lines.append(f"{inner_indent_str}{line_content}\n")

        # Last line (closing bracket) - align with base
        new_lines.append(f"{indent_str}{captured_lines[-1].strip()}{trailing_comment}\n")

        lines[found_idx:end_idx + 1] = new_lines
    else:
        # Single-line
        indent_str = ' ' * base_indent
        lines[found_idx:end_idx + 1] = [
            f"{indent_str}{task.expectations_var} = {task.new_value_str}{trailing_comment}\n"]

    with open(filepath, 'w') as f:
        f.writelines(lines)

    return True


def update_plain_dict(filepath: str, task: UpdateTask, info: PatternInfo) -> bool:
    """Update plain dict pattern."""
    with open(filepath) as f:
        lines = f.readlines()

    # Find assignment line
    found_idx = None
    for i in range(info.line_start - 1, min(len(lines), info.line_start + 5)):
        if f'{task.expectations_var} =' in lines[i] and '{' in lines[i]:
            found_idx = i
            break

    if found_idx is None:
        return False

    # Find end of assignment
    end_idx = found_idx
    if not lines[found_idx].rstrip().endswith('}'):
        brace_count = lines[found_idx].count('{') - lines[found_idx].count('}')
        for i in range(found_idx + 1, min(len(lines), found_idx + 100)):
            brace_count += lines[i].count('{') - lines[i].count('}')
            if brace_count == 0:
                end_idx = i
                break

    # Check for trailing comment
    trailing_comment = ""
    if end_idx < len(lines):
        # Find comment in line using simple string operation
        line = lines[end_idx]
        comment_pos = line.find('#')
        if comment_pos != -1:
            trailing_comment = "  " + line[comment_pos:].rstrip('\n')

    base_indent = info.base_indent

    # Check if multi-line
    if is_multiline(task.new_value_str):
        # Keep multi-line format from captured_info
        # Just use the captured_info value directly with proper indentation
        indent_str = ' ' * base_indent

        # Split captured_info into lines and re-indent
        captured_lines = task.new_value_str.split('\n')

        new_lines = []
        new_lines.append(f"{indent_str}{task.expectations_var} = {captured_lines[0]}\n")

        for i in range(1, len(captured_lines)):
            if i == len(captured_lines) - 1:
                # Last line - add trailing comment
                new_lines.append(f"{indent_str}{captured_lines[i]}{trailing_comment}\n")
            else:
                new_lines.append(f"{indent_str}{captured_lines[i]}\n")

        lines[found_idx:end_idx + 1] = new_lines
    else:
        # Single-line
        indent_str = ' ' * base_indent
        lines[found_idx:end_idx + 1] = [
            f"{indent_str}{task.expectations_var} = {task.new_value_str}{trailing_comment}\n"]

    with open(filepath, 'w') as f:
        f.writelines(lines)

    return True


def update_plain_string(filepath: str, task: UpdateTask, info: PatternInfo) -> bool:
    """Update plain string pattern."""
    with open(filepath) as f:
        lines = f.readlines()

    # Find assignment line
    found_idx = None
    for i in range(info.line_start - 1, min(len(lines), info.line_start + 5)):
        if f'{task.expectations_var} =' in lines[i]:
            found_idx = i
            break

    if found_idx is None:
        return False

    # Find end (for multi-line strings)
    end_idx = found_idx
    if '"""' in lines[found_idx]:
        quote_count = lines[found_idx].count('"""')
        if quote_count == 1:
            for i in range(found_idx + 1, min(len(lines), found_idx + 100)):
                if '"""' in lines[i]:
                    end_idx = i
                    break

    base_indent = info.base_indent
    indent_str = ' ' * base_indent

    # If new_value_str starts with quote, use as-is
    if task.new_value_str.startswith('"') or task.new_value_str.startswith("'"):
        new_line = f'{indent_str}{task.expectations_var} = {task.new_value_str}\n'
    else:
        new_line = f'{indent_str}{task.expectations_var} = "{task.new_value_str}"\n'

    lines[found_idx:end_idx + 1] = [new_line]

    with open(filepath, 'w') as f:
        f.writelines(lines)

    return True


def update_inline_simple(filepath: str, task: UpdateTask, info: PatternInfo) -> bool:
    """
    Update an inline literal (number, tuple, etc) that appears as an argument.

    This is a simple replacement that finds and replaces the literal value
    using the position information from CST.

    Args:
        filepath: Path to file
        task: UpdateTask
        info: PatternInfo with exact line positions from CST

    Returns:
        True if successful
    """
    with open(filepath) as f:
        lines = f.readlines()

    start_idx = info.line_start - 1
    end_idx = info.line_end - 1

    if start_idx >= len(lines) or end_idx >= len(lines):
        return False

    # Only handle single-line literals
    if start_idx == end_idx:
        line = lines[start_idx]
        # Use column information to replace exactly at the right position
        # For now, just replace the old value with the new value in the line
        old_val = task.expectations_var
        if old_val in line:
            # Simple replacement - replace first occurrence
            new_line = line.replace(old_val, task.new_value_str, 1)
            lines[start_idx] = new_line

            with open(filepath, 'w') as f:
                f.writelines(lines)
            return True

    return False


def update_inline_string(filepath: str, task: UpdateTask, info: PatternInfo) -> bool:
    """
    Update an inline string literal that appears as an argument in a function call.

    Example: self.assertEqual(generated_text, "old text")
    Replace "old text" with the new value.

    Args:
        filepath: Path to file
        task: UpdateTask
        info: PatternInfo with exact line positions from CST

    Returns:
        True if successful
    """
    with open(filepath) as f:
        lines = f.readlines()

    # CST gives us exact line positions of the string
    start_idx = info.line_start - 1
    end_idx = info.line_end - 1

    if start_idx >= len(lines) or end_idx >= len(lines):
        return False

    # For single-line inline string
    if start_idx == end_idx:
        line = lines[start_idx]

        # The new_value_str from captured_info includes quotes, so strip them
        new_content = task.new_value_str.strip()
        if new_content.startswith('"') and new_content.endswith('"'):
            new_content = new_content[1:-1]
        elif new_content.startswith("'") and new_content.endswith("'"):
            new_content = new_content[1:-1]

        # Find the string in the line - look for quotes
        # Try both " and '
        for quote_char in ['"', "'"]:
            quote_pos = line.find(quote_char)
            if quote_pos != -1:
                # Find closing quote
                end_quote = line.find(quote_char, quote_pos + 1)
                if end_quote != -1:
                    # Replace the string content
                    # Keep the quote character from the original
                    new_line = line[:quote_pos] + quote_char + new_content + quote_char + line[end_quote + 1:]
                    lines[start_idx] = new_line

                    with open(filepath, 'w') as f:
                        f.writelines(lines)
                    return True

    return False


def update_inline_list(filepath: str, task: UpdateTask, info: PatternInfo) -> bool:
    """
    Update an inline list that appears as an argument in a function call.

    Uses position info from CST to replace just the list value.

    Args:
        filepath: Path to file
        task: UpdateTask
        info: PatternInfo with exact line positions from CST

    Returns:
        True if successful
    """
    with open(filepath) as f:
        lines = f.readlines()

    # CST gives us exact line positions of the list
    start_idx = info.line_start - 1
    end_idx = info.line_end - 1

    if start_idx >= len(lines) or end_idx >= len(lines):
        return False

    # For single-line inline list, just replace the value in place
    if start_idx == end_idx:
        line = lines[start_idx]
        # Find the list in the line - look for '[' and matching ']'
        list_start = line.find('[')
        if list_start == -1:
            return False

        # Find matching ']'
        bracket_depth = 0
        list_end = -1
        for i in range(list_start, len(line)):
            if line[i] == '[':
                bracket_depth += 1
            elif line[i] == ']':
                bracket_depth -= 1
                if bracket_depth == 0:
                    list_end = i
                    break

        if list_end == -1:
            return False

        # Replace just the list part
        new_line = line[:list_start] + task.new_value_str + line[list_end + 1:]
        lines[start_idx] = new_line

        with open(filepath, 'w') as f:
            f.writelines(lines)
        return True

    return False


def update_file(filepath: str, tasks: List[UpdateTask], dry_run: bool = True) -> bool:
    """
    Update a file with all tasks using the backward update approach.

    The backward update approach solves the line-shifting problem:
    - When multiple updates target the same file, early updates can change line counts
    - This would invalidate line numbers for subsequent updates
    - Solution: Process updates from HIGHEST to LOWEST line number
    - Each update only affects lines ABOVE it, keeping subsequent positions valid

    Algorithm:
    1. Analyze ALL tasks first (collect original line positions)
    2. Sort tasks by line number DESCENDING (highest first)
    3. Apply updates in backward order
    4. No line number tracking needed!

    Example:
        File with 3 variables at lines 10, 20, 30
        - Update line 30 first (file now changed, but lines 10, 20 unchanged)
        - Update line 20 next (file changed, but line 10 unchanged)
        - Update line 10 last (all done!)

    Args:
        filepath: Path to file to update
        tasks: List of UpdateTask objects for this file
        dry_run: If True, only show what would be updated (don't modify file)

    Returns:
        True (always returns True, even if some updates fail)
    """

    # Step 1: Analyze ALL tasks before making any updates
    # This captures the original line positions before any modifications
    analyzed_tasks = []
    for task in tasks:
        # Check if this is an inline literal expression
        if is_literal_expression(task.expectations_var):
            # Analyze inline expression
            info = analyze_inline_expression(filepath, task.expectations_lineno, task.expectations_var)
        else:
            # Analyze variable assignment
            info = analyze_with_cst(filepath, task.expectations_lineno, task.expectations_var)

        if not info:
            print(f"  ✗ Could not analyze {task.expectations_var} at line {task.expectations_lineno}")
            continue

        analyzed_tasks.append((task, info))

    # Step 2: Sort by line number DESCENDING (highest line first)
    # This ensures updates only affect lines ABOVE, keeping subsequent positions valid
    analyzed_tasks.sort(key=lambda x: x[1].line_start, reverse=True)

    # Step 3: Process updates in backward order (highest line number first)
    for task, info in analyzed_tasks:
        print(f"  Processing {task.expectations_var} at line {task.expectations_lineno}")
        print(f"    Pattern: {info.pattern_type}")
        print(f"    Base indent: {info.base_indent}")
        print(f"    Multi-line: {is_multiline(task.new_value_str)}")

        if dry_run:
            # Check if pattern is supported before claiming we'd update
            if info.pattern_type == "unknown":
                print(f"    ✗ Unknown pattern")
            else:
                print(f"    ✓ Would update")
            continue

        # Apply update based on pattern type
        success = False

        # Check if this is an inline literal expression (no assignment)
        is_inline = is_literal_expression(task.expectations_var)

        if info.pattern_type == "Expectations":
            # For Expectations pattern: try different device keys in order of specificity
            # This handles hardware-specific expected values: ("cuda", (8, 6)) for CUDA 8.6, etc.
            device_keys_to_try = [
                ("cuda", (8, 6)),  # Most specific: cuda with nested tuple (version)
                ("cuda", 8),  # Specific: cuda with simple version number
                ("cuda", None),  # Less specific: cuda without version
                (None, None),  # Least specific: fallback key (matches any hardware)
            ]
            for device in device_keys_to_try:
                success = update_expectations(filepath, task, info, device)
                if success:
                    break  # Stop on first successful match
        elif info.pattern_type == "torch.tensor":
            success = update_torch_tensor(filepath, task, info)
        elif info.pattern_type == "plain_list":
            if is_inline:
                # Use inline-specific updater
                success = update_inline_list(filepath, task, info)
            else:
                success = update_plain_list(filepath, task, info)
        elif info.pattern_type == "plain_dict":
            success = update_plain_dict(filepath, task, info)
        elif info.pattern_type == "plain_string":
            if is_inline:
                # Use inline-specific updater
                success = update_inline_string(filepath, task, info)
            else:
                success = update_plain_string(filepath, task, info)
        elif info.pattern_type == "plain_number":
            # Numbers are always inline, use simple replacement
            success = update_inline_simple(filepath, task, info)
        elif info.pattern_type == "plain_tuple":
            # Tuples are always inline, use simple replacement
            success = update_inline_simple(filepath, task, info)
        else:
            print(f"    ✗ Unknown pattern")
            continue

        if success:
            print(f"    ✓ Updated")
        else:
            print(f"    ✗ Update failed")

    if dry_run:
        print(f"  💡 DRY RUN - No files modified")
    else:
        print(f"  ✅ File updated: {filepath}")

    return True


def main():
    """
    Main entry point for the expectations updater.

    Workflow:
    1. Parse captured_info.txt to extract update tasks
    2. Group tasks by file (multiple updates per file supported)
    3. Process each file using backward update approach
    4. Dry run by default (requires --apply flag to modify files)
    """
    parser = argparse.ArgumentParser(description="Update test expectations")
    parser.add_argument("captured_info", help="Path to captured_info.txt")
    parser.add_argument("--apply", action="store_true", help="Apply changes (default is dry run)")
    args = parser.parse_args()

    print("=" * 80)
    print("CST + String Ops Expectations Updater v7 (Backward)")
    print("=" * 80)
    print()

    # Step 1: Parse captured_info to extract all update tasks
    tasks, stats = parse_captured_info(args.captured_info)
    print(f"[1] Found {len(tasks)} update task(s)")
    for task in tasks:
        print(f"    - {task.expectations_var} at {task.test_file}:{task.expectations_lineno}")
    print()

    # Step 2: Group tasks by file (some files may have multiple updates)
    file_tasks: Dict[str, List[UpdateTask]] = {}
    for task in tasks:
        if task.test_file not in file_tasks:
            file_tasks[task.test_file] = []
        file_tasks[task.test_file].append(task)

    print(f"[2] Grouped into {len(file_tasks)} file(s)")
    print()

    # Step 3: Process each file (backward update approach handles multiple updates safely)
    for filepath, tasks_for_file in file_tasks.items():
        print(f"[3] Processing {filepath}...")
        update_file(filepath, tasks_for_file, dry_run=not args.apply)
        print()

    print("=" * 80)
    print("✅ Complete!")
    print("=" * 80)
    print()
    print(f"STATS: blocks={stats['blocks']} skip={stats['skipped']} tasks={len(tasks)}")


if __name__ == "__main__":
    main()