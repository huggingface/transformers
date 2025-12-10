#!/usr/bin/env python3
"""
Enhanced CST-based updater for Expectations pattern in test files.

This version handles multiple test failures in a single captured_info.txt file.
It processes all updates in a single pass to maintain line number integrity.

Key features:
- Parses ALL test contexts from captured_info.txt
- Groups updates by test file
- Applies all updates to each file in a single CST pass
- Processes updates from bottom to top to preserve line numbers
"""

import re
import argparse
import libcst as cst
from libcst.metadata import PositionProvider
from pathlib import Path
from typing import Optional, Tuple, Any, Union, List, Dict
from dataclasses import dataclass


@dataclass
class UpdateTask:
    """Represents a single update to be performed."""
    test_file: str
    assertion_line: int
    variable_name: str
    new_value_str: str
    expectations_var: str  # The actual variable to update
    expectations_lineno: int  # Line number of the assignment


class ExpectationsUpdater(cst.CSTTransformer):
    """CST transformer that updates multiple assignments in a single pass."""

    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(
            self,
            update_tasks: List[UpdateTask],
            target_keys: list,
            dry_run: bool = True
    ):
        """
        Initialize the transformer.

        Args:
            update_tasks: List of updates to perform (sorted by line number descending)
            target_keys: List of keys to try in order of preference
            dry_run: If True, don't actually modify, just report
        """
        self.update_tasks = update_tasks
        self.target_keys = target_keys
        self.dry_run = dry_run

        # Build a map of line_number -> UpdateTask for quick lookup
        self.updates_by_line = {task.expectations_lineno: task for task in update_tasks}

        # Track results
        self.successful_updates = []
        self.failed_updates = []

    def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.Assign:
        """Visit assignment nodes and check if they need updating."""
        # Get line number of this assignment
        try:
            pos = self.get_metadata(PositionProvider, original_node)
            line_number = pos.start.line
        except KeyError:
            return updated_node

        # Check if this line has an update task
        if line_number not in self.updates_by_line:
            return updated_node

        task = self.updates_by_line[line_number]

        # Check if this is the right variable
        if not isinstance(updated_node.targets[0].target, cst.Name):
            return updated_node

        var_name = updated_node.targets[0].target.value
        if var_name != task.expectations_var:
            return updated_node

        print(f"\n  Processing update at line {line_number}: {var_name}")

        # Find what pattern this is
        pattern_type, pattern_node = self._find_pattern_in_tree(updated_node.value)

        if not pattern_type:
            print(f"    ❌ Could not identify pattern")
            self.failed_updates.append((task, "Pattern not recognized"))
            return updated_node

        print(f"    Pattern: {pattern_type}")

        # Handle based on pattern type
        if pattern_type == "Expectations":
            return self._handle_expectations_pattern(original_node, updated_node, task, pattern_node)
        elif pattern_type in ["torch.tensor", "plain_list", "plain_dict"]:
            return self._handle_direct_pattern(original_node, updated_node, task, pattern_node, pattern_type)
        else:
            self.failed_updates.append((task, f"Unknown pattern: {pattern_type}"))
            return updated_node

    def _find_pattern_in_tree(self, node: cst.BaseExpression) -> Tuple[Optional[str], Optional[Any]]:
        """Recursively find Expectations, torch.tensor, plain list, or plain dict in the tree."""
        # Check for Expectations call
        if isinstance(node, cst.Call):
            if isinstance(node.func, cst.Name) and node.func.value == "Expectations":
                return "Expectations", node
            # Check for torch.tensor
            if isinstance(node.func, cst.Attribute):
                if (isinstance(node.func.value, cst.Name) and
                        node.func.value.value == "torch" and
                        node.func.attr.value == "tensor"):
                    return "torch.tensor", node
            # Recursively check the function being called (for method chaining)
            if isinstance(node.func, cst.Attribute) and isinstance(node.func.value, cst.Call):
                result = self._find_pattern_in_tree(node.func.value)
                if result[0]:
                    return result

        # Check for torch.tensor as attribute access
        if isinstance(node, cst.Attribute):
            result = self._find_pattern_in_tree(node.value)
            if result[0]:
                return result

        # Check for plain list
        if isinstance(node, cst.List):
            return "plain_list", node

        # Check for plain dict
        if isinstance(node, cst.Dict):
            return "plain_dict", node

        return None, None

    def _handle_expectations_pattern(
            self,
            original_node: cst.Assign,
            updated_node: cst.Assign,
            task: UpdateTask,
            pattern_node: cst.Call
    ) -> cst.Assign:
        """Handle Expectations dictionary pattern."""
        if not pattern_node.args or len(pattern_node.args) == 0:
            self.failed_updates.append((task, "No dictionary argument in Expectations"))
            return updated_node

        old_arg = pattern_node.args[0]  # Get the full Arg, not just the value
        dict_arg = old_arg.value
        if not isinstance(dict_arg, cst.Dict):
            self.failed_updates.append((task, "Expectations argument is not a dictionary"))
            return updated_node

        # Find matching key and update
        new_dict = self._update_dict(dict_arg, task)
        if new_dict is None:
            self.failed_updates.append((task, "Could not find matching key"))
            return updated_node

        # Reconstruct the tree with updated dict, preserving the Arg's whitespace
        new_arg = old_arg.with_changes(value=new_dict)  # This preserves whitespace_after_arg!
        new_call = pattern_node.with_changes(
            args=[new_arg] + list(pattern_node.args[1:])  # Preserve any other args
        )

        # Replace the pattern node in the tree
        new_value = self._replace_node_in_tree(updated_node.value, pattern_node, new_call)

        if not self.dry_run:
            self.successful_updates.append((task, "Expectations"))
            return updated_node.with_changes(value=new_value)
        else:
            self.successful_updates.append((task, "Expectations"))
            return updated_node

    def _handle_direct_pattern(
            self,
            original_node: cst.Assign,
            updated_node: cst.Assign,
            task: UpdateTask,
            pattern_node: Any,
            pattern_type: str
    ) -> cst.Assign:
        """Handle direct torch.tensor, plain list, or plain dict pattern."""
        try:
            # Parse the new value - this already has the formatting from captured_info.txt
            new_value_node = cst.parse_expression(task.new_value_str)

            # Use the formatting from captured_info.txt
            if isinstance(pattern_node, cst.Call):
                # For torch.tensor, update the argument
                if pattern_node.args and len(pattern_node.args) > 0:
                    # For torch.tensor, captured_info has 4 spaces, we want to make it base + 8
                    # Get the base indentation from the assignment line
                    base_indent = self._get_assignment_indent(original_node)

                    # We need content at base + 8 (e.g., 16 spaces from line start)
                    # Captured_info has 4 spaces
                    # After testing: adding base spaces gave us base+12 in output
                    # So we need to add base/2 spaces (e.g., 4)
                    extra_spaces = base_indent // 2  # e.g., 8 // 2 = 4

                    lines = task.new_value_str.split('\n')
                    adjusted_lines = []
                    for i, line in enumerate(lines):
                        if i == 0:
                            # First line (opening [)
                            adjusted_lines.append(line)
                        elif line.strip() == '':
                            adjusted_lines.append(line)
                        elif line.strip() == ']':
                            # Closing ] - add extra spaces
                            adjusted_lines.append((" " * extra_spaces) + line.lstrip())
                        elif line.lstrip().startswith('['):
                            # Content line - ADD to existing indent, don't replace
                            leading_spaces = len(line) - len(line.lstrip())
                            adjusted_lines.append((" " * (leading_spaces + extra_spaces)) + line.lstrip())
                        else:
                            adjusted_lines.append((" " * extra_spaces) + line)

                    adjusted_value_str = '\n'.join(adjusted_lines)
                    new_value_node = cst.parse_expression(adjusted_value_str)

                    # Preserve the old Arg
                    old_arg = pattern_node.args[0]
                    new_arg = old_arg.with_changes(value=new_value_node)
                    new_call = pattern_node.with_changes(
                        args=[new_arg] + list(pattern_node.args[1:])
                    )
                    new_full_value = self._replace_node_in_tree(updated_node.value, pattern_node, new_call)
                else:
                    self.failed_updates.append((task, "No arguments in torch.tensor call"))
                    return updated_node
            else:
                # For plain list/dict, use new_value_node directly (respects captured_info.txt formatting)
                new_full_value = new_value_node

            if not self.dry_run:
                self.successful_updates.append((task, pattern_type))
                return updated_node.with_changes(value=new_full_value)
            else:
                self.successful_updates.append((task, pattern_type))
                return updated_node

        except Exception as e:
            self.failed_updates.append((task, f"Error: {str(e)}"))
            return updated_node

    def _update_dict(self, dict_node: cst.Dict, task: UpdateTask) -> Optional[cst.Dict]:
        """Update the dictionary with priority-based key selection."""
        # Parse new value - this has minimal formatting from captured_info.txt
        try:
            new_value_node = cst.parse_expression(task.new_value_str)
        except Exception as e:
            print(f"    ❌ Error parsing new value: {e}")
            return None

        # Find all matching keys with priority
        all_matches = []
        for element in dict_node.elements:
            if not isinstance(element, cst.DictElement):
                continue

            key_tuple = self._match_key(element.key)
            if key_tuple is None:
                continue

            # Check against target keys
            for priority, target_key in enumerate(self.target_keys):
                if self._keys_match(key_tuple, target_key):
                    all_matches.append((priority, element, key_tuple))
                    break

        if not all_matches:
            print(f"    ❌ No matching key found")
            return None

        # Select best match (lowest priority)
        all_matches.sort(key=lambda x: x[0])
        best_priority, best_element, best_key = all_matches[0]

        print(f"    ✓ Found key: {best_key} (priority: {best_priority})")

        # Use the captured_info formatting AS-IS, don't reformat
        # Just use new_value_node directly (preserves single-line vs multi-line from captured_info)
        new_value_adjusted = new_value_node

        # Create new element with updated value
        new_element = best_element.with_changes(value=new_value_adjusted)

        # Replace in dictionary
        new_elements = []
        for element in dict_node.elements:
            if element is best_element:
                new_elements.append(new_element)
            else:
                new_elements.append(element)

        # Preserve the original dict's brace formatting
        return dict_node.with_changes(elements=new_elements)

    def _match_key(self, key_node: cst.BaseExpression) -> Optional[Tuple]:
        """Parse a dictionary key into a tuple."""
        if isinstance(key_node, cst.Tuple):
            elements = []
            for el in key_node.elements:
                if isinstance(el.value, cst.Name):
                    if el.value.value == "None":
                        elements.append(None)
                    else:
                        elements.append(el.value.value)
                elif isinstance(el.value, cst.SimpleString):
                    elements.append(el.value.evaluated_value)
                elif isinstance(el.value, cst.Integer):
                    elements.append(int(el.value.value))
                elif isinstance(el.value, cst.Tuple):
                    # Nested tuple for compute capability
                    nested = []
                    for nel in el.value.elements:
                        if isinstance(nel.value, cst.Integer):
                            nested.append(int(nel.value.value))
                    elements.append(tuple(nested))
            return tuple(elements)
        return None

    def _keys_match(self, key1: Tuple, key2: Tuple) -> bool:
        """Check if two keys match, handling 'ANY' wildcards."""
        if len(key1) != len(key2):
            return False

        for k1, k2 in zip(key1, key2):
            if k2 == "ANY":
                continue
            if k1 == k2:
                continue
            if k1 is None and k2 is None:
                continue
            return False
        return True

    def _get_dict_indent(self, dict_node: cst.Dict) -> int:
        """
        Get the base indentation (in spaces) of dict elements by examining
        the dict's lbrace whitespace.
        """
        if hasattr(dict_node.lbrace, 'whitespace_after'):
            ws = dict_node.lbrace.whitespace_after
            if isinstance(ws, cst.ParenthesizedWhitespace) and hasattr(ws, 'last_line'):
                if hasattr(ws.last_line, 'value'):
                    return len(ws.last_line.value)
        # Default fallback
        return 16

    def _get_assignment_indent(self, assign_node: cst.Assign) -> int:
        """
        Get the base indentation (in spaces) of an assignment statement.
        This looks at the leading whitespace of the assignment line.
        """
        # Try to get the parent statement line
        try:
            # The assignment is wrapped in a SimpleStatementLine
            # We need to access it through metadata or inspection
            # For now, use a heuristic based on common patterns
            # Most test assignments are indented 8 spaces (2 levels)
            return 8
        except:
            return 8  # Default for test methods (2 indentation levels)

    def _get_node_indent(self, node: cst.DictElement) -> int:
        """
        Get the indentation level (in spaces) of a DictElement.
        We need to examine the source code to determine this.
        """
        # Try to get position from metadata if available
        # For now, use a heuristic: check the dict's lbrace whitespace
        # This is tricky without access to the full tree context
        # Return a reasonable default that will be adjusted
        return 12  # Will be adjusted based on actual context

    def _apply_proper_indentation(self, node: Any, base_indent_spaces: int) -> Any:
        """
        Apply proper indentation to a parsed node.
        base_indent_spaces: The starting indentation in spaces for the first level.
        """
        indent_str = " " * base_indent_spaces

        if isinstance(node, cst.List):
            # Set up proper bracket whitespace
            new_lbracket = cst.LeftSquareBracket(
                whitespace_after=cst.ParenthesizedWhitespace(
                    first_line=cst.TrailingWhitespace(
                        whitespace=cst.SimpleWhitespace(""),
                        newline=cst.Newline(value=None)
                    ),
                    indent=True,
                    last_line=cst.SimpleWhitespace(indent_str)
                )
            )

            # The closing bracket should be at base_indent_spaces - 4
            closing_indent = " " * max(0, base_indent_spaces - 4)
            new_rbracket = cst.RightSquareBracket(
                whitespace_before=cst.ParenthesizedWhitespace(
                    first_line=cst.TrailingWhitespace(
                        whitespace=cst.SimpleWhitespace(""),
                        newline=cst.Newline(value=None)
                    ),
                    indent=True,
                    last_line=cst.SimpleWhitespace(closing_indent)
                )
            )

            # Process elements
            new_elements = []
            for i, element in enumerate(node.elements):
                # For list elements, the value uses the same indent level
                # (not base_indent_spaces + 4, that's already applied to the list itself)
                new_value = element.value  # Don't recursively indent simple values

                # Set up comma with newline and indent for next element
                if i < len(node.elements) - 1:
                    new_comma = cst.Comma(
                        whitespace_before=cst.SimpleWhitespace(""),
                        whitespace_after=cst.ParenthesizedWhitespace(
                            first_line=cst.TrailingWhitespace(
                                whitespace=cst.SimpleWhitespace(""),
                                newline=cst.Newline(value=None)
                            ),
                            indent=True,
                            last_line=cst.SimpleWhitespace(indent_str)
                        )
                    )
                else:
                    # Last element - no trailing comma or use the original
                    new_comma = element.comma if hasattr(element, 'comma') else cst.MaybeSentinel.DEFAULT

                new_element = element.with_changes(value=new_value, comma=new_comma)
                new_elements.append(new_element)

            return node.with_changes(
                lbracket=new_lbracket,
                rbracket=new_rbracket,
                elements=new_elements
            )

        elif isinstance(node, cst.Dict):
            # Similar logic for dicts
            new_lbrace = cst.LeftCurlyBrace(
                whitespace_after=cst.ParenthesizedWhitespace(
                    first_line=cst.TrailingWhitespace(
                        whitespace=cst.SimpleWhitespace(""),
                        newline=cst.Newline(value=None)
                    ),
                    indent=True,
                    last_line=cst.SimpleWhitespace(indent_str)
                )
            )

            closing_indent = " " * max(0, base_indent_spaces - 4)
            new_rbrace = cst.RightCurlyBrace(
                whitespace_before=cst.ParenthesizedWhitespace(
                    first_line=cst.TrailingWhitespace(
                        whitespace=cst.SimpleWhitespace(""),
                        newline=cst.Newline(value=None)
                    ),
                    indent=True,
                    last_line=cst.SimpleWhitespace(closing_indent)
                )
            )

            # Process elements
            new_elements = []
            for i, element in enumerate(node.elements):
                if isinstance(element, cst.DictElement):
                    new_key = self._apply_proper_indentation(element.key, base_indent_spaces + 4)
                    new_value = self._apply_proper_indentation(element.value, base_indent_spaces + 4)

                    if i < len(node.elements) - 1:
                        new_comma = cst.Comma(
                            whitespace_before=cst.SimpleWhitespace(""),
                            whitespace_after=cst.ParenthesizedWhitespace(
                                first_line=cst.TrailingWhitespace(
                                    whitespace=cst.SimpleWhitespace(""),
                                    newline=cst.Newline(value=None)
                                ),
                                indent=True,
                                last_line=cst.SimpleWhitespace(indent_str)
                            )
                        )
                    else:
                        new_comma = element.comma

                    new_element = element.with_changes(key=new_key, value=new_value, comma=new_comma)
                    new_elements.append(new_element)

            return node.with_changes(
                lbrace=new_lbrace,
                rbrace=new_rbrace,
                elements=new_elements
            )

        # For other types, return as-is
        return node

    def _copy_indentation_structure(self, old_node: Any, new_node: Any) -> Any:
        """
        Copy the indentation structure from old_node to new_node.
        This preserves the original file's formatting while updating values.
        """
        if isinstance(old_node, cst.List) and isinstance(new_node, cst.List):
            # Copy the bracket whitespace
            new_node = new_node.with_changes(
                lbracket=old_node.lbracket,
                rbracket=old_node.rbracket
            )

            # If element counts match, copy per-element formatting
            if len(old_node.elements) == len(new_node.elements):
                new_elements = []
                for old_el, new_el in zip(old_node.elements, new_node.elements):
                    # Recursively copy structure for the value
                    new_value = self._copy_indentation_structure(old_el.value, new_el.value)
                    # Copy the comma whitespace
                    new_element = new_el.with_changes(
                        value=new_value,
                        comma=old_el.comma
                    )
                    new_elements.append(new_element)
                new_node = new_node.with_changes(elements=new_elements)

            return new_node

        elif isinstance(old_node, cst.Dict) and isinstance(new_node, cst.Dict):
            # Copy the brace whitespace
            new_node = new_node.with_changes(
                lbrace=old_node.lbrace,
                rbrace=old_node.rbrace
            )

            # If element counts match, copy per-element formatting
            if len(old_node.elements) == len(new_node.elements):
                new_elements = []
                for old_el, new_el in zip(old_node.elements, new_node.elements):
                    if isinstance(old_el, cst.DictElement) and isinstance(new_el, cst.DictElement):
                        # Recursively copy structure
                        new_key = self._copy_indentation_structure(old_el.key, new_el.key)
                        new_value = self._copy_indentation_structure(old_el.value, new_el.value)
                        # Copy comma and colon whitespace
                        new_element = new_el.with_changes(
                            key=new_key,
                            value=new_value,
                            comma=old_el.comma,
                            whitespace_before_colon=old_el.whitespace_before_colon,
                            whitespace_after_colon=old_el.whitespace_after_colon
                        )
                        new_elements.append(new_element)
                new_node = new_node.with_changes(elements=new_elements)

            return new_node

        # For other types (strings, numbers, etc.), return new value as-is
        return new_node

    def _get_indent_level(self, node: Any) -> int:
        """
        Detect the indentation level of a node by examining its whitespace.
        Returns the number of 4-space indents.
        """
        if isinstance(node, cst.List):
            # Check the lbracket's whitespace_after
            if hasattr(node.lbracket, 'whitespace_after'):
                ws = node.lbracket.whitespace_after
                if isinstance(ws, cst.ParenthesizedWhitespace) and hasattr(ws, 'last_line'):
                    indent_str = ws.last_line.value if hasattr(ws.last_line, 'value') else ""
                    # Count spaces and divide by 4
                    return len(indent_str) // 4
        elif isinstance(node, cst.Dict):
            # Similar for dicts
            if hasattr(node.lbrace, 'whitespace_after'):
                ws = node.lbrace.whitespace_after
                if isinstance(ws, cst.ParenthesizedWhitespace) and hasattr(ws, 'last_line'):
                    indent_str = ws.last_line.value if hasattr(ws.last_line, 'value') else ""
                    return len(indent_str) // 4

        # Default: assume 5 levels for Expectations values (typical case)
        return 5

    def _adjust_indentation(self, new_node: Any, target_indent_level: int) -> Any:
        """
        Adjust the indentation of a parsed node to match the target context.

        Args:
            new_node: The parsed CST node (from captured_info.txt)
            target_indent_level: Number of 4-space indents needed

        Returns:
            Node with adjusted indentation
        """
        indent_str = "    " * target_indent_level

        if isinstance(new_node, cst.List):
            # Adjust list bracket whitespace
            new_lbracket = cst.LeftSquareBracket(
                whitespace_after=cst.ParenthesizedWhitespace(
                    first_line=cst.TrailingWhitespace(
                        whitespace=cst.SimpleWhitespace(""),
                        newline=cst.Newline(value=None)
                    ),
                    indent=True,
                    last_line=cst.SimpleWhitespace(indent_str)
                )
            )
            new_rbracket = cst.RightSquareBracket(
                whitespace_before=cst.ParenthesizedWhitespace(
                    first_line=cst.TrailingWhitespace(
                        whitespace=cst.SimpleWhitespace(""),
                        newline=cst.Newline(value=None)
                    ),
                    indent=True,
                    last_line=cst.SimpleWhitespace(indent_str[:-4] if target_indent_level > 0 else "")
                )
            )

            # Adjust each element's comma whitespace
            new_elements = []
            for i, element in enumerate(new_node.elements):
                # Recursively adjust nested structures
                new_value = self._adjust_indentation(element.value, target_indent_level + 1)

                # Set comma with newline and indent for next element
                if i < len(new_node.elements) - 1:
                    new_comma = cst.Comma(
                        whitespace_before=cst.SimpleWhitespace(""),
                        whitespace_after=cst.ParenthesizedWhitespace(
                            first_line=cst.TrailingWhitespace(
                                whitespace=cst.SimpleWhitespace(""),
                                newline=cst.Newline(value=None)
                            ),
                            indent=True,
                            last_line=cst.SimpleWhitespace(indent_str)
                        )
                    )
                else:
                    # Last element - no comma or just comma without newline
                    new_comma = cst.MaybeSentinel.DEFAULT

                new_element = element.with_changes(value=new_value, comma=new_comma)
                new_elements.append(new_element)

            return new_node.with_changes(
                lbracket=new_lbracket,
                rbracket=new_rbracket,
                elements=new_elements
            )

        elif isinstance(new_node, cst.Dict):
            # Similar logic for dicts
            new_lbrace = cst.LeftCurlyBrace(
                whitespace_after=cst.ParenthesizedWhitespace(
                    first_line=cst.TrailingWhitespace(
                        whitespace=cst.SimpleWhitespace(""),
                        newline=cst.Newline(value=None)
                    ),
                    indent=True,
                    last_line=cst.SimpleWhitespace(indent_str)
                )
            )
            new_rbrace = cst.RightCurlyBrace(
                whitespace_before=cst.ParenthesizedWhitespace(
                    first_line=cst.TrailingWhitespace(
                        whitespace=cst.SimpleWhitespace(""),
                        newline=cst.Newline(value=None)
                    ),
                    indent=True,
                    last_line=cst.SimpleWhitespace(indent_str[:-4] if target_indent_level > 0 else "")
                )
            )

            new_elements = []
            for i, element in enumerate(new_node.elements):
                if isinstance(element, cst.DictElement):
                    new_key = self._adjust_indentation(element.key, target_indent_level + 1)
                    new_value = self._adjust_indentation(element.value, target_indent_level + 1)

                    if i < len(new_node.elements) - 1:
                        new_comma = cst.Comma(
                            whitespace_before=cst.SimpleWhitespace(""),
                            whitespace_after=cst.ParenthesizedWhitespace(
                                first_line=cst.TrailingWhitespace(
                                    whitespace=cst.SimpleWhitespace(""),
                                    newline=cst.Newline(value=None)
                                ),
                                indent=True,
                                last_line=cst.SimpleWhitespace(indent_str)
                            )
                        )
                    else:
                        new_comma = cst.MaybeSentinel.DEFAULT

                    new_element = element.with_changes(key=new_key, value=new_value, comma=new_comma)
                    new_elements.append(new_element)

            return new_node.with_changes(
                lbrace=new_lbrace,
                rbrace=new_rbrace,
                elements=new_elements
            )

        # For other types (strings, numbers), return as-is
        return new_node

    def _preserve_indentation(self, old_node: Any, new_node: Any) -> Any:
        """
        Copy formatting from old node to new node, including multi-line structure.

        This recursively applies formatting from the original file to the parsed
        new value, preserving indentation, line breaks, and structure.
        """
        if isinstance(old_node, cst.List) and isinstance(new_node, cst.List):
            # Copy bracket whitespace
            new_node = new_node.with_changes(
                lbracket=old_node.lbracket,
                rbracket=old_node.rbracket
            )

            # If both have same number of elements, copy per-element formatting
            if len(old_node.elements) == len(new_node.elements):
                new_elements = []
                for old_el, new_el in zip(old_node.elements, new_node.elements):
                    # Recursively preserve formatting for the element value
                    new_value = self._preserve_indentation(old_el.value, new_el.value)
                    # Copy the comma and whitespace
                    new_element = new_el.with_changes(
                        value=new_value,
                        comma=old_el.comma
                    )
                    new_elements.append(new_element)
                new_node = new_node.with_changes(elements=new_elements)

            return new_node

        elif isinstance(old_node, cst.Dict) and isinstance(new_node, cst.Dict):
            # Copy brace whitespace
            new_node = new_node.with_changes(
                lbrace=old_node.lbrace,
                rbrace=old_node.rbrace
            )

            # If both have same number of elements, copy per-element formatting
            if len(old_node.elements) == len(new_node.elements):
                new_elements = []
                for old_el, new_el in zip(old_node.elements, new_node.elements):
                    # Recursively preserve formatting for key and value
                    new_key = self._preserve_indentation(old_el.key, new_el.key)
                    new_value = self._preserve_indentation(old_el.value, new_el.value)
                    # Copy the comma and whitespace
                    new_element = new_el.with_changes(
                        key=new_key,
                        value=new_value,
                        comma=old_el.comma
                    )
                    new_elements.append(new_element)
                new_node = new_node.with_changes(elements=new_elements)

            return new_node

        elif isinstance(old_node, cst.SimpleString) and isinstance(new_node, cst.SimpleString):
            # Preserve quote style
            return new_node

        elif isinstance(old_node, cst.Integer) and isinstance(new_node, cst.Integer):
            # Numbers don't have formatting to preserve
            return new_node

        return new_node

    def _replace_node_in_tree(self, tree: Any, old_node: Any, new_node: Any) -> Any:
        """Recursively replace old_node with new_node in tree."""
        if tree is old_node:
            return new_node

        if isinstance(tree, cst.Call):
            new_func = self._replace_node_in_tree(tree.func, old_node, new_node)
            return tree.with_changes(func=new_func)
        elif isinstance(tree, cst.Attribute):
            new_value = self._replace_node_in_tree(tree.value, old_node, new_node)
            return tree.with_changes(value=new_value)

        return tree


class ExpectationsFileProcessor:
    """Main processor that orchestrates the update workflow."""

    def __init__(self, device_type: str = "cuda", precision: Optional[int] = 8):
        """
        Initialize processor.

        Args:
            device_type: Device type (cuda, rocm, xpu, etc.)
            precision: Precision bits (8, 16, etc.) or None
        """
        self.device_type = device_type

        # Build target keys in priority order
        self.target_keys = []

        if precision is not None and isinstance(precision, tuple):
            self.target_keys.append((device_type, precision))
        elif precision is not None:
            self.target_keys.append((device_type, (precision, 6)))
            self.target_keys.append((device_type, precision))

        self.target_keys.extend([
            (device_type, None),
            (None, None),
            (device_type, "ANY"),
        ])

    def parse_all_test_contexts(self, captured_file: Path) -> List[Tuple[str, int, str, str]]:
        """
        Parse ALL test contexts from captured_info.txt.

        Returns:
            List of (test_file, assertion_line, variable_name, new_value_str) tuples
        """
        with open(captured_file, 'r') as f:
            content = f.read()

        # Split by the separator line
        sections = re.split(r'={70,}', content)

        test_contexts = []

        for section in sections:
            if not section.strip():
                continue

            # Try to extract test info from this section
            try:
                # Extract test file and line number
                context_match = re.search(r'/transformers/(.+?):(\d+)', section)
                if not context_match:
                    continue

                test_file = context_match.group(1)
                assertion_line = int(context_match.group(2))

                # Extract variable name from assertion
                variable_name = None

                # Try different assertion patterns
                patterns = [
                    r'assert_close\(.*?,\s+([a-zA-Z_]\w*)[\.\s,\[\]]',
                    r'assertEqual\(.*?,\s+([a-zA-Z_]\w*)[\.\s,\)\[]',
                    r'assertListEqual\(.*?,\s+([a-zA-Z_]\w*)[\.\s,\)\[]',
                    r'assertDictEqual\(.*?,\s+([a-zA-Z_]\w*)[\.\s,\)\[]',
                ]

                for pattern in patterns:
                    match = re.search(pattern, section, re.DOTALL)
                    if match:
                        # Make sure we didn't capture a number from array indexing
                        candidate = match.group(1).strip()
                        if not candidate[0].isdigit():
                            variable_name = candidate
                            break

                if not variable_name:
                    continue

                # Extract actual value
                actual_section = None
                for arg_name in ['actual', 'first', 'list1', 'd1']:
                    actual_section = re.search(
                        rf'argument name: `{arg_name}`.*?argument value:\s*\n\s*\n(.+?)(?:\n\s*\n-+|$)',
                        section, re.DOTALL
                    )
                    if actual_section:
                        break

                if not actual_section:
                    continue

                new_value_str = actual_section.group(1).strip()

                test_contexts.append((test_file, assertion_line, variable_name, new_value_str))

            except Exception as e:
                # Skip sections that can't be parsed
                print(f"  ⚠️  Skipping unparseable section: {e}")
                continue

        return test_contexts

    def find_expectations_variable(self, file_path: Path, variable_name: str, assertion_line: int) -> Optional[
        Tuple[str, int]]:
        """Find the variable assignment that needs updating."""
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Search backward from assertion line
        for lineno in range(assertion_line - 1, max(0, assertion_line - 50), -1):
            line = lines[lineno]

            # Pattern 1: Expectations pattern
            pattern = rf'^\s*{re.escape(variable_name)}\s*=.*?(\w+)\.get_expectation\(\)'
            match = re.search(pattern, line)

            if match:
                expectations_var = match.group(1)
                # Find the Expectations assignment
                for search_line in range(assertion_line - 1, max(0, assertion_line - 100), -1):
                    search_text = lines[search_line]
                    exp_pattern = rf'^\s*{re.escape(expectations_var)}\s*=\s*Expectations\('
                    if re.search(exp_pattern, search_text):
                        return expectations_var, search_line + 1
                return None

            # Pattern 2: Direct tensor pattern
            pattern = rf'^\s*{re.escape(variable_name)}\s*=\s*torch\.tensor\('
            if re.search(pattern, line):
                return variable_name, lineno + 1

            # Pattern 3: Plain list or dict pattern
            pattern = rf'^\s*{re.escape(variable_name)}\s*=\s*[\[\{{]'
            if re.search(pattern, line):
                return variable_name, lineno + 1

        return None

    def process(self, captured_file: Path, apply: bool = False) -> bool:
        """
        Main processing function - handles multiple test contexts.

        Args:
            captured_file: Path to captured_info.txt
            apply: If True, actually modify the file. If False, dry-run only.

        Returns:
            True if all updates successful, False otherwise
        """
        print("=" * 80)
        print("CST-based Expectations Updater (Multi-Context)")
        print("=" * 80)

        # Step 1: Parse ALL test contexts
        print("\n[1] Parsing captured_info.txt...")
        try:
            test_contexts = self.parse_all_test_contexts(captured_file)
            print(f"  Found {len(test_contexts)} test context(s)")
            for i, (test_file, line, var, _) in enumerate(test_contexts, 1):
                print(f"    {i}. {var} at line {line}")
        except Exception as e:
            print(f"❌ Error parsing captured_info.txt: {e}")
            return False

        if not test_contexts:
            print("❌ No test contexts found in captured_info.txt")
            return False

        # Step 2: Group by test file
        print("\n[2] Grouping updates by file...")
        updates_by_file: Dict[str, List[UpdateTask]] = {}

        for test_file, assertion_line, variable_name, new_value_str in test_contexts:
            file_path = Path(test_file)
            if not file_path.exists():
                print(f"  ⚠️  Test file not found: {file_path}")
                continue

            # Find the variable assignment
            result = self.find_expectations_variable(file_path, variable_name, assertion_line)
            if not result:
                print(f"  ⚠️  Could not find variable '{variable_name}' at line {assertion_line}")
                continue

            expectations_var, expectations_lineno = result

            # Create update task
            task = UpdateTask(
                test_file=test_file,
                assertion_line=assertion_line,
                variable_name=variable_name,
                new_value_str=new_value_str,
                expectations_var=expectations_var,
                expectations_lineno=expectations_lineno
            )

            if test_file not in updates_by_file:
                updates_by_file[test_file] = []
            updates_by_file[test_file].append(task)

        print(f"  Updates grouped into {len(updates_by_file)} file(s)")

        # Step 3: Process each file
        all_successful = True

        for file_path, tasks in updates_by_file.items():
            print(f"\n[3] Processing {file_path}...")
            print(f"  {len(tasks)} update(s) to apply")

            # Sort tasks by line number (descending) for safety
            tasks.sort(key=lambda t: t.expectations_lineno, reverse=True)

            # Parse file with CST
            try:
                with open(file_path, 'r') as f:
                    source_code = f.read()

                wrapper = cst.MetadataWrapper(cst.parse_module(source_code))
                print("  ✓ File parsed successfully")
            except Exception as e:
                print(f"  ❌ Error parsing file: {e}")
                all_successful = False
                continue

            # Transform the tree with all updates
            transformer = ExpectationsUpdater(
                update_tasks=tasks,
                target_keys=self.target_keys,
                dry_run=not apply
            )

            new_tree = wrapper.visit(transformer)

            # Report results
            print(f"\n  Results:")
            print(f"    ✓ Successful: {len(transformer.successful_updates)}")
            print(f"    ❌ Failed: {len(transformer.failed_updates)}")

            for task, pattern_type in transformer.successful_updates:
                print(f"      ✓ {task.variable_name} (line {task.expectations_lineno}) - {pattern_type}")

            for task, error in transformer.failed_updates:
                print(f"      ❌ {task.variable_name} (line {task.expectations_lineno}) - {error}")
                all_successful = False

            # Write back if applying
            if apply and transformer.successful_updates:
                try:
                    with open(file_path, 'w') as f:
                        f.write(new_tree.code)
                    print(f"\n  ✅ File updated: {file_path}")
                except Exception as e:
                    print(f"  ❌ Error writing file: {e}")
                    all_successful = False
            elif not apply:
                print(f"\n  ⚠️  DRY RUN - No changes made to {file_path}")

        # Final summary
        print("\n" + "=" * 80)
        if apply:
            print("✅ UPDATE COMPLETE!" if all_successful else "⚠️  UPDATE COMPLETED WITH ERRORS")
        else:
            print("ℹ️  DRY RUN COMPLETE - Use --apply to make changes")
        print("=" * 80)

        return all_successful


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Update Expectations dictionary values in test files using CST",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry-run (safe, shows what will change)
  %(prog)s captured_info.txt

  # Apply changes
  %(prog)s captured_info.txt --apply

  # Different device
  %(prog)s captured_info.txt --device rocm --apply

  # Different precision
  %(prog)s captured_info.txt --precision 16 --apply
        """
    )

    parser.add_argument(
        'captured_info',
        type=Path,
        help='Path to captured_info.txt file'
    )

    parser.add_argument(
        '--apply',
        action='store_true',
        help='Actually modify the file (default is dry-run)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device type (cuda, rocm, xpu, etc.) [default: cuda]'
    )

    parser.add_argument(
        '--precision',
        type=int,
        default=8,
        help='Precision bits (8, 16, or 0 for None) [default: 8]'
    )

    args = parser.parse_args()

    # Convert precision 0 to None
    precision = None if args.precision == 0 else args.precision

    # Create processor and run
    processor = ExpectationsFileProcessor(
        device_type=args.device,
        precision=precision
    )

    success = processor.process(args.captured_info, apply=args.apply)
    exit(0 if success else 1)


if __name__ == '__main__':
    main()