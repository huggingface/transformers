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

        dict_arg = pattern_node.args[0].value
        if not isinstance(dict_arg, cst.Dict):
            self.failed_updates.append((task, "Expectations argument is not a dictionary"))
            return updated_node

        # Find matching key and update
        new_dict = self._update_dict(dict_arg, task)
        if new_dict is None:
            self.failed_updates.append((task, "Could not find matching key"))
            return updated_node

        # Reconstruct the tree with updated dict
        new_call = pattern_node.with_changes(
            args=[cst.Arg(value=new_dict)]
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
            # Parse the new value
            new_value_node = cst.parse_expression(task.new_value_str)

            # Preserve indentation
            if isinstance(pattern_node, cst.Call):
                # For torch.tensor, update the argument
                if pattern_node.args and len(pattern_node.args) > 0:
                    old_arg = pattern_node.args[0].value
                    new_arg = self._preserve_indentation(old_arg, new_value_node)
                    new_call = pattern_node.with_changes(
                        args=[cst.Arg(value=new_arg)] + list(pattern_node.args[1:])
                    )
                    new_full_value = self._replace_node_in_tree(updated_node.value, pattern_node, new_call)
                else:
                    self.failed_updates.append((task, "No arguments in torch.tensor call"))
                    return updated_node
            else:
                # For plain list/dict, replace directly
                new_full_value = self._preserve_indentation(pattern_node, new_value_node)

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
        # Parse new value
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

        # Preserve indentation from old value
        old_value = best_element.value
        new_value_preserved = self._preserve_indentation(old_value, new_value_node)

        # Create new element with updated value
        new_element = best_element.with_changes(value=new_value_preserved)

        # Replace in dictionary
        new_elements = []
        for element in dict_node.elements:
            if element is best_element:
                new_elements.append(new_element)
            else:
                new_elements.append(element)

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

    def _preserve_indentation(self, old_node: Any, new_node: Any) -> Any:
        """Copy whitespace from old node to new node."""
        if isinstance(old_node, cst.List) and isinstance(new_node, cst.List):
            return new_node.with_changes(
                lbracket=old_node.lbracket,
                rbracket=old_node.rbracket
            )
        elif isinstance(old_node, cst.Dict) and isinstance(new_node, cst.Dict):
            return new_node.with_changes(
                lbrace=old_node.lbrace,
                rbrace=old_node.rbrace
            )
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