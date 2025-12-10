#!/usr/bin/env python3
"""
CST-based updater for Expectations pattern in test files.

This script uses LibCST (Concrete Syntax Tree) to preserve all formatting,
comments, and whitespace while making surgical updates to dictionary values.

Example pattern:
    expectations = Expectations({
        (None, None): [0.2166, -0.4368, 0.2191],
        ("cuda", 8): [0.2168, -0.4367, 0.2190],
    })
    expected_slice = torch.tensor(expectations.get_expectation()).to(torch_device)

The script will:
1. Parse the file using LibCST
2. Find the Expectations dictionary
3. Update the appropriate key based on device type
4. Preserve ALL formatting, comments, and whitespace automatically
"""

import re
import argparse
import libcst as cst
from libcst.metadata import PositionProvider
from pathlib import Path
from typing import Optional, Tuple, Any, Union


class ExpectationsUpdater(cst.CSTTransformer):
    """CST transformer that updates Expectations dictionary values."""

    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(
            self,
            expectations_var: str,
            expectations_lineno: int,
            target_keys: list,
            new_value_str: str,
            dry_run: bool = True
    ):
        """
        Initialize the transformer.

        Args:
            expectations_var: Name of the Expectations variable (e.g., "expectations")
            expectations_lineno: Line number of the specific Expectations assignment to update
            target_keys: List of keys to try in order of preference
            new_value_str: New value as a string (preserves formatting)
            dry_run: If True, don't actually modify, just report what would change
        """
        super().__init__()
        self.expectations_var = expectations_var
        self.expectations_lineno = expectations_lineno
        self.target_keys = target_keys
        self.new_value_str = new_value_str
        self.dry_run = dry_run

        # Track what we found
        self.found_expectations = False
        self.matched_key = None
        self.old_value = None
        self.updated = False

    def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.Assign:
        """
        Visit assignment statements looking for:
        1. expectations_var = Expectations(...)
        2. variable = torch.tensor([...])
        3. variable = torch.tensor([...]).to(device)  # with method chaining
        Only update the one that matches the expected line number.
        """
        # Check if this is an assignment we care about
        if not isinstance(updated_node.targets[0], cst.AssignTarget):
            return updated_node

        target = updated_node.targets[0].target
        if not isinstance(target, cst.Name):
            return updated_node

        if target.value != self.expectations_var:
            return updated_node

        # Get the line number using metadata
        try:
            pos = self.get_metadata(PositionProvider, original_node)
            node_lineno = pos.start.line
        except:
            node_lineno = -1

        # Only process if this is close to our target line
        if node_lineno != -1 and abs(node_lineno - self.expectations_lineno) > 3:
            return updated_node

        # Check what kind of assignment this is
        if not isinstance(updated_node.value, cst.Call):
            return updated_node

        # Pattern 1: Expectations(...)
        if isinstance(updated_node.value.func, cst.Name) and updated_node.value.func.value == "Expectations":
            return self._handle_expectations_pattern(original_node, updated_node, node_lineno)

        # Pattern 2: torch.tensor(...)
        if isinstance(updated_node.value.func, cst.Attribute):
            # Could be torch.tensor(...) or torch.tensor(...).to(...)
            func_attr = updated_node.value.func

            # Check if it's .to() chained after torch.tensor()
            if (isinstance(func_attr.value, cst.Call) and
                    isinstance(func_attr.value.func, cst.Attribute) and
                    isinstance(func_attr.value.func.value, cst.Name) and
                    func_attr.value.func.value.value == "torch" and
                    func_attr.value.func.attr.value == "tensor"):
                # This is torch.tensor(...).to(...)
                return self._handle_torch_tensor_with_to(original_node, updated_node, node_lineno)

            # Direct torch.tensor(...)
            if (isinstance(func_attr.value, cst.Name) and
                    func_attr.value.value == "torch" and
                    func_attr.attr.value == "tensor"):
                return self._handle_torch_tensor_pattern(original_node, updated_node, node_lineno)

        return updated_node

    def _handle_torch_tensor_with_to(self, original_node: cst.Assign, updated_node: cst.Assign,
                                     node_lineno: int) -> cst.Assign:
        """Handle the torch.tensor([...]).to(device) pattern."""
        self.found_expectations = True
        print(f"\n✓ Found torch.tensor().to() assignment for variable '{self.expectations_var}' at line {node_lineno}")

        # The outer call is .to(), the inner call is torch.tensor()
        to_call = updated_node.value
        tensor_call = to_call.func.value  # This is the torch.tensor() call

        # Get the tensor argument
        if not tensor_call.args:
            return updated_node

        arg = tensor_call.args[0]
        old_value = arg.value

        print(f"  Old value: {self._format_value(old_value)}")
        print(f"  New value: {self.new_value_str}")

        self.matched_key = "direct"
        self.old_value = self._format_value(old_value)

        if not self.dry_run:
            # Parse the new value and preserve indentation
            new_value_node = cst.parse_expression(self.new_value_str)
            new_value_node = self._preserve_indentation(old_value, new_value_node)

            # Replace the argument in the tensor call
            new_arg = arg.with_changes(value=new_value_node)
            new_tensor_call = tensor_call.with_changes(args=(new_arg,) + tensor_call.args[1:])

            # Update the .to() call's func.value with the new tensor call
            new_func = to_call.func.with_changes(value=new_tensor_call)
            new_to_call = to_call.with_changes(func=new_func)

            self.updated = True
            return updated_node.with_changes(value=new_to_call)
        else:
            print("  ⚠️  DRY RUN - Not actually modifying")

        return updated_node

    def _handle_expectations_pattern(self, original_node: cst.Assign, updated_node: cst.Assign,
                                     node_lineno: int) -> cst.Assign:
        """Handle the Expectations({...}) pattern."""
        self.found_expectations = True
        print(f"\n✓ Found Expectations assignment for variable '{self.expectations_var}' at line {node_lineno}")

        # The argument should be a dictionary
        if not updated_node.value.args:
            return updated_node

        arg = updated_node.value.args[0]
        if not isinstance(arg.value, cst.Dict):
            return updated_node

        # Update the dictionary
        dict_node = arg.value
        updated_dict = self._update_dict(dict_node)

        if updated_dict is not dict_node and not self.dry_run:
            new_arg = arg.with_changes(value=updated_dict)
            new_call = updated_node.value.with_changes(args=[new_arg])
            return updated_node.with_changes(value=new_call)

        return updated_node

    def _handle_torch_tensor_pattern(self, original_node: cst.Assign, updated_node: cst.Assign,
                                     node_lineno: int) -> cst.Assign:
        """Handle the torch.tensor([...]) pattern."""
        self.found_expectations = True  # Reuse this flag to mean "found target"
        print(f"\n✓ Found torch.tensor assignment for variable '{self.expectations_var}' at line {node_lineno}")

        # The argument should be a list
        if not updated_node.value.args:
            return updated_node

        arg = updated_node.value.args[0]
        old_value = arg.value

        print(f"  Old value: {self._format_value(old_value)}")
        print(f"  New value: {self.new_value_str}")

        # Mark this as found (use matched_key to signal success even though there's no key)
        self.matched_key = "direct"
        self.old_value = self._format_value(old_value)

        if not self.dry_run:
            # Parse the new value and preserve indentation
            new_value_node = cst.parse_expression(self.new_value_str)
            new_value_node = self._preserve_indentation(old_value, new_value_node)

            # Replace the argument
            new_arg = arg.with_changes(value=new_value_node)
            new_call = updated_node.value.with_changes(args=(new_arg,) + updated_node.value.args[1:])
            self.updated = True
            return updated_node.with_changes(value=new_call)
        else:
            print("  ⚠️  DRY RUN - Not actually modifying")

        return updated_node

    def _update_dict(self, dict_node: cst.Dict) -> cst.Dict:
        """
        Update the dictionary by finding the target key and replacing its value.
        Only updates the BEST matching key (first in priority order).
        """
        # First pass: find all matching keys and their priorities
        matches = []
        for i, element in enumerate(dict_node.elements):
            if not isinstance(element, cst.DictElement):
                continue

            key_match = self._match_key(element.key)
            if key_match:
                # Find the priority (lower index = higher priority)
                priority = self._get_key_priority(key_match)
                matches.append((priority, i, element, key_match))
                print(f"✓ Found matching key: {key_match} (priority: {priority})")
                print(f"  Old value: {self._format_value(element.value)}")

        if not matches:
            return dict_node

        # Sort by priority (lower is better) and take the best match
        matches.sort(key=lambda x: x[0])
        best_priority, best_idx, best_element, best_key = matches[0]

        self.matched_key = best_key
        self.old_value = self._format_value(best_element.value)

        if len(matches) > 1:
            print(f"\n  → Selecting best match: {best_key} (priority: {best_priority})")
        print(f"  New value: {self.new_value_str}")

        # Second pass: rebuild elements, only updating the best match
        new_elements = []
        for i, element in enumerate(dict_node.elements):
            if i == best_idx and not self.dry_run:
                # Parse the new value as CST
                new_value_node = cst.parse_expression(self.new_value_str)

                # Preserve indentation from the old value
                old_value = best_element.value
                new_value_node = self._preserve_indentation(old_value, new_value_node)

                # Replace the value while keeping the key
                new_element = element.with_changes(value=new_value_node)
                new_elements.append(new_element)
                self.updated = True
            else:
                new_elements.append(element)

        if self.dry_run:
            print("  ⚠️  DRY RUN - Not actually modifying")

        return dict_node.with_changes(elements=new_elements)

    def _preserve_indentation(self, old_node: cst.BaseExpression, new_node: cst.BaseExpression) -> cst.BaseExpression:
        """
        Preserve exact whitespace structure from old node to new node.
        Currently just copies the structure directly.

        TODO: In future, implement smart re-indentation based on captured_info.txt format.
        """
        if not isinstance(old_node, cst.List) or not isinstance(new_node, cst.List):
            return new_node

        # Copy the bracket whitespace
        new_node = new_node.with_changes(
            lbracket=old_node.lbracket,
            rbracket=old_node.rbracket
        )

        # Copy whitespace from old elements to new elements
        new_elements = []
        for i, new_elem in enumerate(new_node.elements):
            if not isinstance(new_elem, cst.Element):
                new_elements.append(new_elem)
                continue

            if i < len(old_node.elements):
                old_elem = old_node.elements[i]
                if isinstance(old_elem, cst.Element):
                    new_elem = new_elem.with_changes(comma=old_elem.comma)

            new_elements.append(new_elem)

        return new_node.with_changes(elements=new_elements)

    def _match_key(self, key_node: cst.BaseExpression) -> Optional[tuple]:
        """
        Check if a key node matches any of our target keys.

        Returns the matched target key tuple if found, None otherwise.
        """
        if not isinstance(key_node, cst.Tuple):
            return None

        # Extract the key tuple values
        if len(key_node.elements) != 2:
            return None

        # Get first element (device)
        first_elem = key_node.elements[0]
        if isinstance(first_elem, cst.Element):
            first_val = first_elem.value
        else:
            return None

        # Get second element (precision)
        second_elem = key_node.elements[1]
        if isinstance(second_elem, cst.Element):
            second_val = second_elem.value
        else:
            return None

        # Parse the values
        device = self._parse_key_value(first_val)
        precision = self._parse_key_value(second_val)

        actual_key = (device, precision)

        # Check against target keys
        for target_key in self.target_keys:
            if target_key[1] == "ANY":
                # Lenient match - just check device
                if device == target_key[0]:
                    return actual_key
            elif actual_key == target_key:
                return actual_key

        return None

    def _get_key_priority(self, key: tuple) -> int:
        """
        Get the priority of a key based on the target_keys list.
        Lower number = higher priority.
        """
        for i, target_key in enumerate(self.target_keys):
            if target_key[1] == "ANY":
                # Lenient match
                if key[0] == target_key[0]:
                    return i
            elif key == target_key:
                return i
        return 999  # Unknown key (shouldn't happen)

    def _parse_key_value(self, node: cst.BaseExpression) -> Union[str, int, None, tuple]:
        """
        Parse a key value node to extract its Python value.
        """
        if isinstance(node, cst.Name) and node.value == "None":
            return None
        elif isinstance(node, cst.SimpleString):
            # Remove quotes
            return node.value.strip('"\'')
        elif isinstance(node, cst.Integer):
            return int(node.value)
        elif isinstance(node, cst.Tuple):
            # Handle tuple precision like (8, 6)
            values = []
            for elem in node.elements:
                if isinstance(elem, cst.Element):
                    if isinstance(elem.value, cst.Integer):
                        values.append(int(elem.value.value))
            return tuple(values) if values else None
        else:
            return None

    def _format_value(self, value_node: cst.BaseExpression) -> str:
        """Format a value node for display."""
        # Create a minimal module wrapper to get the code representation
        module = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=value_node)])])
        code = module.code
        # Extract just the expression part (remove trailing newline)
        return code.strip()


class ExpectationsFileProcessor:
    """Main processor for updating expectations in test files."""

    def __init__(self, device_type: str = "cuda", precision: Optional[int] = 8):
        """
        Initialize the processor.

        Args:
            device_type: Device type (cuda, rocm, xpu, etc.)
            precision: Precision bits (8, 16, None for default)
        """
        self.device_type = device_type
        self.precision = precision
        self.target_keys = self._build_target_keys()

    def _build_target_keys(self) -> list:
        """
        Build list of keys to try, in order of preference.

        For tuple precisions like (8, 6), we try multiple patterns:
        1. (device, (precision, 6)) - e.g., ("cuda", (8, 6))
        2. (device, precision) - e.g., ("cuda", 8)
        3. (device, None) - e.g., ("cuda", None)
        4. (None, None) - fallback
        5. (device, ANY) - lenient match for any precision

        Returns:
            List of tuples to try in order of preference
        """
        keys = []

        # First try: tuple precision (device, (precision, 6))
        if self.precision is not None:
            keys.append((self.device_type, (self.precision, 6)))

        # Second try: simple precision (device, precision)
        if self.precision is not None:
            keys.append((self.device_type, self.precision))

        # Third try: device with any precision (device, None)
        keys.append((self.device_type, None))

        # Fourth try: fallback default (None, None)
        keys.append((None, None))

        # Fifth try: ANY key with matching device (handles other tuple precisions)
        keys.append((self.device_type, "ANY"))

        return keys

    def parse_captured_info(self, captured_file: Path) -> Tuple[str, int, str, str]:
        """
        Parse captured_info.txt to extract test information and new value.

        Returns:
            Tuple of (test_file_path, assertion_line, variable_name, new_value_str)
        """
        with open(captured_file, 'r') as f:
            content = f.read()

        # Extract test file path and line number
        test_context_match = re.search(r'test context:\s*([^:]+):(\d+)', content)
        if not test_context_match:
            raise ValueError("Could not find test context in captured_info.txt")

        test_file_raw = test_context_match.group(1).strip()
        assertion_line = int(test_context_match.group(2))

        # Normalize path (remove /transformers/ prefix if present)
        test_file = test_file_raw.split('/transformers/', 1)[-1] if '/transformers/' in test_file_raw else test_file_raw

        # Extract the variable name from the assertion
        # Pattern 1: assert_close(actual, expected_variable, ...)
        # Pattern 2: self.assertEqual(actual, expected_variable) or assertEqual(actual, expected_variable, ...)
        assertion_match = re.search(r'assert_close\((.*?),\s*(\w+)\s*,', content, re.DOTALL)
        if not assertion_match:
            # Try assertEqual pattern - may have comma or closing paren after variable
            assertion_match = re.search(r'assertEqual\(\s*(.*?),\s*(\w+)\s*[,\)]', content, re.DOTALL)

        if not assertion_match:
            raise ValueError("Could not find assertion with variable name")

        variable_name = assertion_match.group(2).strip()

        # Extract the actual value (new value to use)
        actual_section = re.search(
            r'argument name: `actual`.*?argument value:\s*\n\s*\n(.+?)(?:\n\s*\n-+|========)',
            content, re.DOTALL
        )
        if not actual_section:
            actual_section = re.search(
                r'argument name: `first`.*?argument value:\s*\n\s*\n(.+?)(?:\n\s*\n-+|========)',
                content, re.DOTALL
            )

        if not actual_section:
            raise ValueError("Could not find actual value in captured_info.txt")

        new_value_str = actual_section.group(1).strip()

        return test_file, assertion_line, variable_name, new_value_str

    def find_expectations_variable(self, file_path: Path, variable_name: str, assertion_line: int) -> Optional[
        Tuple[str, int]]:
        """
        Find the variable assignment that needs updating.

        Handles two patterns:
        1. Expectations pattern: variable = torch.tensor(expectations.get_expectation())
        2. Direct tensor pattern: variable = torch.tensor([...])

        Returns:
            Tuple of (variable_or_expectations_name, line_number) or None
            - For Expectations: returns ("expectations", line_of_Expectations_dict)
            - For direct tensor: returns (variable_name, line_of_assignment)
        """
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Search backward from assertion line to find the variable assignment
        for lineno in range(assertion_line - 1, max(0, assertion_line - 50), -1):
            line = lines[lineno]

            # Pattern 1: Expectations pattern
            # Look for: variable_name = torch.tensor(something.get_expectation())
            pattern = rf'^\s*{re.escape(variable_name)}\s*=.*?(\w+)\.get_expectation\(\)'
            match = re.search(pattern, line)

            if match:
                expectations_var = match.group(1)
                print(f"✓ Found Expectations pattern at line {lineno + 1}")
                print(f"  Variable: {variable_name}")
                print(f"  Expectations variable: {expectations_var}")

                # Now find the Expectations assignment
                for search_line in range(assertion_line - 1, max(0, assertion_line - 100), -1):
                    search_text = lines[search_line]
                    exp_pattern = rf'^\s*{re.escape(expectations_var)}\s*=\s*Expectations\('
                    if re.search(exp_pattern, search_text):
                        expectations_lineno = search_line + 1
                        print(f"  Expectations assignment at line: {expectations_lineno}")
                        return expectations_var, expectations_lineno

                return None

            # Pattern 2: Direct tensor pattern
            # Look for: variable_name = torch.tensor([...])
            pattern = rf'^\s*{re.escape(variable_name)}\s*=\s*torch\.tensor\('
            if re.search(pattern, line):
                print(f"✓ Found direct tensor pattern at line {lineno + 1}")
                print(f"  Variable: {variable_name}")
                # For direct pattern, we return the variable name and line number
                # This signals to use direct replacement mode
                return variable_name, lineno + 1

        return None

    def process(self, captured_file: Path, apply: bool = False) -> bool:
        """
        Main processing function.

        Args:
            captured_file: Path to captured_info.txt
            apply: If True, actually modify the file. If False, dry-run only.

        Returns:
            True if successful, False otherwise
        """
        print("=" * 80)
        print("CST-based Expectations Updater")
        print("=" * 80)

        # Step 1: Parse captured_info.txt
        print("\n[1] Parsing captured_info.txt...")
        try:
            test_file, assertion_line, variable_name, new_value_str = self.parse_captured_info(captured_file)
            print(f"  Test file: {test_file}")
            print(f"  Assertion line: {assertion_line}")
            print(f"  Variable name: {variable_name}")
            print(f"  New value: {new_value_str}")
        except Exception as e:
            print(f"❌ Error parsing captured_info.txt: {e}")
            return False

        # Step 2: Find Expectations variable and its line number
        print("\n[2] Finding Expectations variable...")
        file_path = Path(test_file)
        if not file_path.exists():
            print(f"❌ Test file not found: {file_path}")
            return False

        result = self.find_expectations_variable(file_path, variable_name, assertion_line)
        if not result:
            print(f"❌ Could not find Expectations variable for '{variable_name}'")
            return False

        expectations_var, expectations_lineno = result

        # Step 3: Parse the file with CST
        print("\n[3] Parsing file with LibCST...")
        try:
            with open(file_path, 'r') as f:
                source_code = f.read()

            wrapper = cst.MetadataWrapper(cst.parse_module(source_code))
            print("✓ File parsed successfully")
        except Exception as e:
            print(f"❌ Error parsing file: {e}")
            return False

        # Step 4: Transform the tree
        print(f"\n[4] {'Updating' if apply else 'Analyzing'} Expectations dictionary...")
        print(f"  Target line: {expectations_lineno}")
        print(f"  Target keys (in order): {self.target_keys}")

        transformer = ExpectationsUpdater(
            expectations_var=expectations_var,
            expectations_lineno=expectations_lineno,
            target_keys=self.target_keys,
            new_value_str=new_value_str,
            dry_run=not apply
        )

        new_tree = wrapper.visit(transformer)

        # Step 5: Report results
        print("\n[5] Results:")
        if not transformer.found_expectations:
            print(f"❌ Could not find target assignment for '{expectations_var}'")
            return False

        # For direct tensor pattern, matched_key will be "direct"
        # For Expectations pattern, matched_key will be the actual key tuple
        if not transformer.matched_key:
            print(f"❌ Could not find matching key in dictionary")
            print(f"  Tried: {self.target_keys}")
            return False

        if apply:
            # Write the modified code back
            with open(file_path, 'w') as f:
                f.write(new_tree.code)
            print(f"✅ Successfully updated {file_path}")
            if transformer.matched_key == "direct":
                print(f"  Pattern: Direct torch.tensor assignment")
            else:
                print(f"  Key: {transformer.matched_key}")
            print(f"  Old value: {transformer.old_value}")
            print(f"  New value: {new_value_str}")
        else:
            print("  ⚠️  DRY RUN - No changes made")
            if transformer.matched_key == "direct":
                print(f"  Would update: Direct torch.tensor assignment")
            else:
                print(f"  Would update key: {transformer.matched_key}")

        print("\n" + "=" * 80)
        if apply:
            print("✅ UPDATE COMPLETE!")
        else:
            print("ℹ️  DRY RUN COMPLETE - Use --apply to make changes")
        print("=" * 80)

        return True


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