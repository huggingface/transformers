#!/usr/bin/env python3
"""
AST-based updater for Expectations pattern in test files.

This script handles the dynamic lookup pattern where expected values are stored
in an Expectations dictionary and retrieved at runtime via get_expectation().

Example pattern:
    expectations = Expectations({
        (None, None): [0.2166, -0.4368, 0.2191],
        ("cuda", 8): [0.2168, -0.4367, 0.2190],
    })
    expected_slice = torch.tensor(expectations.get_expectation()).to(torch_device)

The script will:
1. Parse the file using Python's AST
2. Find the Expectations dictionary
3. Update the appropriate key based on device type
4. Preserve formatting and comments
"""

import ast
import re
import argparse
from pathlib import Path
from typing import Optional, Tuple, Any


class ExpectationsUpdater:
    """Updates Expectations dictionary values in test files."""

    def __init__(self, device_type: str = "cuda", precision: Optional[int] = 8):
        """
        Initialize the updater.

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
        # This is marked as "lenient" for the search function
        keys.append((self.device_type, "ANY"))

        return keys

    def parse_captured_info(self, captured_file: Path) -> Tuple[str, int, str, Any]:
        """
        Parse captured_info.txt to extract test information and new value.

        Returns:
            Tuple of (test_file_path, assertion_line, variable_name, new_value)
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
        # Pattern 2: self.assertEqual(actual, expected_variable)
        # The actual argument could have brackets like outputs.logits[0, :3]
        # So we need to match more carefully
        assertion_match = re.search(r'assert_close\((.*?),\s*(\w+)\s*,', content, re.DOTALL)
        if not assertion_match:
            # Try assertEqual pattern
            assertion_match = re.search(r'assertEqual\((.*?),\s*(\w+)\s*\)', content, re.DOTALL)

        if not assertion_match:
            raise ValueError("Could not find assertion with variable name (tried assert_close and assertEqual)")

        # The second argument is usually the expected value variable
        variable_name = assertion_match.group(2).strip()

        # Extract the actual value (new value to use)
        # Try different argument names: 'actual', 'first', 'list1'
        actual_section = re.search(r'argument name: `actual`.*?argument value:\s*\n\s*\n(.+?)(?:\n\s*\n-+|========)',
                                   content, re.DOTALL)
        if not actual_section:
            actual_section = re.search(r'argument name: `first`.*?argument value:\s*\n\s*\n(.+?)(?:\n\s*\n-+|========)',
                                       content, re.DOTALL)
        if not actual_section:
            actual_section = re.search(r'argument name: `list1`.*?argument value:\s*\n\s*\n(.+?)(?:\n\s*\n-+|========)',
                                       content, re.DOTALL)

        if not actual_section:
            raise ValueError("Could not find actual value in captured_info.txt (tried: actual, first, list1)")

        new_value_str = actual_section.group(1).strip()

        # Keep the string representation to preserve formatting (e.g., 0.2180 not 0.218)
        # But also parse it to verify it's valid
        try:
            # Verify it's valid Python literal
            ast.literal_eval(new_value_str)
            # Use the string representation to preserve formatting
            new_value = new_value_str
        except:
            # If it fails, keep as string anyway
            new_value = new_value_str

        return test_file, assertion_line, variable_name, new_value

    def find_expectations_assignment(self, file_path: Path, variable_name: str, assertion_line: int) -> Optional[
        Tuple[int, str]]:
        """
        Find the Expectations() assignment that feeds into the variable.

        This searches backward from the assertion line to find:
        1. The variable assignment (e.g., expected_slice = torch.tensor(...))
        2. Check if it uses .get_expectation()
        3. Find the expectations = Expectations({...}) assignment

        Returns:
            Tuple of (line_number, expectations_variable_name) or None
        """
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Search backward from assertion line
        expectations_var = None

        # First, find the variable assignment
        for lineno in range(assertion_line - 1, max(0, assertion_line - 50), -1):
            line = lines[lineno]

            # Look for: variable_name = torch.tensor(something.get_expectation())
            pattern = rf'^\s*{re.escape(variable_name)}\s*=.*?(\w+)\.get_expectation\(\)'
            match = re.search(pattern, line)

            if match:
                expectations_var = match.group(1)
                print(f"Found variable assignment at line {lineno + 1}: {line.strip()}")
                print(f"Expectations variable name: {expectations_var}")
                break

        if not expectations_var:
            return None

        # Now find the expectations variable assignment
        for lineno in range(assertion_line - 1, max(0, assertion_line - 100), -1):
            line = lines[lineno]

            # Look for: expectations_var = Expectations({
            pattern = rf'^\s*{re.escape(expectations_var)}\s*=\s*Expectations\('
            if re.search(pattern, line):
                print(f"Found Expectations assignment at line {lineno + 1}: {line.strip()}")
                return lineno + 1, expectations_var  # Convert to 1-indexed

        return None

    def find_dict_key_line(self, lines: list, start_line: int, target_key: tuple) -> Optional[int]:
        """
        Find the line number of a specific dictionary key in the Expectations dict.

        Args:
            lines: List of file lines
            start_line: Line where Expectations({ starts (1-indexed)
            target_key: The key tuple to find, e.g., ("cuda", 8)

        Returns:
            1-indexed line number where the key is found, or None
        """
        # Convert target key to string representations we might find
        # Could be: ("cuda", 8) or ('cuda', 8) or (None, None) or ("cuda", (8, 6))
        key_patterns = []

        if target_key == (None, None):
            # Match: (None, None)
            key_patterns.append(r'\(\s*None\s*,\s*None\s*\)')
        elif target_key[1] == "ANY":
            # Lenient match: ANY key starting with this device
            # Matches: ("cuda", 8) or ("cuda", None) or ("cuda", (8, 0)) etc.
            device = target_key[0]
            key_patterns.append(rf'\(\s*["\']?{re.escape(device)}["\']?\s*,')
            print(f"  Using lenient match for device '{device}' (will match any precision)")
        elif target_key[1] is None:
            # Match: ("cuda", None) or ('cuda', None)
            device = target_key[0]
            key_patterns.append(rf'\(\s*["\']?{re.escape(device)}["\']?\s*,\s*None\s*\)')
        elif isinstance(target_key[1], tuple):
            # Match tuple precision: ("cuda", (8, 6)) or ('cuda', (8, 6))
            device = target_key[0]
            precision_tuple = target_key[1]
            # Build pattern for tuple: (8, 6)
            tuple_pattern = r'\(\s*' + r'\s*,\s*'.join(str(p) for p in precision_tuple) + r'\s*\)'
            key_patterns.append(rf'\(\s*["\']?{re.escape(device)}["\']?\s*,\s*{tuple_pattern}\s*\)')
        else:
            # Match: ("cuda", 8) or ('cuda', 8)
            device, precision = target_key
            key_patterns.append(rf'\(\s*["\']?{re.escape(device)}["\']?\s*,\s*{precision}\s*\)')

        # Search forward from start_line, but stop at the end of the dictionary
        # Look for closing patterns:
        # - }  ) on same line
        # - } on one line, ) on next line
        dict_closed = False
        for i in range(start_line - 1, min(len(lines), start_line + 100)):
            line = lines[i]

            # Check if we've reached the closing of the dict
            if re.search(r'^\s*\}', line):
                dict_closed = True
                print(f"  Found closing }} of dict at line {i + 1}")
                # Continue to find the ) but don't go beyond it

            # If dict is closed and we see ), we're done
            if dict_closed and re.search(r'^\s*\)', line):
                print(f"  Reached end of Expectations at line {i + 1}")
                break

            # Check if this line matches our target key (only if dict not closed yet)
            if not dict_closed:
                for pattern in key_patterns:
                    if re.search(pattern, line):
                        print(f"Found key {target_key} at line {i + 1}: {line.strip()}")
                        return i + 1  # Return 1-indexed

        return None

    def update_dict_value(self, file_path: Path, key_line: int, new_value: Any) -> bool:
        """
        Update the dictionary value at the given line.

        Args:
            file_path: Path to the test file
            key_line: 1-indexed line number where the key is located
            new_value: New value to set (e.g., [0.2180, -0.4355, 0.2198])

        Returns:
            True if successful, False otherwise
        """
        with open(file_path, 'r') as f:
            lines = f.readlines()

        line_idx = key_line - 1  # Convert to 0-indexed
        original_line = lines[line_idx]

        print(f"\nOriginal line {key_line}:")
        print(f"  {original_line.rstrip()}")

        # Parse the line to find the value part
        # Pattern: ("cuda", 8): [old_value],
        # We want to replace [old_value] with [new_value]

        # Find the colon that separates key from value
        colon_idx = original_line.find(':')
        if colon_idx == -1:
            print("ERROR: Could not find ':' in line")
            return False

        # Everything before colon (including leading spaces)
        before_colon = original_line[:colon_idx + 1]

        # Format the new value
        if isinstance(new_value, str):
            # If new_value is already a string (from captured_info), use it directly
            # This preserves formatting like [0.2180, -0.4355, 0.2198]
            formatted_value = new_value
        elif isinstance(new_value, list):
            # Format as a nice list
            formatted_value = '[' + ', '.join(str(v) for v in new_value) + ']'
        else:
            formatted_value = str(new_value)

        # Check if original line has trailing comma
        has_trailing_comma = original_line.rstrip().endswith(',')
        trailing = ',' if has_trailing_comma else ''

        # Reconstruct the line
        new_line = f"{before_colon} {formatted_value}{trailing}\n"

        print(f"New line {key_line}:")
        print(f"  {new_line.rstrip()}")

        # Update the line
        lines[line_idx] = new_line

        # Write back
        with open(file_path, 'w') as f:
            f.writelines(lines)

        print(f"\n✅ Successfully updated line {key_line}")
        return True

    def process(self, captured_file: Path, apply: bool = False) -> bool:
        """
        Main processing function.

        Args:
            captured_file: Path to captured_info.txt
            apply: If True, actually modify the file. If False, dry-run.

        Returns:
            True if successful, False otherwise
        """
        print("=" * 80)
        print("AST-based Expectations Updater")
        print("=" * 80)

        # Step 1: Parse captured info
        print("\n[1] Parsing captured_info.txt...")
        test_file, assertion_line, variable_name, new_value = self.parse_captured_info(captured_file)

        print(f"  Test file: {test_file}")
        print(f"  Assertion line: {assertion_line}")
        print(f"  Variable name: {variable_name}")
        print(f"  New value: {new_value}")

        # Step 2: Find Expectations assignment
        print(f"\n[2] Finding Expectations assignment...")
        test_file_path = Path(test_file)
        result = self.find_expectations_assignment(test_file_path, variable_name, assertion_line)

        if not result:
            print(f"  ❌ Could not find Expectations assignment for {variable_name}")
            return False

        expectations_line, expectations_var = result

        # Step 3: Find the correct dictionary key
        print(f"\n[3] Finding dictionary key...")
        print(f"  Target keys (in order): {self.target_keys}")

        with open(test_file_path, 'r') as f:
            lines = f.readlines()

        found_key = None
        found_line = None

        for key in self.target_keys:
            print(f"  Trying key: {key}")
            line_num = self.find_dict_key_line(lines, expectations_line, key)
            if line_num:
                found_key = key
                found_line = line_num
                print(f"  ✅ Found matching key: {key} at line {line_num}")
                break

        if not found_line:
            print(f"  ❌ Could not find any matching dictionary key")
            print(f"  Tried: {self.target_keys}")
            return False

        # Step 4: Update the value
        print(f"\n[4] Updating dictionary value...")

        if not apply:
            print("  ⚠️  DRY RUN - Not actually modifying file")
            print(f"  Would update line {found_line} with value: {new_value}")
            return True

        success = self.update_dict_value(test_file_path, found_line, new_value)

        if success:
            print("\n" + "=" * 80)
            print("✅ UPDATE COMPLETE!")
            print("=" * 80)

        return success


def main():
    parser = argparse.ArgumentParser(
        description="Update Expectations dictionary values in test files using AST parsing"
    )
    parser.add_argument(
        "captured_info",
        type=Path,
        help="Path to captured_info.txt file"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually modify the file (default is dry-run)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device type (cuda, rocm, xpu, etc.). Default: cuda"
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=8,
        help="Precision bits (8, 16, etc.). Use 0 for None. Default: 8"
    )

    args = parser.parse_args()

    # Convert precision 0 to None
    precision = None if args.precision == 0 else args.precision

    # Create updater
    updater = ExpectationsUpdater(device_type=args.device, precision=precision)

    # Process
    success = updater.process(args.captured_info, apply=args.apply)

    if not success:
        exit(1)


if __name__ == "__main__":
    main()