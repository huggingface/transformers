#!/usr/bin/env python3
"""
Syntax validation for the Mixtral auxiliary loss fix.
This checks that the modified code is syntactically correct without requiring full imports.
"""

import ast
import sys
import os

def validate_python_syntax(file_path):
    """Validate that a Python file has correct syntax."""
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        
        # Parse the source code
        ast.parse(source)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def main():
    """Main validation function."""
    file_path = "/tmp/oss-transformers/src/transformers/models/mixtral/modular_mixtral.py"
    
    print("Validating syntax of modified file...")
    print(f"File: {file_path}")
    
    is_valid, error = validate_python_syntax(file_path)
    
    if is_valid:
        print("✅ Syntax validation PASSED")
        
        # Also check that our fix is present in the file
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for our fix
        if "self.router_aux_loss_coef != 0" in content:
            print("✅ Fix detected: router_aux_loss_coef condition found")
        else:
            print("❌ Fix not found: router_aux_loss_coef condition missing")
            return False
            
        # Check that the old problematic condition is gone
        if "if output_router_logits:" not in content.split("aux_loss = None")[1].split("return MoeCausalLMOutputWithPast")[0]:
            print("✅ Old condition removed: aux_loss no longer depends only on output_router_logits")
        else:
            print("❌ Old condition still present: aux_loss still depends on output_router_logits")
            return False
            
        return True
    else:
        print(f"❌ Syntax validation FAILED: {error}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)