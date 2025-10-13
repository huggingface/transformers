#!/usr/bin/env python3
"""
Verification script for P-002: Missing RAG examples directory
Tests that the RAG examples directory and README exist and are accessible.
"""
import os
import sys

def verify_p002():
    """Verify that P-002 (missing RAG examples) is resolved."""
    
    print("🔍 Verifying P-002: Missing RAG examples directory")
    print("=" * 60)
    
    # Test 1: Check directory exists
    rag_dir = "examples/rag"
    readme_path = "examples/rag/README.md"
    
    print(f"1. Checking if directory exists: {rag_dir}")
    if os.path.exists(rag_dir):
        print("   ✅ Directory exists")
    else:
        print("   ❌ Directory missing")
        return False
    
    # Test 2: Check README exists
    print(f"2. Checking if README exists: {readme_path}")
    if os.path.exists(readme_path):
        print("   ✅ README.md exists")
    else:
        print("   ❌ README.md missing")
        return False
    
    # Test 3: Check README content
    print("3. Checking README content...")
    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for required elements
        checks = [
            ("RAG Examples", "Contains title"),
            ("Quick Start", "Has quick start section"),
            ("facebook/rag-", "References actual models"),
            ("RagSequenceForGeneration", "Contains working code"),
            ("arxiv.org/pdf/2005.11401", "Links to original paper")
        ]
        
        for check_text, description in checks:
            if check_text in content:
                print(f"   ✅ {description}")
            else:
                print(f"   ❌ Missing: {description}")
                return False
                
    except Exception as e:
        print(f"   ❌ Error reading README: {e}")
        return False
    
    # Test 4: Check file size (should have substantial content)
    size = os.path.getsize(readme_path)
    print(f"4. README file size: {size} bytes")
    if size > 1000:  # Should be at least 1KB of content
        print("   ✅ README has substantial content")
    else:
        print("   ❌ README seems too small")
        return False
    
    print("\n🎉 P-002 VERIFICATION: PASSED")
    print("The RAG examples directory issue has been resolved!")
    return True

if __name__ == "__main__":
    success = verify_p002()
    sys.exit(0 if success else 1)