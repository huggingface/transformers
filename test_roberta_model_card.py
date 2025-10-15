#!/usr/bin/env python3
"""
Test script to validate the RoBERTa model card update.
Checks for required sections, code syntax, and formatting.
"""

import re
import ast
import sys
from pathlib import Path

def test_model_card():
    """Test the RoBERTa model card for compliance with issue #36979."""
    
    model_card_path = Path("docs/source/en/model_doc/roberta.md")
    
    if not model_card_path.exists():
        print("❌ Model card file not found!")
        return False
    
    with open(model_card_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("🧪 Testing RoBERTa model card...")
    
    # Test 1: Check for required sections
    required_sections = [
        "## Resources",
        "## Notes", 
        "## RobertaConfig",
        "## RobertaTokenizer",
        "## RobertaModel"
    ]
    
    missing_sections = []
    for section in required_sections:
        if section not in content:
            missing_sections.append(section)
    
    if missing_sections:
        print(f"❌ Missing sections: {missing_sections}")
        return False
    else:
        print("✅ All required sections present")
    
    # Test 2: Check badge structure
    badge_pattern = r'<div style="float: right;">\s*<div class="flex flex-wrap space-x-1">'
    if not re.search(badge_pattern, content):
        print("❌ Badge structure not found")
        return False
    else:
        print("✅ Badge structure correct")
    
    # Test 3: Check for conversational tone
    conversational_phrases = [
        "like BERT's smarter cousin",
        "makes it even better",
        "particularly great for"
    ]
    
    found_phrases = [phrase for phrase in conversational_phrases if phrase in content]
    if len(found_phrases) >= 2:
        print("✅ Conversational tone detected")
    else:
        print("❌ Missing conversational tone")
        return False
    
    # Test 4: Check code examples syntax
    code_blocks = re.findall(r'```py\n(.*?)\n```', content, re.DOTALL)
    
    for i, code_block in enumerate(code_blocks):
        try:
            # Remove comments and clean up for syntax checking
            clean_code = re.sub(r'#.*$', '', code_block, flags=re.MULTILINE)
            clean_code = re.sub(r'^\s*$', '', clean_code, flags=re.MULTILINE)
            
            if clean_code.strip():
                ast.parse(clean_code)
        except SyntaxError as e:
            print(f"❌ Syntax error in code block {i+1}: {e}")
            return False
    
    print("✅ All code examples have valid syntax")
    
    # Test 5: Check for "Fixes #36979" (should NOT be present)
    if "Fixes #36979" in content:
        print("❌ Found 'Fixes #36979' - should not be included")
        return False
    else:
        print("✅ No 'Fixes #36979' found (correct)")
    
    # Test 6: Check for contributor attribution
    if "Joao Gante" in content:
        print("✅ Contributor attribution found")
    else:
        print("❌ Contributor attribution missing")
        return False
    
    # Test 7: Check for sentiment analysis examples
    if "sentiment-analysis" in content:
        print("✅ Sentiment analysis examples found")
    else:
        print("❌ Sentiment analysis examples missing")
        return False
    
    # Test 8: Check for Resources section with links
    resources_section = re.search(r'## Resources(.*?)(?=##|$)', content, re.DOTALL)
    if resources_section:
        resources_text = resources_section.group(1)
        if "huggingface.co/papers/1907.11692" in resources_text:
            print("✅ Resources section with paper link found")
        else:
            print("❌ Resources section missing paper link")
            return False
    else:
        print("❌ Resources section not found")
        return False
    
    print("\n🎉 All tests passed! Model card is ready for submission.")
    return True

if __name__ == "__main__":
    success = test_model_card()
    sys.exit(0 if success else 1)
