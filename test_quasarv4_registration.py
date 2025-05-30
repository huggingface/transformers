import os
import sys

# Add the local transformers to the path
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

# Check if QuasarV4 is in the CONFIG_MAPPING
print("Checking if QuasarV4 is registered in AutoConfig...")
try:
    # This will raise an exception if QuasarV4 is not registered
    config = AutoConfig.for_model("quasarv4")
    print("✓ QuasarV4 is registered in AutoConfig")
    print(f"Config class: {config.__class__.__name__}")
except Exception as e:
    print(f"✗ QuasarV4 is not registered in AutoConfig: {e}")

# Check if QuasarV4 is in the MODEL_MAPPING
print("\nChecking if QuasarV4 is registered in AutoModel...")
try:
    # This will raise an exception if QuasarV4 is not registered
    from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES
    if "quasarv4" in MODEL_MAPPING_NAMES:
        print("✓ QuasarV4 is registered in MODEL_MAPPING_NAMES")
        print(f"Model class: {MODEL_MAPPING_NAMES['quasarv4']}")
    else:
        print("✗ QuasarV4 is not registered in MODEL_MAPPING_NAMES")
except Exception as e:
    print(f"✗ Error checking MODEL_MAPPING_NAMES: {e}")

# Check if QuasarV4 is in the MODEL_FOR_CAUSAL_LM_MAPPING
print("\nChecking if QuasarV4 is registered in AutoModelForCausalLM...")
try:
    # This will raise an exception if QuasarV4 is not registered
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
    if "quasarv4" in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
        print("✓ QuasarV4 is registered in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES")
        print(f"Model class: {MODEL_FOR_CAUSAL_LM_MAPPING_NAMES['quasarv4']}")
    else:
        print("✗ QuasarV4 is not registered in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES")
except Exception as e:
    print(f"✗ Error checking MODEL_FOR_CAUSAL_LM_MAPPING_NAMES: {e}")
