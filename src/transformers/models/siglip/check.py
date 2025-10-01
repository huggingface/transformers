import transformers

def check_whether_transformers_replace_is_installed_correctly():
    return transformers.__version__ == "4.53.2"