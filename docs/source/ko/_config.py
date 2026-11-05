# docstyle-ignore
INSTALL_CONTENT = """
# Transformers 설치 방법
! pip install transformers datasets evaluate accelerate
# 마지막 릴리스 대신 소스에서 설치하려면, 위 명령을 주석으로 바꾸고 아래 명령을 해제하세요.
# ! pip install git+https://github.com/huggingface/transformers.git
"""

notebook_first_cells = [{"type": "code", "content": INSTALL_CONTENT}]
black_avoid_patterns = {
    "{processor_class}": "FakeProcessorClass",
    "{model_class}": "FakeModelClass",
    "{object_class}": "FakeObjectClass",
}
