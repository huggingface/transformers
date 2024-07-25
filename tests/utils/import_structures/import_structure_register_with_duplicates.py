# fmt: off

from transformers.utils.import_utils import export


@export(backends=("torch", "torch"))
class C0:
    def __init__(self):
        pass


@export(backends=("torch", "torch"))
def c0():
    pass


@export(backends=("torch", "torch"))
# That's a statement
class C1:
    def __init__(self):
        pass


@export(backends=("torch", "torch"))
# That's a statement
def c1():
    pass


@export(backends=("torch", "torch"))
# That's a statement
class C2:
    def __init__(self):
        pass


@export(backends=("torch", "torch"))
# That's a statement
def c2():
    pass


@export(
    backends=(
        "torch",
        "torch"
    )
)
# That's a statement
class C3:
    def __init__(self):
        pass


@export(
    backends=(
        "torch",
        "torch"
    )
)
# That's a statement
def c3():
    pass
