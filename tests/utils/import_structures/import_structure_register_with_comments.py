# fmt: off

from transformers.utils.import_utils import export


@export()
# That's a statement
class B0:
    def __init__(self):
        pass


@export()
# That's a statement
def b0():
    pass


@export(backends=("torch", "tf"))
# That's a statement
class B1:
    def __init__(self):
        pass


@export(backends=("torch", "tf"))
# That's a statement
def b1():
    pass


@export(backends=("torch", "tf"))
# That's a statement
class B2:
    def __init__(self):
        pass


@export(backends=("torch", "tf"))
# That's a statement
def b2():
    pass


@export(
    backends=(
        "torch",
        "tf"
    )
)
# That's a statement
class B3:
    def __init__(self):
        pass


@export(
    backends=(
        "torch",
        "tf"
    )
)
# That's a statement
def b3():
    pass
