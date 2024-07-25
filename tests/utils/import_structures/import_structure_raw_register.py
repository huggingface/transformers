# fmt: off

from transformers.utils.import_utils import export


@export()
class A0:
    def __init__(self):
        pass


@export()
def a0():
    pass


@export(backends=("torch", "tf"))
class A1:
    def __init__(self):
        pass


@export(backends=("torch", "tf"))
def a1():
    pass


@export(
    backends=("torch", "tf")
)
class A2:
    def __init__(self):
        pass


@export(
    backends=("torch", "tf")
)
def a2():
    pass


@export(
    backends=(
        "torch",
        "tf"
    )
)
class A3:
    def __init__(self):
        pass


@export(
    backends=(
            "torch",
            "tf"
    )
)
def a3():
    pass

@export(backends=())
class A4:
    def __init__(self):
        pass
