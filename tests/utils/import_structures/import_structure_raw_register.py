# fmt: off

from transformers.utils.import_utils import register


@register()
class A0:
    def __init__(self):
        pass


@register()
def a0():
    pass


@register(backends=("torch", "tf"))
class A1:
    def __init__(self):
        pass


@register(backends=("torch", "tf"))
def a1():
    pass


@register(
    backends=("torch", "tf")
)
class A2:
    def __init__(self):
        pass


@register(
    backends=("torch", "tf")
)
def a2():
    pass


@register(
    backends=(
        "torch",
        "tf"
    )
)
class A3:
    def __init__(self):
        pass


@register(
    backends=(
            "torch",
            "tf"
    )
)
def a3():
    pass
