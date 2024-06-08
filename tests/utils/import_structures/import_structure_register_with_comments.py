# fmt: off

from transformers.utils.import_utils import register


@register()
# That's a statement
class B0:
    def __init__(self):
        pass


@register()
# That's a statement
def b0():
    pass


@register(backends=("torch", "tf"))
# That's a statement
class B1:
    def __init__(self):
        pass


@register(backends=("torch", "tf"))
# That's a statement
def b1():
    pass


@register(
    backends=("torch", "tf")
)
# That's a statement
class B2:
    def __init__(self):
        pass


@register(
    backends=("torch", "tf")
)
# That's a statement
def b2():
    pass


@register(
    backends=(
            "torch",
            "tf"
    )
)
# That's a statement
class B3:
    def __init__(self):
        pass


@register(
    backends=(
            "torch",
            "tf"
    )
)
# That's a statement
def b3():
    pass
