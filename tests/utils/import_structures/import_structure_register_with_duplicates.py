# fmt: off

from transformers.utils.import_utils import register


@register(backends=("torch", "torch"))
class C0:
    def __init__(self):
        pass


@register(backends=("torch", "torch"))
def c0():
    pass


@register(backends=("torch", "torch"))
# That's a statement
class C1:
    def __init__(self):
        pass


@register(backends=("torch", "torch"))
# That's a statement
def c1():
    pass


@register(
    backends=("torch", "torch")
)
# That's a statement
class C2:
    def __init__(self):
        pass


@register(
    backends=("torch", "torch")
)
# That's a statement
def c2():
    pass


@register(
    backends=(
        "torch",
        "torch"
    )
)
# That's a statement
class C3:
    def __init__(self):
        pass


@register(
    backends=(
        "torch",
        "torch"
    )
)
# That's a statement
def c3():
    pass
