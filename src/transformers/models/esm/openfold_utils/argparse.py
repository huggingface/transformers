from argparse import HelpFormatter
from operator import attrgetter


class ArgparseAlphabetizer(HelpFormatter):
    """
    Sorts the optional arguments of an argparse parser alphabetically
    """

    @staticmethod
    def sort_actions(actions):
        return sorted(actions, key=attrgetter("option_strings"))

    # Formats the help message
    def add_arguments(self, actions):
        actions = ArgparseAlphabetizer.sort_actions(actions)
        super(ArgparseAlphabetizer, self).add_arguments(actions)

    # Formats the usage message
    def add_usage(self, usage, actions, groups, prefix=None):
        actions = ArgparseAlphabetizer.sort_actions(actions)
        args = usage, actions, groups, prefix
        super(ArgparseAlphabetizer, self).add_usage(*args)


def remove_arguments(parser, args):
    for arg in args:
        for action in parser._actions:
            opts = vars(action)["option_strings"]
            if arg in opts:
                parser._handle_conflict_resolve(None, [(arg, action)])
