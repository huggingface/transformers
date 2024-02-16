

# Not just the corpus, but also an arbitrary number of query sets, indexed by name in a dictionary/dotdict.
# And also query sets with top-k PIDs.
# QAs too? TripleSets too?


class Dataset:
    def __init__(self):
        pass

    def select(self, key):
        # Select the {corpus, queryset, tripleset, rankingset} determined by uniqueness or by key and return a "unique" dataset (e.g., for key=train)
        pass
