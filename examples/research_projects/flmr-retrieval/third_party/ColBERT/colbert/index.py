

# TODO: This is the loaded index, underneath a searcher.


"""
## Operations:

index = Index(index='/path/to/index')
index.load_to_memory()

batch_of_pids = [2324,32432,98743,23432]
index.lookup(batch_of_pids, device='cuda:0') -> (N, doc_maxlen, dim)

index.iterate_over_parts()

"""
