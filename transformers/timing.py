import time


def timeit(method):
    def timed(*args, **kw):

        print(f'{method.__qualname__} called...')
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print(f'{method.__qualname__} took {(te - ts) * 1000} ms')
        return result
    return timed
