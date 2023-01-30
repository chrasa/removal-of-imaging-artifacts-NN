import time

def timeit(func):
    def timed(*args, **kwargs):
        ts = time.time()
        result = func(*args, **kwargs)
        te = time.time()
        print(f'Function:{func.__name__:>30} Execution time:{(te -ts):10.2f}s')
        return result
    return timed