import time

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def timeit(func):
    def timed(*args, **kwargs):
        ts = time.time()
        result = func(*args, **kwargs)
        te = time.time()
        print(f'Function:{bcolors.WARNING}{func.__name__:>30}{bcolors.ENDC} Execution time:{bcolors.OKGREEN}{(te -ts):10.2f}s{bcolors.ENDC}')
        return result
    return timed