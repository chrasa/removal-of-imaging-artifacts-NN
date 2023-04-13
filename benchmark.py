import time
import sys
import datetime

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

class ProgressBar:
    def __init__(self):
        self.__pb_started = False

    def print_progress(self, progress, max, progress_bar_length=40, title="Progress"):
        progress_ratio = progress/max
        if not self.__pb_started:
            self.__ts = datetime.datetime.now()
            self.__pb_started = True

        tc = datetime.datetime.now()
        tf = (1/progress_ratio -1)*(tc - self.__ts).total_seconds()

        title = f"\r{title}: {progress:5}/{max:5}: "
        success_rate = f" {progress_ratio*100:3.2f}%"
        remaining_time = f" ERT: {int(tf/60):02}:{int(tf%60):02}"
        number_of_progress_indicators = int(progress * progress_bar_length // (max))

        sys.stdout.write(title + "[" + number_of_progress_indicators*"#" + (progress_bar_length - number_of_progress_indicators)*"-" + "]" + success_rate + remaining_time)

    def end(self):
        sys.stdout.write("\n")
        self.__pb_started = False
