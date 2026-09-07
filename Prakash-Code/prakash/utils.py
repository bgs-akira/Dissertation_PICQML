import os
import logging
import sys
import time

class DFUtils:

    @staticmethod
    def create_filename(filename: str):

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        return filename

class LogUtils:

    @staticmethod
    def log_config(time_stamp, log_dir, filename, level=logging.INFO):
        # time_stamp = datetime.datetime.now().strftime("%d-%b-%Y-(%H.%M.%S.%f)")

        # This will write into existing log file, if there is one
        logging_filename = log_dir + rf'\{filename}.txt'
        os.makedirs(os.path.dirname(logging_filename), exist_ok=True)

        # Remove previous FileHandler and StreamHandler
        try:
            logging.getLogger().removeHandler(logging.getLogger().handlers[-1])
            logging.getLogger().removeHandler(logging.getLogger().handlers[-1])
        except:
            pass

        # This sets the basic config for the root logger
        logging.basicConfig(filename=logging_filename, level=level, format='%(levelname)s %(asctime)s %(message)s')

        stdout_handler = logging.StreamHandler(sys.stdout)
        # make logger print to console (it will not if multithreaded)
        logging.getLogger().addHandler(stdout_handler)  # After this line, the root logger will have one file handler and one streamhandler
        logging.info('')
        logging.info(f'{time_stamp}')

def progressbar(it, prefix="", size=60, out=sys.stdout):
    """Function for progress bar. https://stackoverflow.com/questions/3160699/python-progress-bar """

    count = len(it)
    start = time.time()

    def show(j):
        x = int(size * j / count)
        remaining = ((time.time() - start) / j) * (count - j)

        mins, sec = divmod(remaining, 60)
        time_str = f"{int(mins):02}:{sec:05.2f}"

        print(f"{prefix}[{u'█' * x}{('.' * (size - x))}] {j}/{count} Est wait {time_str}", end='\r', file=out,
              flush=True)

    for i, item in enumerate(it):
        yield item
        show(i + 1)
    print("\n", flush=True, file=out)