import logging
import sys


class SuppressStdout:
    def __enter__(self):
        self.stdout = sys.stdout
        dev_null = open("/dev/null", "w")
        sys.stdout = dev_null

    def __exit__(self, typ, value, traceback):
        fp = sys.stdout
        sys.stdout = self.stdout
        fp.close()


class SuppressLogging:
    def __init__(self, level):
        self.level = level

    def __enter__(self):
        logging.disable(self.level)

    def __exit__(self, typ, value, traceback):
        logging.disable(logging.NOTSET)
