import os
from datetime import datetime

log_file_location = ''
log_file_name = 'log.txt'


def print_message(message):
    # type: (str) -> None
    if not os.path.exists(log_file_location + log_file_name):
        __create_log_file()
    separator = "  -  "
    with open(log_file_location + log_file_name, mode='a') as f:
        f.write(str(datetime.now()) + separator + message + '\n')


def __create_log_file():
    if not log_file_location == '':
        if not log_file_location.endswith('/'):
            raise Exception('Invalid location of log file.')
        if not os.path.exists(log_file_location):
            os.makedirs(log_file_location)

    # Needed for creating the file
    with open(log_file_location + log_file_name, mode='w') as _:
        pass


def reset():
    __create_log_file()
