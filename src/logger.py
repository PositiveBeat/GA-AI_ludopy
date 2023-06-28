"""
Logging passed information to csv files format.

Author: Nicoline Louise Thomsen
"""

import csv
import numpy as np

class Logger():

    def __init__(self, id):

        file_name = 'logs/data' + '_' + str(id) + '.csv'
        log = open(file_name, 'w+', newline='')  # w+ mode truncates (clears) the file (new file for every test)   

        self.logger = csv.writer(log, dialect = 'excel')


    def log_to_file(self, t, *data):

        row = [t]
        row.extend(data)

        self.logger.writerow(row)
