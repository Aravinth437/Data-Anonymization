import numpy as np
from loguru import logger
from pathlib import Path

def mask(t, diff):
    if (diff >= 10):
        t = round(t)
    else:
        t = round(t, 4)
    if diff < 1:
        return str(round(t, 2)) + "xx"
    if diff < 10:
        return str(round(t)) + ".xx"
    if diff < 100:
        return str(t)[:len(str(t))-1] + "x"
    if diff < 1000:
        return str(t)[:len(str(t))-2] + "x"


class DatasetAnonymized:
    def __init__(self, anonymized_data: list = list(), pattern_anonymized_data: dict = dict(), suppressed_data: list = list()):
        self.anonymized_data = anonymized_data 
        self.pattern_anonymized_data = pattern_anonymized_data 
        self.suppressed_data = suppressed_data
        self.final_data_anonymized = dict()


    def compute_anonymized_data(self):
        """
        Create dataset ready to be anonymized
        :return:
        """
        logger.info("Start creation dataset anonymized")
        logger.info("Added {} anonymized group".format(len(self.anonymized_data)))
        for index in range(0, len(self.anonymized_data)): 
            #logger.info("Start creation Group {}".format(index))

            group = self.anonymized_data[index]
                        
            max_value = np.amax(np.array(list(group.values())), 0)
            min_value = np.amin(np.array(list(group.values())), 0)
            for key in group.keys():
                # key = row product
                self.final_data_anonymized[key] = list()
                value_row = list()
                for column_index in range(0, len(max_value)):
                     t1 = min_value[column_index]
                    t2 = max_value[column_index]
                    diff = t2-t1
                    t1 = mask(t1, diff)
                    t2 = mask(t2, diff)
                    value_row.append("[{}-{}]".format(t1, t2))

                
                value_row.append(self.pattern_anonymized_data[key]) 
                value_row.append("Group: {}".format(index))

                self.final_data_anonymized[key] = value_row
            #logger.info("Finish creation Group {}".format(index))
        
        logger.info("Added {} suppressed group".format(len(self.suppressed_data)))
        for index in range(0, len(self.suppressed_data)):
            group = self.suppressed_data[index]
            for key in group.keys():
                value_row = [" - "]*len(group[key])
                value_row.append(" - ") # pattern rapresentation
                value_row.append(" - ") # group
                self.final_data_anonymized[key] = value_row

    def save_on_file(self, output_path):
        logger.info("Saving on file dataset anonymized")
        with open(output_path, "w") as file_to_write:
            value_to_print_on_file = ""
            for key, value in self.final_data_anonymized.items():
                value_to_print_on_file = key
                value_to_print_on_file = "{},{}".format(value_to_print_on_file, ",".join(value))
                file_to_write.write(value_to_print_on_file+"\n")
