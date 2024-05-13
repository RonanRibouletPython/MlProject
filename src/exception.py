# Website link to understand the customed exception handler in python
# https://docs.python.org/3/tutorial/errors.html#handling-exceptions

import sys 
from src.logger import logging

# This function must be called when an error is raised 
def error_message_detail(error, error_detail:sys):
    # Gives the information on which file and at which line the exception has occured
    _,_,exc_tb = error_detail.exc_info()
    # Filename where the exception occured
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "An error has occured in the python script named [{0}] at line [{1}] with error message[{2}]".format(
        file_name,exc_tb.tb_lineno,str(error))

    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        # Method to initialize the attributes of the parent class
        super().__init__(error_message)

        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message

# Test of the exception handler system
if __name__ == "__main__":

    try:
        a = 1/0
    except Exception as e:
        logging.info("Divided by Zero Error")
        raise CustomException(e, sys)
    
        