import sys


class CustomException(Exception):

    def __init__(
        self,
        error_message,
        error_detail: sys
    ):

        self.error_message = error_message

        _, _, exc_tb = (
            error_detail.exc_info()
        )

        self.line_number = exc_tb.tb_lineno

    def __str__(self):

        return (
            f"Error occurred at line "
            f"{self.line_number}: "
            f"{self.error_message}"
        )