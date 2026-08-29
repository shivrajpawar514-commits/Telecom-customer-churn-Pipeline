import sys
from src.logger import get_logger

logger = get_logger(__name__)


def error_message_detail(error, error_detail: sys):
    """
    Returns a detailed error message including
    filename and line number.
    """
    _, _, exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename

    message = (
        f"Error occurred in Python script [{file_name}] "
        f"at line [{exc_tb.tb_lineno}] "
        f"Error message: {str(error)}"
    )

    return message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)

        self.error_message = error_message_detail(
            error_message,
            error_detail
        )

        logger.error(self.error_message)

    def __str__(self):
        return self.error_message