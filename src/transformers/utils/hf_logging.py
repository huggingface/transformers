import logging


def get_logger(name: str = "transformers"):
    """
    Returns a standard Python logger without modifying handlers or log levels.
    Required for passing HuggingFace tests.
    """
    return logging.getLogger(name)



# import logging

# _logger = None

# def get_logger(name: str = "transformers"):
#     global _logger
#     if _logger is None:
#         _logger = logging.getLogger(name)
#         if not _logger.handlers:
#             handler = logging.StreamHandler()
#             formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
#             handler.setFormatter(formatter)
#             _logger.addHandler(handler)
#         _logger.setLevel(logging.INFO)
#     return _logger
