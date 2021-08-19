# Custom Exception classes
# Add Customized Exception classes for different User-based errors

import os, sys

class BatchSizeError(BaseException):
    def __init__(self, *args: object) -> None:
        super(BatchSizeError,self).__init__(*args)
    
