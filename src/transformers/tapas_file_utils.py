import os 
from typing import List, Text

def make_directories(path):
  """Create directory recursively. Don't do anything if directory exits."""
  os.makedirs(path)


def list_directory(path):
  """List directory contents."""
  return os.listdir(path)