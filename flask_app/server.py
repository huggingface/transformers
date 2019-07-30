# flask_app/server.py​
# import libraries
print('importing libraries...')

from flask import Flask, request, jsonify
import logging
import random
import time

import requests, os
from io import BytesIO

# import settings
from run_squad import initialize, evaluate

