# flask_app/server.py​

from flask import Flask, request, jsonify, render_template
# from flask_dropzone import Dropzone

import logging
import random
import time
import os


import requests, os

# import settingspip u
from run_squad import initialize, evaluate
from data.squad_generator import convert_text_input_to_squad
from settings import *

os.makedirs(output_dir, exist_ok=True)

# args, model, tokenizer = None, None, None
args, model, tokenizer = initialize()

app = Flask(__name__)
# dropzone = Dropzone(app)

@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/", methods=['POST'])
def process_text():
    input = request.form["textbox"]
    if validate_input(input):
        return predict(input)
    return input

# @app.route('/', methods=['POST'])
# def index():
#     file_obj = request.files
#     for f in file_obj:
#         file = request.files.get(f)
#         print (file.filename)
#     return "uploading..."

def validate_input(input):
    paragraphs = input.split("\n\n")

    for p in paragraphs:
        p = p.split("\n")
        if len(p) < 3:
            return False
        else:
            questions = p[2:]
            for q in questions:
                if not q:
                    return False
    return True

def predict(input):

    convert_text_input_to_squad(input, "./data/squad_input.json")
    args.predict_file = "./data/squad_input.json"

    t = time.time()

    predictions = evaluate(args, model, tokenizer)
    dt = time.time() - t
    app.logger.info("Execution time: %0.02f seconds" % (dt))
    return jsonify(predictions)




if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=False, port=PORT)