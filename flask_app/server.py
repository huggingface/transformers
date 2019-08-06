# flask_app/server.py​

from flask import Flask, request, jsonify, render_template, session, url_for, redirect
from flask_dropzone import Dropzone
import time
from urllib.parse import unquote
import wikipedia
import os

from run_squad import initialize, evaluate
from data.squad_generator import convert_text_input_to_squad, \
    convert_file_input_to_squad, convert_context_and_questions_to_squad
from settings import *

os.makedirs(output_dir, exist_ok=True)

# args, model, tokenizer = None, None, None
args, model, tokenizer = initialize()

app = Flask(__name__)

app.config['DROPZONE_UPLOAD_MULTIPLE'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'text'
app.config['SECRET_KEY'] = 'supersecret'

dropzone = Dropzone(app)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/", methods=['POST'])
def process_input():
    if request.files:
        if "file_urls" not in session:
            session['file_urls'] = []
            # list to hold our uploaded image urls
        file_urls = session['file_urls']

        file_obj = request.files
        for f in file_obj:
            file = request.files.get(f)
            app.logger.info("file upload {}".format(file.filename))
            os.makedirs("./uploads", exist_ok=True)
            filepath = os.path.join('./uploads', file.filename)
            file.save(filepath)
            file_urls.append(filepath)
        return "upload"
    else:
        input = request.form["textbox"]
        try:
            return predict_from_text_squad(input)
        except AssertionError:
            return index()

@app.route("/_random_page")
def random_page():
    r = wikipedia.random(1)
    try:
        res = wikipedia.page(r)
        res_title = res.title
        res_sum = res.summary
    except wikipedia.exceptions.DisambiguationError as e:
        return random_page()
    return jsonify(context='\n'.join([res_title, res_sum]),
                   question="What is {}?".format(res_title))



def predict_from_text_squad(input):
    squad_dict = convert_text_input_to_squad(input, gen_file)
    return package_squad_prediction(evaluate_input(gen_file), squad_dict)

def predict_from_file_squad(input):
    try:
        squad_dict = convert_file_input_to_squad(input, gen_file)
    except AssertionError:
        return []
    return package_squad_prediction(evaluate_input(gen_file), squad_dict)

def predict_from_input_squad(context, questions):
    squad_dict = convert_context_and_questions_to_squad(context, questions, gen_file)
    return package_squad_prediction(evaluate_input(gen_file), squad_dict)

def package_squad_prediction(prediction, squad_dict):
    squad_dict = squad_dict["data"]
    packaged_predictions = []
    for entry in squad_dict:
        title = entry["title"]
        inner_package = []
        for p in entry["paragraphs"]:
            context = p["context"]
            qas = [(q["question"], prediction[q["id"]]) for q in p["qas"]]
            inner_package.append((context, qas))
        packaged_predictions.append((title, inner_package))
    return packaged_predictions



def evaluate_input(predict_file, passthrough=False):
    args.predict_file = predict_file
    t = time.time()
    predictions = evaluate(args, model, tokenizer)
    dt = time.time() - t
    app.logger.info("Execution time: %0.02f seconds" % (dt))
    if passthrough:
        return predictions, predict_file
    return predictions

@app.route('/_input_helper')
def input_helper():
    text = unquote(request.args.get("text_data", "", type=str)).strip()
    questions = unquote(request.args.get("question_data", "", type=str)).strip()
    app.logger.info("input text: {}\n\nquestions:{}".format(text, questions))
    if text and questions:
        return jsonify(result=
                       render_template('results.html',
                                       file_urls=[text],
                                       predict=lambda x: predict_from_input_squad(text, questions)))
    else:
        if "file_urls" not in session or session['file_urls'] == []:
            return redirect(url_for('index'))
        file_urls = session['file_urls']
        session.pop('file_urls', None)
        app.logger.info("input file list: {}".format(file_urls))
        return jsonify(result=
                       render_template('results.html',
                                       file_urls=file_urls,
                                       predict=predict_from_file_squad))


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=PORT)