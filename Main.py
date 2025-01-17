import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from flask import Flask, request, render_template, jsonify
from markupsafe import Markup

import pandas as pd
import numpy as np
import difflib

from sklearn.model_selection import train_test_split

import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the model and tokenizer
model_path = "./gpt2/model/chatbot_gpt_2"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

tokenizer.pad_token = "<pad>"
tokenizer.bos_token = "<bos>"
tokenizer.eos_token = "<eos>"
model.eval()

simple_phrases = {
    "halo": "Halo! Ada yang bisa saya bantu?",
    "apa kabar": "Saya baik, terima kasih! Bagaimana dengan Anda?",
    "selamat pagi": "Selamat pagi! Semoga harimu menyenangkan!",
    "selamat sore": "Selamat sore! Bagaimana kabar Anda hari ini?",
    "terima kasih": "Sama-sama! Senang bisa membantu.",
}


# Fungsi untuk mencocokkan input dengan frasa sederhana menggunakan toleransi
def match_simple_phrase(user_input, threshold=0.6):
    matches = difflib.get_close_matches(
        user_input, simple_phrases.keys(), n=1, cutoff=threshold
    )
    return matches[0] if matches else None


# Function to generate a response
def generate_response(user_input):
    user_input_normalized = user_input.lower().strip()

    matched_phrase = match_simple_phrase(user_input_normalized)
    if matched_phrase:
        return simple_phrases[matched_phrase]

    # Format the input with special tokens
    input_text = f"<bos> {user_input} <bot>"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    input_ids = input_ids.to(model.device)

    output_ids = model.generate(
        input_ids=input_ids,
        do_sample=True,
        max_length=512,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if "<bot>" in response:
        response = response.split("<bot>")[-1].strip()
    return response


def __MAIN__():
    app = Flask(__name__, template_folder="templates", static_folder="static")

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/generate-response", methods=["POST"])
    def chat_response():
        data = request.get_json()
        user_input = data.get("message", "")
        response = generate_response(user_input)
        # response = generate_response_multiprocess(user_input)
        return jsonify({"response": response})

    app.run(debug=True, threaded=True)


__MAIN__()
