from flask import Flask, render_template, request, flash, jsonify
import io
from PIL import Image
import json
from flask_cors import CORS
from predict import *


app = Flask(__name__)
CORS(app)

@app.route("/predict_img", methods=['POST', 'GET'])
def predict_label():
    if request.method == 'GET':
        return render_template("index.html", value="Image")
    if request.method == "POST":
        if "file" not in request.files:
            return "Image not uploaded"

        file = request.files["file"].read()

        try:
            img = Image.open(io.BytesIO(file))
        except IOError:
            return jsonify(predictions="Not an Image, please upload file a gain!")

        img = img.convert("RGB")
        label = predict(img)

        return jsonify(predictions=label)


if __name__ == '__main__':
    app.run(host='localhost', port='6868', debug=False)