import io
import numpy as np
from PIL import Image
import requests
from rembg import remove
import uuid

from flask import Flask, request
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        content = request.json
        image_url = content['image']
        image_data = requests.get(image_url).content
        image = Image.open(io.BytesIO(image_data))

        output = remove(image)
        filepath = f'static/{str(uuid.uuid4())}.png'
        output.save(filepath)

        return f'http://localhost:4999/{filepath}'

if __name__ == '__main__':
    app.run()
