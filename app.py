import cv2
import torch
import numpy as np
from torchvision import transforms
import requests
import uuid
import argparse
import os
import io
from PIL import Image
import pdb

from flask import Flask, jsonify, request
app = Flask(__name__)

model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
model.eval()

def make_transparent_foreground(pic, mask):
    # split the image into channels
    b, g, r = cv2.split(np.array(pic).astype('uint8'))
    # add an alpha channel with and fill all with transparent pixels (max 255)
    a = np.ones(mask.shape, dtype='uint8') * 255
    # merge the alpha channel back
    alpha_im = cv2.merge([b, g, r, a], 4)
    # create a transparent background
    bg = np.zeros(alpha_im.shape)
    # setup the new mask
    new_mask = np.stack([mask, mask, mask, mask], axis=2)
    # copy only the foreground color pixels from the original image where mask is set
    foreground = np.where(new_mask, alpha_im, bg).astype(np.uint8)

    return foreground


def remove_background(input_image):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.backends.mps.is_available():
        input_batch = input_batch.to('mps')
        model.to('mps')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    # create a binary (black and white) mask of the profile foreground
    bin_mask = torch.where(output_predictions > 0, 255, torch.zeros_like(output_predictions)).byte().cpu().numpy()

    foreground = make_transparent_foreground(input_image, bin_mask)

    return foreground, bin_mask


def inference(img):
    foreground, _ = remove_background(img)
    return foreground

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        content = request.json
        image_url = content['image']
        image_data = requests.get(image_url).content
        image = Image.open(io.BytesIO(image_data))
        image.save(f'static/input_{str(uuid.uuid4())}.png')
        foreground, bin_mask = remove_background(image)
        Image.fromarray(bin_mask).save(f'static/input_{str(uuid.uuid4())}.png')
        filepath = f'static/{str(uuid.uuid4())}.png'
        Image.fromarray(foreground).save(filepath)
        return f'http://localhost:4999/{filepath}'

if __name__ == '__main__':
    app.run()
