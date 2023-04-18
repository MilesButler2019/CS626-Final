from flask import Flask, request, jsonify
import flask
import torch
from model import Net
from torchvision import datasets, transforms
import json
import numpy as np
import torch.utils.data as data_utils

app = flask.Flask(__name__)


num_clients = 0

global_weights = torch.load('model_checkpoint.pt')


@app.route("/upload", methods=["POST"])
def upload_file():
    # Get the uploaded file
    uploaded_file = request.files["model_file"]
    # Save the file to disk
    uploaded_file.save("./CS626-Final/models")
    return "File uploaded successfully"
 

@app.route('/get_global_model', methods=['GET'])
def get_global_model():
    global global_weights
    # If the loaded object is an OrderedDict, convert it to a PyTorch model
    if isinstance(global_weights, dict):
        model = Net()
        model.load_state_dict(global_weights)
    else:
        model = global_weights

    # Extract the model parameters and convert them to a dictionary
    model_params = model.state_dict()
    model_params_dict = {}
    for param_name, param_tensor in model_params.items():
        model_params_dict[param_name] = param_tensor.tolist()

    # Serialize the model parameters dictionary to a JSON string
    model_params_json = json.dumps(model_params_dict)

    # Return the serialized model parameters as a response to the client
    return model_params_json





if __name__ == '__main__':
    # Start the Flask server
    app.run(host='0.0.0.0', port=5000)
