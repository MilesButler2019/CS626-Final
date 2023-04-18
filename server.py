from flask import Flask, request, jsonify
import flask
import torch
from model import Net
import json

app = flask.Flask(__name__)

      

global_model = torch.load('model_checkpoint.pt')

@app.route('/get_global_model', methods=['GET'])
def get_global_model():
    # Extract the model parameters and convert them to a dictionary
    model_params = global_model.state_dict()
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
