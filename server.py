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


weights_dict = {}

@app.route('/submit_weights', methods=['POST'])
def submit_weights():
    # Extract the client ID from the request
    client_id = request.form.get('client_id')

    # Extract the weights from the request
    weights = request.form.get('weights')

    # Store the weights in the dictionary
    weights_dict[client_id] = weights

    # Check if we have received weights from all clients
    if len(weights_dict) == NUM_CLIENTS:
        # Aggregate the weights
        global_weights = aggregate_weights(weights_dict)

        # Reset the weights dictionary for the next round
        weights_dict = {}

        # Return the global weights to the clients
        return global_weights
    else:
        # Return a response indicating that the weights were received
        return 'Weights received'

def aggregate_weights(weights_dict):
    # Compute the total number of clients
    num_clients = len(weights_dict)

    # Initialize the global weights to zero
    global_weights = [0] * NUM_WEIGHTS

    # Sum the weights from each client
    for client_weights in weights_dict.values():
        for i, weight in enumerate(client_weights):
            global_weights[i] += weight

    # Average the weights by the number of clients
    for i in range(NUM_WEIGHTS):
        global_weights[i] /= num_clients

    # Return the global weights as a string
    return ','.join(str(weight) for weight in global_weights)
          
 

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
