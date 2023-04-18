from flask import Flask, request, jsonify
import flask
import torch
from model import Net
import json

app = flask.Flask(__name__)

      


@app.route('/get_global_model', methods=['GET'])
def get_global_model():
    loaded_object = torch.load('model_checkpoint.pt')
    # If the loaded object is an OrderedDict, convert it to a PyTorch model
    if isinstance(loaded_object, dict):
        model = Net()
        model.load_state_dict(loaded_object)
    else:
        model = loaded_object

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
