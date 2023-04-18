from flask import Flask, request, jsonify
import flask
import torch
from model import Net
import json

app = flask.Flask(__name__)

      

global_model = torch.load('model_checkpoint.pt')

@app.route('/get_global_model', methods=['GET'])
def get_global_model():
    # Serialize the global model to a JSON string
    global_model_json = json.dumps(global_model.state_dict())

    # Return the serialized global model as a response to the client
    return global_model_json




if __name__ == '__main__':
    # Start the Flask server
    app.run(host='0.0.0.0', port=80)
