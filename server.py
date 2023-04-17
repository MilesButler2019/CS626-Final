import flask
import torch
from model import Net

app = flask.Flask(__name__)

      
global_model = Net()

@app.route('/update_model', methods=['POST'])
def update_model():
    client_model_params = flask.request.get_json()
    client_model = Net()
    client_model.load_state_dict(client_model_params)
    global_model_params = global_model.state_dict()
    for key in global_model_params:
        global_model_params[key] += client_model.state_dict()[key]
    global_model.load_state_dict(global_model_params)
    return 'Model updated'
