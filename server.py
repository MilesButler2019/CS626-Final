from flask import Flask, request, jsonify
import torch
from model import Net

app = flask.Flask(__name__)

      

global_model = torch.load('model_checkpoint.pt')

@app.route('/update_model', methods=['POST'])
def update_model():
     
    torch.save(global_model, 'global_model.pt')
      
#     client_model_params = flask.request.get_json()
#     client_model = Net()
#     client_model.load_state_dict(client_model_params)
#     global_model_params = global_model.state_dict()
#     for key in global_model_params:
#         global_model_params[key] += client_model.state_dict()[key]
#     global_model.load_state_dict(global_model_params)



    return 'Model updated'




if __name__ == '__main__':
    # Start the Flask server
    app.run(host='0.0.0.0', port=5000)
