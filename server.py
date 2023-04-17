import flask
import torch

app = flask.Flask(__name__)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output
      
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
