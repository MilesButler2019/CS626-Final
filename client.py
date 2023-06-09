import requests
import torch.nn.functional as F
from tqdm import tqdm

from model import Net

local_data = ...



response = requests.get('http://3.84.112.131:5000/get_global_model')
global_model_params = response.json()


local_model = Net()
local_model.load_state_dict(global_model_params)

num_epochs = 1
for epoch in range(num_epochs):
    local_model_params = model.state_dict()
    optimizer = optim.SGD(local_model.parameters(), lr=lr)
    optimizer.zero_grad()
    output = local_model(local_data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
#     requests.post('http://server_ip:5000/update_model', json=local_model_params)
    tqdm.write(f'Epoch {epoch+1}/{num_epochs}: Loss={loss.item():.4f}')
