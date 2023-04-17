import requests
import torch.nn.functional as F
from tqdm import tqdm

from net import model

local_data = ...




local_model = Net()

for epoch in range(num_epochs):
    local_model_params = local_model.state_dict()
    optimizer = optim.SGD(local_model.parameters(), lr=lr)
    optimizer.zero_grad()
    output = local_model(local_data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    requests.post('http://server_ip:5000/update_model', json=local_model_params)
    tqdm.write(f'Epoch {epoch+1}/{num_epochs}: Loss={loss.item():.4f}')
