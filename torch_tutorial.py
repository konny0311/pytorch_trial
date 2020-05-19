# trial code of torch tutorial on 
# https://www.atmarkit.co.jp/ait/articles/2002/06/news025.html

import torch
import torch.nn as nn

INPUT_FEATURES = 2
OUTPUT_NEURONS = 1

activation = torch.nn.Tanh()

class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.layer1 = nn.Linear(INPUT_FEATURES, OUTPUT_NEURONS)

    def forward(self, input):
        output = activation(self.layer1(input))

        return output

model = NeuralNetwork()
print(model)

weight_array = nn.Parameter(torch.tensor([[0.6, -0.2]]))
bias_array = nn.Parameter(torch.tensor([0.8]))

model.layer1.weight = weight_array
model.layer1.bias = bias_array

params = model.state_dict()
print(params)