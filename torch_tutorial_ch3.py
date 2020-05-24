import plygdata as pg
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

PROBLEM_DATA_TYPE = pg.DatasetType.ClassifyCircleData
TRAINING_DATA_RATIO = 0.5
DATA_NOISE = 0.0
LEARNING_RATE = 0.03
REGULARIZATION = 0.03
EPOCHS = 100


data_list = pg.generate_data(PROBLEM_DATA_TYPE, DATA_NOISE)
X_train, y_train, X_valid, y_valid = pg.split_data(data_list, training_size=TRAINING_DATA_RATIO)

BATCH_SIZE = 15

t_X_train = torch.from_numpy(X_train).float()
t_y_train = torch.from_numpy(y_train).float()
t_X_valid = torch.from_numpy(X_valid).float()
t_y_valid = torch.from_numpy(y_valid).float()

dataset_train = TensorDataset(t_X_train, t_y_train)
dataset_valid = TensorDataset(t_X_valid, t_y_valid)

loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
loader_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE)

def discretize(proba):
    threshold = torch.Tensor([0.0])
    discretized = (proba >= threshold).float()
    return discretized * 2 - 1.0

class Discritize(nn.Module):
    '''
    Examples:
        >>> d = Discretize()
        >>> proba = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float)
        >>> binary = d(proba)
    '''        
    def __init__(self):
        super().__init__()

    def forward(self, proba): # forward() is called in torch.nn.Module.__call__()
        return discretize(proba)

proba = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float)
binary = discretize(proba)
print(binary)

# 定数（モデル定義時に必要となるもの）
INPUT_FEATURES = 2      # 入力（特徴）の数： 2
LAYER1_NEURONS = 3      # ニューロンの数： 3
LAYER2_NEURONS = 3      # ニューロンの数： 3
OUTPUT_RESULTS = 1      # 出力結果の数： 1

# 変数（モデル定義時に必要となるもの）
activation1 = torch.nn.Tanh()  # 活性化関数（隠れ層用）： tanh関数（変更可能）
activation2 = torch.nn.Tanh()  # 活性化関数（隠れ層用）： tanh関数（変更可能）
acti_out = torch.nn.Tanh()     # 活性化関数（出力層用）： tanh関数（固定）

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.layer1 = nn.Linear(INPUT_FEATURES, LAYER1_NEURONS)
        self.layer2 = nn.Linear(LAYER1_NEURONS, LAYER2_NEURONS)
        self.layer_out = nn.Linear(LAYER2_NEURONS, OUTPUT_RESULTS)

    def forward(self, x): # memo: looks like Keras
        x  = activation1(self.layer1(x))
        x  = activation2(self.layer2(x))
        x  = acti_out(self.layer_out(x))

        return x

model = NeuralNetwork()
print(model.state_dict())

optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)
criterion = nn.MSELoss()

def train_step(X_train, y_train):
    model.train()

    pred_y = model(X_train)
    optimizer.zero_grad()
    loss = criterion(pred_y, y_train)
    loss.backward()

    optimizer.step()

    with torch.no_grad():
        discr_y = discretize(pred_y)
        acc = (discr_y == y_train).sum()

        return (loss.item(), acc.item())

def valid_step(X_valid, y_valid):
    model.eval()
    pred_y = model(X_valid)
    loss = criterion(pred_y, y_valid)

    with torch.no_grad():
        discr_y = discretize(pred_y)
        acc = (discr_y == y_valid).sum()

        return (loss.item(), acc.item())

def init_parameters(layer):
    if type(layer) == nn.Linear:
        nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0.0)

model.apply(init_parameters)
avg_loss = 0.0
avg_acc = 0.0
avg_val_loss = 0.0
avg_val_acc = 0.0

train_history = []
valid_history = []

for epoch in range(EPOCHS):
    total_loss = 0.0
    total_acc = 0.0
    total_val_loss = 0.0
    total_val_acc = 0.0
    total_train = 0
    total_valid = 0

    for X_train, y_train in loader_train:
        loss, acc = train_step(X_train, y_train)

        total_loss += loss
        total_acc += acc
        total_train += len(y_train)

    for X_valid, y_valid in loader_valid:
        val_loss, val_acc = valid_step(X_valid, y_valid)

        total_val_loss += val_loss
        total_val_acc += val_acc
        total_valid += len(y_valid)

    n = epoch + 1
    avg_loss = total_loss / n
    avg_acc = total_acc / total_train
    avg_val_loss = total_val_loss / n
    avg_val_acc = total_val_acc / total_valid

    train_history.append(avg_loss)
    valid_history.append(avg_val_loss)

    print(f'[Epoch {epoch+1:3d}/{EPOCHS:3d}]' \
          f' loss: {avg_loss:.5f}, acc: {avg_acc:.5f}' \
          f' val_loss: {avg_val_loss:.5f}, val_acc: {avg_val_acc:.5f}')

print('Finished Training')
print(model.state_dict())