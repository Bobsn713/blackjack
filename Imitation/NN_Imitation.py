import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import pandas as pd
from sklearn.model_selection import train_test_split

#Helper functions for data processing
def card_to_num(card):
    raw_rank = card[:-1]
    
    ranks = {
        '2' : 0,
        '3' : 1,
        '4' : 2, 
        '5' : 3,
        '6' : 4, 
        '7' : 5, 
        '8' : 6, 
        '9' : 7, 
        '10': 8, 
        'J' : 9, 
        'Q' : 10, 
        'K' : 11, 
        'A': 12
    }

    return ranks[raw_rank]

def hand_to_list(hand):
    '''Takes hand like KH-AC and outputs list of card numbers'''
    hand_list_1 = hand.split("-")
    hand_list_2 = [card_to_num(card) for card in hand_list_1]
    return hand_list_2

result_mapping = {
    'hit' : 0,
    'stand' : 1,
    'double down' : 2
}

batch_size = 32

# Defining Dataset Class
class Blackjack_Dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

# Data processing
split_or_not_raw_df = pd.read_csv('CSVs/split_or_not.csv')
hit_stand_dd_df = pd.read_csv('CSVs/hit_stand_dd.csv')

# Cleaned split_or_not
split_or_not_raw_df['dealer_upcard'] = split_or_not_raw_df['dealer_upcard'].apply(card_to_num)
split_or_not_raw_df['player_hand'] = split_or_not_raw_df['player_hand'].apply(hand_to_list)
split_or_not_raw_df['player_hand'] = split_or_not_raw_df['player_hand'].apply(lambda hand: hand[0])
split_or_not_df = split_or_not_raw_df.rename(columns = {'player_hand':'player_upcard'})

# Cleaned hit_stand_dd
MAX_LEN = 7
hit_stand_dd_df['dealer_upcard'] = hit_stand_dd_df['dealer_upcard'].apply(card_to_num)
hit_stand_dd_df['player_hand'] = hit_stand_dd_df['player_hand'].apply(hand_to_list)
hit_stand_dd_df['result'] = hit_stand_dd_df['result'].map(result_mapping)
hit_stand_dd_df['player_hand'] = [
    hand + [0] * (MAX_LEN - len(hand)) if len(hand) < MAX_LEN else hand[:MAX_LEN] for hand in hit_stand_dd_df['player_hand']
]

# Turning into tensor matrices
# split_or_not
x1 = torch.tensor(split_or_not_df['player_upcard'].values, dtype=torch.float32).unsqueeze(1)
x2 = torch.tensor(split_or_not_df['dealer_upcard'].values, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(split_or_not_df['result'].values, dtype=torch.long)

X = torch.cat([x1,x2], dim=1)

split_or_not_dataset = Blackjack_Dataset(X,y)
train_sn, test_sn = train_test_split(split_or_not_dataset, test_size=0.2) #I might have to do this earlier, on the dataframe, but this was easier so let's see if it works

sn_train_dataloader = DataLoader(train_sn, batch_size=batch_size, shuffle=True)
sn_test_dataloader = DataLoader(test_sn, batch_size=batch_size, shuffle=True)

# hit_stand_dd
x1 = torch.tensor(hit_stand_dd_df['player_hand'].to_list(), dtype=torch.float32)
x2 = torch.tensor(hit_stand_dd_df['dealer_upcard'].values, dtype=torch.float32).unsqueeze(1)
x3 = torch.tensor(hit_stand_dd_df['can_double'].values, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(hit_stand_dd_df['result'].values, dtype=torch.long)

X = torch.cat([x1,x2,x3], dim=1)

hit_stand_dd_dataset = Blackjack_Dataset(X,y)
train_hsd, test_hsd = train_test_split(hit_stand_dd_dataset, test_size=0.2)

hsd_train_dataloader = DataLoader(train_hsd, batch_size=batch_size, shuffle=True)
hsd_test_dataloader = DataLoader(test_hsd, batch_size=batch_size, shuffle=True)

class sn_NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

sn_model = sn_NeuralNetwork()

class hsd_NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

hsd_model = hsd_NeuralNetwork()

learning_rate = 0.001 
epochs = 20

loss_fn = nn.CrossEntropyLoss()
sn_optimizer = torch.optim.SGD(sn_model.parameters(), lr=learning_rate)
hsd_optimizer = torch.optim.SGD(hsd_model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()

    for batch, (X,y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Printing Training Update on every 100th batch
        if (batch + 1) % 100 == 0: 
            loss = loss.item()
            current = batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    #Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    #Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True

    with torch.no_grad():
        for X, y in dataloader: 
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size 
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

for t in range(epochs):
    print(f"Epoch {t+1}\n---------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")