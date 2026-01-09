import game_logic as bj
import basic_strategy as hc
import text as txt

import torch
from torch import nn 
from torch.utils.data import Dataset, DataLoader, random_split

import time
import numpy as np
import pandas as pd
import tqdm 
import matplotlib.pyplot as plt

from base import GameState, GameInterface

import os
import sys
import csv

class NeuralNetwork(nn.Module):
    def __init__(self, n_inputs, n_layers, n_p_layer, n_outputs_y1, n_outputs_y2):
        super().__init__()
        self.flatten = nn.Flatten()
        
        # Build the shared backbone
        layers = []
        layers.append(nn.Linear(n_inputs, n_p_layer))
        layers.append(nn.LeakyReLU(negative_slope=0.01))

        for _ in range(n_layers):
            layers.append(nn.Linear(n_p_layer, n_p_layer))
            layers.append(nn.LeakyReLU(negative_slope=0.01))

        self.shared = nn.Sequential(*layers)

        # Multi-headed output
        self.head1 = nn.Linear(n_p_layer, n_outputs_y1)
        self.head2 = nn.Linear(n_p_layer, n_outputs_y2)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(1e-3)

    def forward(self, x):
        x = self.flatten(x)
        features = self.shared(x)
        y1 = self.head1(features)
        y2 = self.head2(features)
        return y1, y2

ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
rank_to_index = {rank: i for i, rank in enumerate(ranks)}

namespace = ['basic_strategy', 'sample_neural_net']

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
csv_dir = os.path.join(current_dir, 'csvdata')
model_dir = os.path.join(current_dir, 'models')

def get_models():
    custom_models = []
    with os.scandir(model_dir) as files: 
        for file in files:
            if file.name.endswith('.pt'):
                custom_models.append(file.name)
    
    return custom_models


for model_name in get_models():
    clean_name = model_name[:-3]
    namespace.append(clean_name)

def onehot_card(card):
    '''This is assuming different input than the other version of this function...so I guess how it ends up working is kind of up in the air still'''
    one_hot = [0]*len(ranks)
    one_hot[rank_to_index[card[0]]] = 1
    return np.array(one_hot)
# Turn a card into a hand just by adding two onehot_card arrays

split_result_mapping = {
    'n' : 0, 
    'y' : 1,
    'NA' : 2 # included for reader clarity, not function
}
hsd_result_mapping = {
    'hit' : 0,
    'stand' : 1,
    'double down' : 2,
    'NA' : 3 # do I even need this?
}

reverse_split_mapping = {v: k for k, v in split_result_mapping.items()}
reverse_hsd_mapping = {v: k for k, v in hsd_result_mapping.items()}

def encode_split_hsd(player_hand, dealer_hand, can_double, ui):
    onehot_p_hand = np.zeros(len(ranks))
    for card in player_hand:
        onehot_p_hand += onehot_card(card)
    
    onehot_d_upcard = onehot_card(dealer_hand[0])

    can_split = int(bj.can_split(player_hand)) # will this break on hands with multiple splits?

    # Uncomment the following if we switch to a game-based model vs this round-based model
    can_double = int(len(player_hand) == 2) # and cash >= 2 * bet

    raw_split_result = hc.get_split_choice_hardcode(player_hand, dealer_hand, ui)
    split_result = split_result_mapping[raw_split_result]

    raw_hsd_result = hc.get_hit_stand_dd_hardcode(player_hand, dealer_hand, can_double, ui)
    hsd_result = hsd_result_mapping[raw_hsd_result]

    # Overrides
    if can_split == 0: 
        split_result = 2 #Split_or_not is not allowed
    
    if split_result == 1: 
        hsd_result = 3 #Split makes hsd decision redundant

    return [onehot_p_hand, onehot_d_upcard, can_split, can_double, split_result, hsd_result], raw_split_result, raw_hsd_result

def split_to_csv(player_hand, dealer_hand, ui):
    # Uncomment the following if we switch to a game-based model vs this round-based model
    can_double = int(len(player_hand) == 2) # and cash >= 2 * bet

    row, raw_split_result, raw_hsd_result = encode_split_hsd(player_hand, dealer_hand, can_double, ui)

    with open(os.path.join(csv_dir, 'training_data.csv'), 'a', newline='') as csv_file:
        split_or_not = csv.writer(csv_file)
        split_or_not.writerow(row)

    return raw_split_result
    
def hsd_to_csv(player_hand, dealer_hand, can_double):
    row, raw_split_result, raw_hsd_result = encode_split_hsd(player_hand, dealer_hand, can_double, ui)

    with open(os.path.join(csv_dir, 'training_data.csv'), 'a', newline='') as csv_file:
        split_or_not = csv.writer(csv_file)
        split_or_not.writerow(row)

    return raw_hsd_result

def make_training_data(iterations):
    deck = bj.create_deck()
    bj.shuffle_deck(deck)
    
    print(f"Playing {iterations:,} hands...")
    for _ in tqdm.tqdm(range(iterations)): 
        bj.play_round(
            cash = 1000, #infinite cash relative to bet size
            deck = deck, 
            sleep = False, 
            get_bet = lambda cash: 1, #minimal bet size, the lambda is so its callable to avoid an error
            get_split_choice = split_to_csv,
            get_card = bj.get_card_deal,
            display = hc.display_nothing_hardcode, 
            get_hit_stand_dd = hsd_to_csv,  
            display_hand = hc.display_nothing_hardcode, 
            display_emergency_reshuffle = hc.display_nothing_hardcode, 
            display_final_results = hc.display_nothing_hardcode
        )
    return

def reset_training_data():
    with open(os.path.join(csv_dir, 'training_data.csv'), 'w') as f:
        f.write('onehot_p_hand,onehot_d_upcard,can_split,can_double,split_result,hsd_result\n')

def load_data(batch_size):
    class Blackjack_Dataset(Dataset):
        def __init__(self, X, y1, y2):
            self.X = X
            self.y1 = y1
            self.y2 = y2

        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return self.X[idx], self.y1[idx], self.y2[idx]
        
    df = pd.read_csv(os.path.join(csv_dir, 'training_data.csv'))

    x1 = torch.tensor(np.vstack(df['onehot_p_hand'].apply(lambda s: np.fromstring(s[1:-1], sep=' '))), dtype=torch.float32)
    x2 = torch.tensor(np.vstack(df['onehot_d_upcard'].apply(lambda s: np.fromstring(s[1:-1], sep=' '))), dtype=torch.float32)
    x3 = torch.tensor(df['can_split'].values, dtype=torch.float32).unsqueeze(1)
    x4 = torch.tensor(df['can_double'].values, dtype=torch.float32).unsqueeze(1)
    y1 = torch.tensor(df['split_result'].values, dtype=torch.long)
    y2 = torch.tensor(df['hsd_result'].values, dtype=torch.long)

    X = torch.cat([x1,x2,x3, x4], dim=1)

    dataset = Blackjack_Dataset(X,y1,y2)

    n = len(dataset)
    n_test = int(0.2 * n)
    n_train = n - n_test
    train_ds, test_ds = random_split(dataset, [n_train, n_test])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def train_loop(dataloader, model, loss_fn, optimizer, batch_size, l1_weight):
    size = len(dataloader.dataset)
    model.train()
    batch_losses = []

    for batch, (X, y1, y2) in enumerate(dataloader):
        optimizer.zero_grad()
        pred1, pred2 = model(X)
        loss1 = loss_fn(pred1, y1) 
        loss2 = loss_fn(pred2, y2)
        loss = loss1*l1_weight + loss2

        batch_losses.append(loss.item())

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Printing Training Update on every 100th batch
        if (batch + 1) % 100 == 0: 
            loss = loss.item()
            current = batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return sum((batch_losses))/len(batch_losses)

def test_loop(dataloader, model, loss_fn, l1_weight):
    model.eval()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y1, y2 in dataloader: 
            pred1, pred2 = model(X)

            loss1 = loss_fn(pred1, y1)
            loss2 = loss_fn(pred2, y2)
            test_loss += (loss1*l1_weight + loss2).item()

            correct += ((pred1.argmax(1) == y1) & (pred2.argmax(1) == y2)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size 
    print(f"Test Error: \n Accuracy: {(100*correct):>0.3f}%, Avg loss: {test_loss:>8f} \n")
    return 100*correct, test_loss

def train_model(): 
    while True:
        print()
        model_name = input("Model Name: ")
        if model_name in namespace: 
            print("Please enter a name that is not already taken.")
            print("The following names are taken: ")
            for name in namespace:
                print("    " + name)
            print()
            continue
        else: 
            namespace.append(model_name)
            break

    while True: 
        try: 
            n_layers = int(input("Layers: "))
            break
        except: 
            print("Please enter an integer.")
            print()
            continue

    while True: 
        try: 
            n_p_layer = int(input("Neurons per Layer: "))
            break
        except: 
            print("Please enter an integer.")
            print()
            continue

    
    # print(f"Model: {model_name}")
    # print(f"Layers: {n_layers}")
    # print(f"Neurons per Layer: {n_p_layer}")
    print(f"Total Neurons: {n_layers*n_p_layer}")
    print()

    # The following prints out the model's structure but isn't really very pretty
    print(f"{model_name}'s Structure: ")
    # build structure: list of layers, each layer is a list of neurons (represented here as '.')
    neur_struct = [['.' for _ in range(n_layers)] for _ in range(n_p_layer)]

    for i, layer in enumerate(neur_struct):
        midpoint = len(neur_struct) // 2 

        if i == midpoint - 1:
            if n_p_layer < 3:
                front_buffer = "        "
                back_buffer = "        "
            else: 
                front_buffer = "   /    "
                back_buffer = "   \\    "
        elif i == midpoint:
            front_buffer = " in     "
            back_buffer = "    out "
        elif i == midpoint + 1: 
            if n_p_layer < 3:
                front_buffer = "        "
                back_buffer = "        "
            else:
                front_buffer = "   \\    "
                back_buffer = "   /    "
        else: 
            front_buffer = "        "
            back_buffer = "        "

        
        
        # a simple visual: show layer number and the neuron list
        print(front_buffer + ' '.join(layer) + back_buffer)
    print()


    print("Choose your training hyperparamters: ")
    learning_rate = float(input("Learning Rate: "))
    epochs = int(input("Epochs: "))
    batch_size = int(input("Batch Size: "))
    # loss_fn = 
    # optimizer = 
    #dataset_hands = int(input("Hands in Dataset: "))

    # I need to fix my torch version to make sure this is working right
    # device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    # print(f"Device: {device}")
    device = "cpu"

    n_inputs = 28
    n_outputs_y1 = 3
    n_outputs_y2 = 4

    model = NeuralNetwork(n_inputs, n_layers, n_p_layer, n_outputs_y1, n_outputs_y2).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    #input("Do you want to make new data from scratch or add to your old data?")
    #reset_training_data()
    #make_training_data(dataset_hands)

    train_loader, test_loader = load_data(batch_size)

    # Training
    train_losses = []
    test_losses = []
    test_accs = []

    l1_weight = 1

    print()
    txt.print_title_box(["BEGIN TRAINING"])
    print()

    start_time = time.perf_counter()

    for t in range(epochs):
        print(f"Epoch {t+1}\n---------------------------")
        train_losses.append(train_loop(train_loader, model, loss_fn, optimizer, batch_size, l1_weight))
        test_acc, test_loss = test_loop(test_loader, model, loss_fn, l1_weight)
        test_accs.append(test_acc)
        test_losses.append(test_loss)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(f"\033[3mTraining Completed in {elapsed_time:.4f} seconds.\033[0m")
    print("-"*80)

    # Graphing
    plt.plot(test_accs)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

    #TODO: Label epochs starting at 1 instead of at 0 for graphs

    plt.plot(test_losses, color = 'tab:blue', label = 'test')
    plt.plot(train_losses, color = 'orange', label = 'train')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    # For zooming in 
    # start = 30
    # plt.plot(test_losses[start:], color = 'tab:blue', label = 'test')
    # plt.plot(train_losses[start:], color = 'orange', label = 'train')
    # plt.legend()
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.show()

    # Create a dictionary containing everything needed to recreate the model
    model_params_weights = {
        'architecture_config': {
            'n_inputs': n_inputs,
            'n_layers': n_layers,
            'n_p_layer': n_p_layer,
            'n_outputs_y1': n_outputs_y1,
            'n_outputs_y2': n_outputs_y2
        },
        'model_state_dict': model.state_dict()
    }

    print()
    print("Would you like to save your model? ")
    save_yn = input(">>> ")
    if save_yn == "y": 
        model_name += ".pt"
        model_file_location = os.path.join(model_dir, model_name)
        torch.save(model_params_weights, model_file_location)
        print("\nModel Saved.")
    # TODO: How do I want to integrate the performance tracker? 

    txt.print_title_box(["EXITING TRAINING MODE"])
    print()

#train_model()

def load_model(model_name):
    # 1. Load the checkpoint file
    model_file_location = os.path.join(model_dir, model_name)
    model_params_weights = torch.load(model_file_location)

    # 2. Reconstruct the architecture using the saved config
    model = NeuralNetwork(**model_params_weights['architecture_config'])

    # 3. Load the weights into that architecture
    model.load_state_dict(model_params_weights['model_state_dict'])
    model.eval()

    return model

def get_choices_nn(player_hand, dealer_hand, loaded_model, ui):
    can_double = int(len(player_hand)==2)
    row, _, _ = encode_split_hsd(player_hand, dealer_hand, can_double, ui)

    # This is copied and modified slightly from above, making it fairly redundant and probably not as efficient as it could be
    x1 = torch.tensor(row[0], dtype=torch.float32)
    x2 = torch.tensor(row[1], dtype=torch.float32)
    x3 = torch.tensor([row[2]], dtype=torch.float32)
    x4 = torch.tensor([row[3]], dtype=torch.float32)

    X = torch.cat([x1,x2,x3, x4])
    X = X.unsqueeze(0)
    
    with torch.no_grad():
        y1, y2 = loaded_model(X)

        # # Use softmax to see probabilities if you like
        # probs_split = torch.softmax(y1, dim=1)
        # probs_hsd = torch.softmax(y2, dim=1)

        # print(f"Split Probs: {probs_split}")
        # print(f"HSD Probs: {probs_hsd}")

        split_decision = reverse_split_mapping[torch.argmax(y1).item()]
        hsd_decision = reverse_hsd_mapping[torch.argmax(y2).item()]

        return split_decision, hsd_decision
    
def get_split_choice_nn(player_hand, dealer_hand, loaded_model, ui):
    split_decision, _ = get_choices_nn(player_hand, dealer_hand, loaded_model, ui)
    return split_decision

def get_hit_stand_dd_nn(player_hand, dealer_hand, can_double, loaded_model, ui):
    _, hsd_decision = get_choices_nn(player_hand, dealer_hand, loaded_model, ui)
    return hsd_decision


#TODO: I need to make it so we can use trained models for inference/the performance tracker

if __name__ == '__main__':
    ui = GameInterface()
