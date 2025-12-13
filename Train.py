import Logic as bj
from torch import nn
import torch
import Hardcode as hc
import numpy as np
import os
import sys
import csv


ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
rank_to_index = {rank: i for i, rank in enumerate(ranks)}

namespace = ['hardcode', 'hc', 'imitation', 'imit']
n_inputs = 105
n_outputs = 3

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
csv_dir = os.path.join(current_dir, 'csvdata')

def onehot_card(card):
    '''This is assuming different input than the other version of this function...so I guess how it ends up working is kind of up in the air still'''
    one_hot = [0]*len(ranks)
    one_hot[rank_to_index[card[0]]] = 1
    return np.array(one_hot)
# Turn a card into a hand just by adding two onehot_card arrays

# Can I make a hardcode_to_csv function that works for both splits and hsds so there can be one training set? 
# Something like (onehot_phand, d_upcard, can_split, can_double, split, hsd)

def make_training_data(hands):
    deck = bj.create_deck()
    bj.shuffle_deck(deck)
        
    bj.play_round(
        cash = 1000, #infinite cash relative to bet size
        deck = deck, 
        sleep = False, 
        get_bet = lambda cash: 1, #minimal bet size, the lambda is so its callable to avoid an error
        get_split_choice = hc.get_split_choice_hardcode, #split_choice_to_csv, 
        display = hc.display_nothing_hardcode, 
        get_hit_stand_dd = hc.get_hit_stand_dd_hardcode, #hit_stand_dd_to_csv, 
        display_hand = hc.display_nothing_hardcode, 
        display_emergency_reshuffle = hc.display_nothing_hardcode, 
        display_final_results = hc.display_nothing_hardcode
    )
    return

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
    print("Model Structure: ")
    # build structure: list of layers, each layer is a list of neurons (represented here as '.')
    neur_struct = [['.' for _ in range(n_p_layer)] for _ in range(n_layers)]

    for layer in neur_struct:
        # a simple visual: show layer number and the neuron list
        print(' '.join(layer))
    print()


    print("Choose your training hyperparamters")
    learning_rate = float(input("Learning Rate: "))
    epochs = int(input("Epochs: "))

    # I need to fix my torch version to make sure this is working right
    # device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    # print(f"Device: {device}")
    device = "cpu"

    class NeuralNetwork(nn.Module):
        def __init__(self, n_layers, n_p_layer):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential()
            
            layers = []
            for i in range(n_layers):
                if i == 0:
                    in_features=n_inputs
                    out_features = n_p_layer if n_layers > 1 else n_outputs
                elif i == n_layers - 1:
                    in_features=n_p_layer
                    out_features= n_outputs
                else: 
                    in_features=n_p_layer
                    out_features=n_p_layer

                layers.append(nn.Linear(in_features, out_features))

                if i != n_layers -1:
                    layers.append(nn.ReLU())

            self.linear_relu_stack = nn.Sequential(*layers)

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    model = NeuralNetwork(n_layers, n_p_layer).to(device)
    print(model)
    loss_fn = nn.CrossEntropyLoss()
    hsd_optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#train_model()