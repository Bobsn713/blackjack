import os
import torch 
from torch import nn
from Build_Imitation import Build_Imitation_Util as biu #we have to copy the csv formatting to load into the nn


# HELPER FUNCTIONS/DEFINITIONS

# Model Definitions copied from Build_Imitation/sn_ntbk and Build_Imitation/hs_ntbk
# I feel like technically I should make this dynamic but I don't really want to figure that out right now. 
# I should also potentially do this whole thing as model weights

class sn_NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(26, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
class hsd_NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(105, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
##### COPIED FROM Jupyter Notebooks
#(So these should probably get defined somewhere else to get rid of the redundancy)
ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
rank_to_index = {rank: i for i, rank in enumerate(ranks)}
rank_len = len(ranks)

#Helper functions for data processing
def card_to_vec(card):
    raw_rank = card[:-1]
    one_hot_vector = [0] * rank_len
    one_hot_vector[rank_to_index[raw_rank]] = 1
    return one_hot_vector

def hand_to_list(hand):
    '''Takes hand like KH-AC and outputs list of card numbers'''
    hand_list_1 = hand.split("-")
    hand_list_2 = [card_to_vec(card) for card in hand_list_1]
    return hand_list_2

# GET FUNCTIONS

def get_split_choice_imit(player_hand, dealer_hand):
    model_path = "Build_Imitation/sn_imit_weights.pt"
    split_model = sn_NeuralNetwork()
    split_model.load_state_dict(torch.load(model_path))
    split_model.eval()

    formatted_p_hand = biu.formatted_hand(player_hand)
    formatted_d_card = biu.formatted_card(dealer_hand[0])

    #This feels pretty redundant because it does a lot of what the cleanup function does
    # But the cleanup function needs Xs and ys and here we have Xs and are trying to get Ys
    vec_d_card = card_to_vec(formatted_d_card)
    list_p_hand = hand_to_list(formatted_p_hand)
    vec_p_card = list_p_hand[0]

    x1 = torch.tensor(vec_p_card, dtype=torch.float32).unsqueeze(0)
    x2 = torch.tensor(vec_d_card, dtype=torch.float32).unsqueeze(0)
    X = torch.cat([x1,x2], dim=1)

    with torch.no_grad():
        raw_result = split_model(X)
        result = (torch.sigmoid(raw_result) > 0.5).item()

    return "y" if result else "n"

def get_hit_stand_dd_imit(player_hand, dealer_hand, can_double):
    model_path = "Build_Imitation/hsd_imit_weights.pt"
    hsd_model = hsd_NeuralNetwork()
    hsd_model.load_state_dict(torch.load(model_path))
    hsd_model.eval()

    formatted_p_hand = biu.formatted_hand(player_hand)
    formatted_d_card = biu.formatted_card(dealer_hand[0])

    #As above, much of this is copied from cleanup function in hsd_ntbk
    max_len = 7
    vec_d_card = card_to_vec(formatted_d_card)
    list_p_hand = hand_to_list(formatted_p_hand)

    zero_vector = [0] * rank_len

    if len(list_p_hand) < max_len:
        padded_hand = list_p_hand + [zero_vector] * (max_len - len(list_p_hand))
    else: 
        padded_hand = list_p_hand[:max_len]

    flattened_hand = [item for card_vec in padded_hand for item in card_vec]

    x1 = torch.tensor(flattened_hand, dtype=torch.float32).unsqueeze(0)
    x2 = torch.tensor(vec_d_card, dtype=torch.float32).unsqueeze(0)
    x3 = torch.tensor(can_double, dtype=torch.float32).unsqueeze(0).unsqueeze(1)

    X = torch.cat([x1,x2,x3], dim=1)

    with torch.no_grad():
        raw_result = hsd_model(X)
        result = torch.argmax(raw_result, dim=1).item()

    if result == 0:
        return "hit"
    elif result == 1:
        return "stand"
    elif result == 2:
        return "double down"
    else:
        return "Something went wrong"