# Before I try to go full reinforcement learning, I will train a neural net to imitate the perfect player whose logic is stored in "Hardcode"
# The first step is to make a big csv file with lots of examples to train from. 
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import Logic as bj
import Hardcode as hc
import Text as text
import csv

import torch


csv_dir = os.path.join(current_dir, 'CSVs')

def formatted_card(card):
    rank, suit = card[0], card[1]
    return f"{rank}{suit}"

def formatted_hand(hand):
    clean_hand = []
    for card in hand:
        clean_hand.append(formatted_card(card))

    return "-".join(clean_hand)


def split_choice_to_csv(player_hand, dealer_hand):
    result = hc.get_split_choice_hardcode(player_hand, dealer_hand)
    dealer_upcard = dealer_hand[0]

    print_result = 1 if result == "y" else 0

    with open(os.path.join(csv_dir, 'split_or_not.csv'), 'a', newline='') as csv_file:
        split_or_not = csv.writer(csv_file)
        split_or_not.writerow([formatted_hand(player_hand), formatted_card(dealer_upcard), print_result])

    return result 

def hit_stand_dd_to_csv(player_hand, dealer_hand, can_double):
    result = hc.get_hit_stand_dd_hardcode(player_hand, dealer_hand, can_double)
    dealer_upcard = dealer_hand[0]

    print_can_double = 1 if can_double else 0

    with open(os.path.join(csv_dir, 'hit_stand_dd.csv'), 'a', newline='') as csv_file: 
        hit_stand_dd = csv.writer(csv_file)
        hit_stand_dd.writerow([formatted_hand(player_hand), formatted_card(dealer_upcard), print_can_double, result])

    return result

def imitation_loop(iterations):
    # By replacing get_split_choice and get_hit_stand_dd from their hardcode defaults to the local csv versions,
    # this function generates csv data. 

    #I could make it a little prettier, and put in some if statements to make it clearer what was going on
    for _ in range(iterations):     
        deck = bj.create_deck()
        bj.shuffle_deck(deck)

        bj.play_round(
            cash = 1000, #infinite cash relative to bet size
            deck = deck, 
            sleep = False, 
            get_bet = lambda cash: 1, #minimal bet size, the lambda is so its callable to avoid an error
            get_split_choice = hc.get_split_choice_hardcode, #split_choice_to_csv, 
            display = hc.display_hardcode, 
            get_hit_stand_dd = hc.get_hit_stand_dd_hardcode, #hit_stand_dd_to_csv, 
            display_hand = text.display_hand_print, # I think as long as the display function is empty this shouldn't print
            display_emergency_reshuffle = text.display_hand_print, #Ditto
            display_final_results = hc.display_hardcode
        )

#imitation_loop(100000) #10,000

##### COPIED FROM Jupyter Notebooks
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
###########################


def get_split_choice_imit(player_hand, dealer_hand):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "sn_imit_nn.pt")
    split_model = torch.load(model_path, map_location=torch.device('cpu'))
    split_model.eval()

    formatted_p_hand = formatted_hand(player_hand)
    formatted_d_card = formatted_card(dealer_hand[0])

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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "hsd_imit_nn.pt")
    hsd_model = torch.load(model_path, map_location=torch.device('cpu'))
    hsd_model.eval()

    formatted_p_hand = formatted_hand(player_hand)
    formatted_d_card = formatted_card(dealer_hand[0])

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

