# Before I try to go full reinforcement learning, I will train a neural net to imitate the perfect player whose logic is stored in "Hardcode"
# The first step is to make a big csv file with lots of examples to train from. 
import os
import sys

import Logic as bj
import Hardcode as hc
import Text as text
import csv

import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

csv_dir = os.path.join(current_dir, 'CSVs')

# Two functions to help with reformatting for CSV
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

def add_to_csv(iterations):
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
            # get_card = bj.get_card_deal, 
            display = hc.display_nothing_hardcode, 
            get_hit_stand_dd = hc.get_hit_stand_dd_hardcode, #hit_stand_dd_to_csv, 
            display_hand = text.display_hand_print, # I think as long as the display function is empty this shouldn't print
            display_emergency_reshuffle = text.display_hand_print, #Ditto
            display_final_results = hc.display_nothing_hardcode
        )

#add_to_csv(100000)

