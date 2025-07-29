# Before I try to go full reinforcement learning, I will train a neural net to imitate the perfect player whose logic is stored in "Hardcode"
# The first step is to make a big csv file with lots of examples to train from. 
import Hardcode as hc
import Text as text
import csv

def get_split_choice_csv(player_hand, dealer_hand):
    result = hc.get_split_choice_hardcode(player_hand, dealer_hand)
    dealer_upcard = dealer_hand[0]

    print_result = 1 if result == "y" else 0

    with open('CSVs/split_or_not.csv', 'a', newline='') as csv_file:
        split_or_not = csv.writer(csv_file)
        split_or_not.writerow([player_hand, dealer_upcard, print_result])

    return result 

def get_hit_stand_dd_csv(player_hand, dealer_hand, can_double):
    result = hc.get_hit_stand_dd_hardcode(player_hand, dealer_hand, can_double)
    dealer_upcard = dealer_hand[0]

    with open('CSVs/hit_stand_dd.csv', 'a', newline='') as csv_file: 
        hit_stand_dd = csv.writer(csv_file)
        hit_stand_dd.writerow([player_hand, dealer_upcard, can_double, result])

    return result
