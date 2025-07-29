# Before I try to go full reinforcement learning, I will train a neural net to imitate the perfect player whose logic is stored in "Hardcode"
# The first step is to make a big csv file with lots of examples to train from. 
import Logic as bj
import Hardcode as hc
import Text as text
import csv

def formatted_card(card):
    rank, suit = card[0], card[1]
    return f"{rank}{suit}"

def formatted_hand(hand):
    clean_hand = []
    for card in hand:
        clean_hand.append(formatted_card(card))

    return "-".join(clean_hand)


def get_split_choice_csv(player_hand, dealer_hand):
    result = hc.get_split_choice_hardcode(player_hand, dealer_hand)
    dealer_upcard = dealer_hand[0]

    print_result = 1 if result == "y" else 0

    with open('CSVs/split_or_not.csv', 'a', newline='') as csv_file:
        split_or_not = csv.writer(csv_file)
        split_or_not.writerow([formatted_hand(player_hand), formatted_card(dealer_upcard), print_result])

    return result 

def get_hit_stand_dd_csv(player_hand, dealer_hand, can_double):
    result = hc.get_hit_stand_dd_hardcode(player_hand, dealer_hand, can_double)
    dealer_upcard = dealer_hand[0]

    print_can_double = 1 if can_double == True else 0

    with open('CSVs/hit_stand_dd.csv', 'a', newline='') as csv_file: 
        hit_stand_dd = csv.writer(csv_file)
        hit_stand_dd.writerow([formatted_hand(player_hand), formatted_card(dealer_upcard), print_can_double, result])

    return result

def imitation_loop(iterations):
    for _ in range(iterations):     
        deck = bj.create_deck()
        bj.shuffle_deck(deck)

        bj.play_round(
            cash = 1000, #infinite cash relative to bet size
            deck = deck, 
            sleep = False, 
            get_bet = lambda cash: 1, #minimal bet size, the lambda is so its callable to avoid an error
            get_split_choice = get_split_choice_csv, 
            display = hc.display_hardcode, 
            get_hit_stand_dd = get_hit_stand_dd_csv, 
            display_hand = text.display_hand_print, # I think as long as the display function is empty this shouldn't print
            display_emergency_reshuffle = text.display_hand_print, #Ditto
            display_final_results = hc.display_hardcode
        )

imitation_loop(100000) #10,000