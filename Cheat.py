import re
import time 

import Logic as bj
import Hardcode as hc
import Text as txt

# Players could have the option of having the computer track results (which would require them to input bet sizes and final dealer cards) or just recommend hand by hand with no continuity. 

def regex_parse_input(user_input): 
    # If the user didn't type any suits, just give it a heart
    if all(suit not in user_input for suit in bj.suits):
        ranks_input = re.findall(r'10|[AKQJ2-9]', user_input)
        output = [(rank, 'H') for rank in ranks_input]
    else: 
        output = re.findall(r'(10|[AKQJ2-9])([HSDC])', user_input) 
    return output

    # I could try and edit this so that if one card is given a suit and one isn't it accepts it as valid, but that is not a priority at all 

def get_hand_cheat(): 
    run_again = True
    while run_again:
        raw_hand = input("Input your hand: ").upper()
        clean_hand = regex_parse_input(raw_hand)     

        if len(clean_hand) >= 2: 
            run_again = False
        else: 
            print("Please enter a valid 2 card hand.")
            print("Cards can be entered as rank suit pairs, e.g. 'KD, 10C', '4d5s'")
            print("or simply as ranks, e.g. 'kk', '4, 9'")
            print()

    return clean_hand

def get_dealer_upcard_cheat():
    run_again = True
    while run_again: 
        upcard_raw = input("Input the Dealer's Upcard: ").upper()
        clean_upcard = regex_parse_input(upcard_raw)

        if len(clean_upcard) == 1:
            run_again = False
        else: 
            print("Please enter a valid, single card.")
            print("Cards can be entered as rank suit pairs, e.g. 'KD', '4d'")
            print("or simply as ranks, e.g. 'k', '4'")
            print()
        
    return clean_upcard

# This has a number of problems/points to improve upon (listed in Welcome.py)
def primitive_play_round_cheat(): # Should I just be calling the play_round function from logic here ?
    hand = get_hand_cheat()
    d_upcard = get_dealer_upcard_cheat()
    print()

    if hand[0][0] == hand[1][0]: # Is this robust enough?
        s_or_n = hc.get_split_choice_hardcode(hand, d_upcard) # Dealer's second card is ignored within the function
        if s_or_n == 'y':
            print('Split')
            print('\n(treat each new hand independently)')
            pass
        

    hsd = hc.get_hit_stand_dd_hardcode(hand, d_upcard, True)
    print()
    print(hsd.capitalize())

def get_split_choice_cheat(player_hand, dealer_hand):
    result = hc.get_split_choice_hardcode(player_hand, dealer_hand)

    time.sleep(1) # Brief delay after entry

    if result == 'y':
        print("Split")
        print()
        print("Play each hand one at a time")
    else:
        print("Don't Split")

    print()
    return result

def get_hit_stand_dd_cheat(player_hand, dealer_hand, can_double):
    result = hc.get_hit_stand_dd_hardcode(player_hand, dealer_hand, can_double)

    time.sleep(1) # Brief delay after the split decision

    print(result.capitalize())
    print()
    return result






# The Logic play_round version
# def play_round_cheat(): 
#     hand = get_hand_cheat()
#     d_upcard = get_dealer_upcard_cheat()
#     dealer_hand = [d_upcard[0], ('2', 'H')] #Second card is fake, just so things don't break

#     # There is an equivalent statement int Logic's play_round() function but it's a headache to use right now. 
#     print(f"\nPlayer hand: {txt.display_hand_print(hand)}")
#     print(f"Dealer hand: {txt.display_hand_print(dealer_hand, hidden=True)}")
#     print()

#     return bj.play_round(
#             # Game State
#             cash = 1000, #dummy cash
#             deck = [], #dummy deck      
#             sleep = False,

#             # Display Functions
#             display = hc.display_nothing_hardcode, 
#             display_hand = hc.display_nothing_hardcode, 
#             display_emergency_reshuffle = hc.display_nothing_hardcode, 
#             display_final_results =hc.display_nothing_hardcode,

#             # Get Functions
#             get_bet = lambda cash: 1, 
#             get_split_choice = get_split_choice_cheat, 
#             get_hit_stand_dd = get_hit_stand_dd_cheat, 

#             # Optional: Pre-dealt cards for cheat mode
#             initial_hand = hand,
#             dealer_hand = dealer_hand
#             )

if __name__ == "__main__":
    primitive_play_round_cheat()