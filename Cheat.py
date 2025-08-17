import re

import Logic as bj
import Hardcode as hc

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

        if len(clean_hand) == 2: #I might make this more expansive. 
            run_again = False
        else: 
            print("Please enter a valid 2 card hand.")
            print("Cards can be entered as rank suit pairs, e.g. 'KD, 10C', '4d5s'")
            print("or simply as ranks, e.g. 'kk', '4, 9'")

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
        
    return clean_upcard


# ###################
# # THIS CODE RUNS, BUT I'M TRYING TO USE THE play_round() FUNCTION FROM LOGIC
# ###################

def play_round_cheat(): # Should I just be calling the play_round function from logic here ?
    hand = get_hand_cheat()
    d_upcard = get_dealer_upcard_cheat()
    print()

    if hand[0][0] == hand[1][0]: # This is nowhere near as robust as what I really want
        s_or_n = hc.get_split_choice_hardcode(hand, d_upcard) # Dealer's second card is ignored within the function
        if s_or_n == 'y':
            print('Split')
            # I need to add functionality for new hands
        else:
            print('Don\'t Split')

    print(hc.get_hit_stand_dd_hardcode(hand, d_upcard, True).capitalize()) 
    # figure out what to do with can_double
    # print prettier
        
# #play_round_cheat()

def play_round_cheat(): 
    hand = get_hand_cheat()
    d_upcard = get_dealer_upcard_cheat()
    print()

    dealer_hand = [d_upcard[0], ('2', 'H')] #Second card is fake, just so things don't break
    

    bj.play_round(1000, )