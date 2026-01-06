import re
import time 

import Logic as bj
import Hardcode as hc
import Text as txt
from base import GameState, GameInterface

# Players could have the option of having the computer track results (which would require them to input bet sizes and final dealer cards) or just recommend hand by hand with no continuity. 

def regex_parse_input(user_input): 
    ranks = re.findall(r'10|[AKQJ2-9]', user_input)

    pairs = re.findall(r'(10|[AKQJ2-9])([HSDC])', user_input)

    if not pairs and ranks: 
        # User entered ranks without suits
        return [(rank, 'ANY') for rank in ranks]
    
    return pairs
    
def find_and_remove_card(state: GameState, rank, suit):
    target_card = None

    for card in state.deck:
        if card[0] == rank and (suit=='ANY' or card[1]==suit):
            target_card = card
            break

    if target_card:
        state.deck.remove(target_card)
        state.cards_left[target_card[0]] -= 1
        return target_card
    else: 
        raise ValueError(f"No more {rank}'s left in the deck!")


def get_hand_cheat(): 
    run_again = True
    while run_again:
        raw_hand = input("Player Hand: ").upper()
        clean_hand = regex_parse_input(raw_hand)     

        if len(clean_hand) == 2: 
            run_again = False
        else: 
            print()
            print("Please enter a valid 2 card hand.")
            print("Cards can be entered as rank suit pairs, e.g. 'KD, 10C', '4d5s'")
            print("or simply as ranks, e.g. 'kk', '4, 9'")
            print()

    return clean_hand

def get_split_choice_cheat(player_hand, dealer_hand, ui):
    result = hc.get_split_choice_hardcode(player_hand, dealer_hand, ui)

    ui.wait(1) # Brief delay after entry

    if result == 'y':
        print()
        print("** RECOMMENDATION: SPLIT **")
        print()
        print("Play each hand one at a time")
    else:
        print()
        print("** RECOMMENDATION: DON'T SPLIT **")

    print()
    return result

def get_hit_stand_dd_cheat(player_hand, dealer_hand, can_double, ui):
    result = hc.get_hit_stand_dd_hardcode(player_hand, dealer_hand, can_double, ui)

    ui.wait(1) # Brief delay after the split decision

    print()
    print(f"** RECOMMENDATION: {result.upper()}**")
    print()
    return result

def get_card_cheat(state: GameState, ui: GameInterface, msg):
    msg_prompt = {
        'hit' : 'Card After Hit: ', 
        'dd' : 'Card After Double Down:',
        'splithand1' : 'First Card after Split: ', # Should probably display which hand this goes on
        'splithand2' : 'Second Card After Split: ', # Ditto
        'dhit' : "Dealer Card After Hit: ", 
        ###
        'dhand1' : 'Dealer Upcard: ', 
        'dhand2' : 'Dealer Second Card: ',
        'bjcheck' : 'Dealer Blackjack? (if yes, enter card, to skip press Enter): '  # This needs to be cleaner
    }

    if msg == 'phand': 
        print()
        hand = get_hand_cheat()
        # Eventually I will need something like this, but ideally more robust to handle suitless input: 
        # deck.pop(hand[0])
        # deck.pop(hand[1])
        
        cleaned_hand = []
        for rank, suit in hand: 
            cleaned_hand.append(find_and_remove_card(state, rank, suit))

        return cleaned_hand
    
    if msg == 'dhand?':
        return ('?', '?')
    
    if msg == 'bjcheck': 
        while True: 
            raw_input = input(msg_prompt[msg]).upper()
            if not raw_input: # User pressed enter
                return ('?', '?')
            
            clean_cards = regex_parse_input(raw_input)
            if len(clean_cards) == 1: 
                clean_card = clean_cards[0]
                rank, suit = clean_card

                card = find_and_remove_card(state, rank, suit)
                return card
            else: 
                print("Invalid input. Enter a single valid card or press Enter to skip.")

    while True: 
        upcard_raw = input(msg_prompt[msg]).upper()
        clean_cards = regex_parse_input(upcard_raw)

        if len(clean_cards) == 1:
            clean_card = clean_cards[0]
            break
        else: 
            print()
            print("Please enter a valid, single card.")
            print("Cards can be entered as rank suit pairs, e.g. 'KD', '4d'")
            print("or simply as ranks, e.g. 'k', '4'")
            print()

        #As above, will need eventually but I need to make it more robust
        #deck.pop(clean_card[0])
        
    rank, suit = clean_card
    card = find_and_remove_card(state, rank, suit)
    return card

def play_round_cheat():
    deck = bj.create_deck()
    bj.shuffle_deck(deck)

    bj.play_round(
        cash = 1000, 
        deck = deck,
        sleep = True, 
        
        display = print, 
        display_hand = txt.display_hand_print, 
        display_emergency_reshuffle = txt.display_emergency_reshuffle_print,
        display_final_results       = txt.display_final_results_print, 

        get_bet                     = txt.get_bet_print,
        get_split_choice            = get_split_choice_cheat,
        get_hit_stand_dd            = get_hit_stand_dd_cheat, 
        get_card                    = get_card_cheat, 
    )

def play_game_cheat():
    bj.play_game(
        sleep                        = True,
        
        display                      = print,
        display_hand                 = txt.display_hand_print,
        display_emergency_reshuffle  = txt.display_emergency_reshuffle_print,
        display_final_results        = txt.display_final_results_print,

        get_another_round            = txt.get_another_round_print,
        get_bet                      = txt.get_bet_print,
        get_split_choice             = get_split_choice_cheat,
        get_hit_stand_dd             = get_hit_stand_dd_cheat,
        get_card                     = get_card_cheat,
    )

if __name__ == "__main__":
    state = GameState()
    cheat_mode = GameInterface(
    # Display functions
    display                     = print,
    display_hand                = txt.display_hand_print,
    display_emergency_reshuffle = txt.display_emergency_reshuffle_print,
    display_final_results       = txt.display_final_results_print,

    # Get Functions
    get_bet                      = txt.get_bet_print,
    get_card                     = get_card_cheat,
    get_split_choice             = get_split_choice_cheat,
    get_hit_stand_dd             = get_hit_stand_dd_cheat,
    get_another_round            = txt.get_another_round_print,

    # Other
    sleep = True
    )
    bj.play_game(state, cheat_mode)