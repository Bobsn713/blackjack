import game_logic as gl
import basic_strategy as bs
import text as tx
import train_nn as tr
from base import GameState, GameInterface

import tqdm 
import time
import numpy as np 
from scipy.stats import t
import matplotlib.pyplot as plt

# 2 good ways to improve this:
# 1. Make it so that it can evaluate hardcode and imit simultaneously, on the same hands
# 2. Make it so that it can play through decks instead of just through rounds. This will eventually make it so I can factor in card-counting strategies. 

def performance_tracker(model, iterations = 10000):
    model = model.lower()

    #Cash Tracking Lists
    round_cash_changes = []
    hand_cash_changes_raw = [] #each entry is one round
    hand_cash_changes_clean = [] #no sub-lists for each round

    #Outcome Tracking Lists
    hand_outcomes_raw = [] #each entry is one round
    hand_outcomes_clean = [] #no sub_lists for each round

    #Split Tracking List
    is_split_round = []

    #Running Total
    running_cash_total = []

    if model == 'basic_strategy':
        get_split_choice = bs.get_split_choice_hardcode
        get_hit_stand_dd = bs.get_hit_stand_dd_hardcode
    elif model in tr.get_models(): # update this
        loaded_model = tr.load_model(model)
        get_split_choice = lambda p_hand, d_hand, ui: tr.get_split_choice_nn(p_hand, d_hand, loaded_model, ui)
        get_hit_stand_dd = lambda p_hand, d_hand, can_double, ui: tr.get_hit_stand_dd_nn(p_hand, d_hand, can_double, loaded_model, ui)
    else: 
        raise ValueError("Pass 'basic_strategy', 'sample_neural_net', or a custom '.pt' file to 'run_hardcode_mode()'")


    state = GameState(cash = iterations)
    simulation_mode = GameInterface(
        # Display functions
        display                     = bs.display_nothing_hardcode,
        display_hand                = bs.display_nothing_hardcode,
        display_emergency_reshuffle = bs.display_nothing_hardcode,
        display_final_results       = bs.display_nothing_hardcode,

        # Get Functions
        get_bet                      = lambda cash: state.bet,
        get_card                     = gl.get_card_deal,
        get_split_choice             = get_split_choice,
        get_hit_stand_dd             = get_hit_stand_dd,
        get_another_round            = lambda: 'y',

        # Other
        sleep = False
    )

    start_time = time.perf_counter()

    for i in tqdm.tqdm(range(iterations)):  
        deck_len = len(state.deck)
        reshuffle_size = int(deck_len / 4) 
        if len(state.deck) < reshuffle_size:
            gl.create_deck(state)

        # Could update bet dynamically by editing state.bet
        
        return_dict = gl.play_round(state, simulation_mode)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        #Making Cash Lists
        hand_cash_change = return_dict['cash_changes']
        hand_cash_changes_raw.append(hand_cash_change)
        hand_cash_changes_clean.extend(hand_cash_change)

        round_cash_change = sum(hand_cash_change)
        round_cash_changes.append(round_cash_change)

        #Making Outcome Lists
        hand_outcomes = return_dict['outcomes']
        hand_outcomes_raw.append(hand_outcomes)
        hand_outcomes_clean.extend(hand_outcomes)

        #Determining if hand is split
        is_split_round.append(len(hand_outcomes) > 1)

        #Making Running Total List
        if i == 0:
            running_cash_total.append(round_cash_change)
        else: 
            running_cash_total.append(running_cash_total[i-1] + round_cash_change)

        #Another thing to potentially add is keeping track of which hands are splits and doubles



    # Cumulative Stats
    cumulative_cash_change = sum(round_cash_changes)
    expected_return_per_hand = cumulative_cash_change / iterations
    std_dev = np.std(round_cash_changes, ddof=1)
    variance = std_dev ** 2
    total_hands = len(hand_outcomes_clean)
    std_error = std_dev / np.sqrt(total_hands)


    confidence = 0.95
    alpha = 1 - confidence 
    t_crit = t.ppf(1 - alpha/2, df=total_hands-1)
    ci_low = expected_return_per_hand - t_crit * std_error
    ci_high = expected_return_per_hand + t_crit * std_error


    #Outcome Counters
    p_bj = hand_outcomes_clean.count('Player Blackjack')
    push_bj = hand_outcomes_clean.count('Blackjack Push')
    d_bj = hand_outcomes_clean.count('Dealer Blackjack')
    p_bust = hand_outcomes_clean.count('Player Bust')
    d_bust = hand_outcomes_clean.count('Dealer Bust')
    p_higher = hand_outcomes_clean.count('Player Higher')
    d_higher = hand_outcomes_clean.count('Dealer Higher')
    push_excl = hand_outcomes_clean.count('Push') #not including push_bjs

    #W/L/P Counters
    won = p_bj + d_bust + p_higher
    lost = d_bj + p_bust + d_higher
    push_incl = push_bj + push_excl #including push_bjs

    #How do I want to do splits and double downs? 
    #Do I want stats for within those?
    split_hands = len(hand_outcomes_clean) - len(hand_outcomes_raw)
    double_downs = hand_cash_changes_clean.count(2) + hand_cash_changes_clean.count(-2)
    

    #Percents with demoninator total_hands
    perc_won = round((won / total_hands) * 100, 2)
    perc_push_incl= round((push_incl/ total_hands) * 100, 2)
    perc_lost = round((lost / total_hands) * 100, 2)

    perc_p_bj = round((p_bj / total_hands) * 100, 2) 
    perc_push_bj = round((push_bj / total_hands) * 100, 2)
    perc_d_bj = round((d_bj / total_hands) * 100, 2)
    perc_p_bust = round((p_bust / total_hands) * 100, 2)
    perc_d_bust = round((d_bust / total_hands) * 100, 2)
    perc_p_higher = round((p_higher / total_hands) * 100, 2)
    perc_d_higher = round((d_higher / total_hands) * 100, 2)
    perc_push_excl = round((push_excl / total_hands) * 100, 2)

    perc_split_hands = round((split_hands / total_hands) * 100, 2)
    perc_double_downs = round((double_downs / total_hands) * 100, 2)


    print()
    tx.print_title_box(["RESULTS"])
    print(f"\033[3m{iterations:,} hands simulated in {elapsed_time:.4f} seconds...\033[0m")
    print()
    print(f"Cumulative Outcome: {cumulative_cash_change} units")
    print(f"Expected Return (per hand): {expected_return_per_hand}")
    print(f"Confidence: {confidence*100}%    ({ci_low:.4}, {ci_high:.4})")
    print(f"Standard Deviation: {std_dev:.5}    Variance: {variance:.5}")
    print()
    print("Totals:")
    print('-'*40)
    print(f"Won: {won}   Push: {push_incl}   Lost: {lost}")
    print(f"Player BJ: {p_bj}   BJ Push: {push_bj}   Dealer BJ: {d_bj}")
    print(f"Player Bust: {p_bust}   Dealer Bust: {d_bust}")
    print(f"Player Higher: {p_higher}   Dealer Higher: {d_higher}   Non-BJ Push: {push_excl}")
    print(f"Split Hands: {split_hands}   Doubled Down Hands: {double_downs}")
    print()
    print("Percentages:")
    print('-'*40)
    print(f"Won: {perc_won}%   Push: {perc_push_incl}%   Lost: {perc_lost}%")
    print(f"Player BJ: {perc_p_bj}%   BJ Push: {perc_push_bj}%   Dealer BJ: {perc_d_bj}%")
    print(f"Player Bust: {perc_p_bust}%   Dealer Bust: {perc_d_bust}%")
    print(f"Player Higher: {perc_p_higher}%   Dealer Higher: {perc_d_higher}%   Non-BJ Push: {perc_push_excl}%")
    print(f"Split Hands: {perc_split_hands}%   Doubled Down Hands: {perc_double_downs}%")

    plt.plot(running_cash_total)
    plt.show()

def choose_model():
    while True: 
        print()
        print("Which model would you like to use?")
        print("Defaults: basic_strategy, sample_neural_net")
        custom_models = tr.get_models()
        display_names = [name.split('.')[0] for name in custom_models]
        display_names.remove('sample_neural_net')
        names_string = ", ".join(display_names)
        
        if display_names:
            print(f"Custom Models: {names_string}")
        else: 
            print("No custom models yet...")
        
        model = input(">>> ").lower() 
        print()

        if model == "basic_strategy":
            break
        elif model in custom_models:
            break
        elif model in display_names or model == 'sample_neural_net':
            model = f"{model}.pt"
            break
        else: 
            print("Please enter one of the options listed.")

    while True: 
        print("How many hands would you like to simulate?")
        try: 
            iterations = int(input(">>> "))
            print()
            break
        except: 
            print()
            print("Please enter an integer.")
            print()

    performance_tracker(model,iterations)

    print()
    print()
    print('\033[3mSimulation Complete.\033[0m')
    print('-'*80)
    print()

if __name__ == "__main__":
    performance_tracker('basic_strategy', iterations = 100000)