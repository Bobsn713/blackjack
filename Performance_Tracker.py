# I was getting import circularity errors so I'm making a separate Performance tracker file
import Logic as bj
import Hardcode as hc
import Text as text
import matplotlib.pyplot as plt
from Imitation import Imitation_Util as imit

def performance_tracker():
    iterations = 10000
    cash = 1000

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

    for i in range(iterations):     
        deck = bj.create_deck()
        bj.shuffle_deck(deck)

        return_dict = bj.play_round(
            cash = cash, #infinite cash relative to bet size
            deck = deck, 
            sleep = False, 
            get_bet = lambda cash: 1, #minimal bet size, the lambda is so its callable to avoid an error
            get_split_choice = imit.get_split_choice_imit, #hc.get_split_choice_hardcode, 
            display = hc.display_hardcode, 
            get_hit_stand_dd = imit.get_hit_stand_dd_imit, #hc.get_hit_stand_dd_hardcode, 
            display_hand = text.display_hand_print, # I think as long as the display function is empty this shouldn't print
            display_emergency_reshuffle = text.display_hand_print, #Ditto
            display_final_results = hc.display_hardcode
        )

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
    total_hands = len(hand_outcomes_clean)

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




    print(f"\nResults of {iterations} iterations...")
    print(f"Cumulative Outcome: {cumulative_cash_change} units")
    print(f"Expected Return (per hand): {expected_return_per_hand}")
    print()
    print(f"Won: {won}   Push: {push_incl}   Lost: {lost}")
    print(f"Player BJ: {p_bj}   BJ Push: {push_bj}   Dealer BJ: {d_bj}")
    print(f"Player Bust: {p_bust}   Dealer Bust: {d_bust}")
    print(f"Player Higher: {p_higher}   Dealer Higher: {d_higher}   Non-BJ Push: {push_excl}")
    print(f"Split Hands: {split_hands}   Doubled Down Hands: {double_downs}")
    print()
    print("Percentages:")
    print(f"Won: {perc_won}%   Push: {perc_push_incl}%   Lost: {perc_lost}%")
    print(f"Player BJ: {perc_p_bj}%   BJ Push: {perc_push_bj}%   Dealer BJ: {perc_d_bj}%")
    print(f"Player Bust: {perc_p_bust}%   Dealer Bust: {perc_d_bust}%")
    print(f"Player Higher: {perc_p_higher}%   Dealer Higher: {perc_d_higher}%   Non-BJ Push: {perc_push_excl}%")
    print(f"Split Hands: {perc_split_hands}%   Doubled Down Hands: {perc_double_downs}%")

    plt.plot(running_cash_total)
    plt.show()

#### Temp Fix ####
# I need access to the model definitions so I'm just copying them in here. 
from torch import nn
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
#####################

performance_tracker()