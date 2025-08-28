#Scrapwork for figuring out game states

#Initial Gamestate
initial_state = {
    'Player Hand' : ['card1', 'card2'],
    'Dealer Hand' : ['card1', 'card2'] # Card 2 will need to be hidden from the player
}

#Check for Blackjack

game_state_bj_check = {
    'Player Hand' : ['card1', 'card2'],
    'Dealer Hand' : ['card1', 'card2'], # Card 2 will need to be hidden from the player
    'Outcome_how' : 'Player Blackjack', # or 'Dealer Blackjack' or 'Both Blackjack' or 'Player Bust' or 'Dealer Bust' or
                                        # 'Player Higher Number' or 'Dealer Higher Number' (anything else?)
    'Outcome' : 'win', #or 'lose' or 'push' COULD THIS JUST BE DERIVED FROM THE ABOVE?
    'Payout' : '(+) bet * 1.5' #COULD THIS JUST BE DERIVED FROM THE ABOVE?
}

#If game continues....check for splits (iteratively)
# COME BACK TO THIS

game_state_split = {
    'Player Initial Hand' : ['card1', 'card2'],
    'Dealer Hand' : ['card1', 'card2'], # Card 2 will need to be hidden from the player
    'is_split' : True,
    'Split Hands' : [['card1', 'card3'], ['card2', 'card4']] 
}

#If no splits prompt player to hit, stand, (or double down: check if can for double down)
#If hit

game_state_hit = {
    'Player Hand' : ['card1', 'card2', 'card3'],
    'Dealer Hand' : ['card1', 'card2'], # Card 2 will need to be hidden from the player

}

#Check for bust, ask again
# (Double down is just add a card 3 and then do the stand procedure below)

#If stand reveal card 2, do dealer decision tree, dealer decision tree returns outcome
game_state_hit = {
    'Player Hand' : ['card1', 'card2'],
    'Dealer Hand' : ['card1', 'card2', 'card3', 'card4'], # Reveal Card 2
    'Outcome how' : 'Dealer Bust',
    #....outcome
    #...payout
}



#So taking a step back, for each hand, including split hands, I should probably have something like 
# The only other thing I think it might make sense to keep track of is like action history?


hand = {
    'cards' : ['card1', 'card2', 'card3'],
    'bet' : 30, 
    'status': 'active', # or stand, bust, something,
    'doubled_down' : False, 
    'is_split_hand' : True, #Idrk why I need this but
    'action_history' : ['action1, action2'] #How should this work??
}

round_state = {
    'dealer hand' : ['card1', 'card2'], # remember to hide when appropriate
    'player_hands' : [hand, hand], 
    'current_hand_index' : 1,  # Do I need this or will this get handled in other functions?
    'Outcome_how' : None, #But eventually Player Blackjack, Dealer Bust, etc.
    'Outcome' : None, #Win, loss, push (redundant?)
    'Payout' : None, #some number i guess
    'meta_action_history' : ['action1', hand['action_history']] #how should this work?
}







# Is_soft_total scrap work
# A, 2, 3   can be 16 or 6, is soft, should return (True, 5)
# # A, 2, 10  can be only 13, is hard, should return (False, 12? or 13?)
# hand_1 = [('A', 'H'), ('2', 'H'), ('3', 'H')]
# hand_2 = [('A', 'H'), ('2', 'H'), ('10', 'H')]

# import Logic as bj

# bj.hand_value(hand_1) #16
# bj.hand_value(hand_2) #13

# hand_ranks = []
# for card in hand:
#     hand_ranks.append(bj.card_value(card))

# for card in hand_ranks: 
#     if sum(hand_ranks) > 21 and card == 11:
#         card = 1

# if 11 in hand_ranks:
#     is_soft = True
#     soft_total = sum(hand_ranks) - 10
# else: 
#     is_soft = False
#     soft_total = sum(hand_ranks) #this is just the hard total


#GOT IT TO WORK!






# Doing some testing of what output looks like from different functions
import Logic as bj
import Text as text

deck = bj.create_deck()
bj.shuffle_deck(deck)

#If I need to customize the deck, steal the append code thats commented out in Logic

# print(bj.play_individual_hand(
#     hand = [('A', 'H'), ('10', 'H')],
#     deck = deck,
#     bet = 1,
#     cash = 1000,
#     dealer_hand = [('6', 'H'), ('10', 'H')],
#     get_hit_stand_dd = text.get_hit_stand_dd_print,
#     display = print, 
#     display_emergency_reshuffle = text.display_emergency_reshuffle_print,
#     display_hand = text.display_hand_print
# ))

#returns 
# dictionary{
#   'hand': [('A', 'H'), ('10', 'H'), ('Q', 'D')], 
#   'result': 'stand', 
#   'bet_multiplier': 1, 
#   'final': False
#}

# Hand is the hand at the conclusion of the player's turn
# Result possibilities are 'bust', 'stand', 'doubled'
# If you double down and bust, the result is 'bust' not 'double'
# Bet_multiplier is 1 unless you double down in which case it's 2
# Final is True if busted (no need to compare with dealer), False otherwise




# cards_to_add = list(reversed([
#     ('8', 'H'), ('8', 'D'),   # Player initial hand (8,8)
#     ('8', 'C'), ('8', 'S'),   # Cards dealt to first and second hands (or to dealer if no split)
#     ('8', 'H'), ('8', 'D'),   # Further split hands
#     ('10', 'H'), ('7', 'D')   # Dealer cards
# ]))

# deck.extend(cards_to_add)



# print(bj.play_round(
#     cash = 1000,
#     deck = deck,
#     sleep = False,
#     get_bet = lambda cash: 1,
#     get_split_choice = text.get_split_choice_print,
#     display = print,
#     get_hit_stand_dd = text.get_hit_stand_dd_print,
#     display_hand = text.display_hand_print,
#     display_emergency_reshuffle = text.display_emergency_reshuffle_print
# ))


# The following is the dictionary returned by play_round
# dictionary = {'cash_change': total_cash_change, 
#               'cash_changes': cash_changes, #list
#               'player_hands' : hand_results, #dictionary returned by play_individual_hand (UNLESS BLACKJACK, WILL HAVE TO FIX THIS)
#               'dealer_hand': dealer_hand, 
#               'outcomes' : outcomes #list
# }

#Notes: Cash change is redundant (can do sum of cash changes)
# Other potential rows: 
# 'win_loss_push' # would be redundant, all info in cash_changes and outcomes
# 'is_split' #would be redundant, could just do len(e.g. cash_changes)

#Here's a sample print from a max split hand
{
 'cash_changes': [1, 2, 1, -1], 
 'player_hands': [[('8', 'H'), ('10', 'H')], 
                  [('8', 'H'), ('7', 'D'), ('A', 'H')], 
                  [('8', 'D'), ('J', 'S')], 
                  [('8', 'D'), ('J', 'C'), ('Q', 'S')]], 
'dealer_hand': [('8', 'C'), ('8', 'S'), ('9', 'H')], 
'outcomes': ['Dealer Bust', 'Dealer Bust', 'Dealer Bust', 'Player Bust']}

#Possible Outcomes 'Blackjack Push', 'Dealer Blackjack', 'Player Blackjack', 'Player Bust', 'Dealer Bust', 'Player Higher', 'Dealer Higher', 'Push'




# This is the "play_normal_round" function that claude made me to keep current settings: 
def play_normal_round(cash, deck, sleep, display, display_hand, 
                     display_emergency_reshuffle, display_final_results):
    return play_round(
        cash=cash,
        deck=deck,
        sleep=sleep,
        display=display,
        display_hand=display_hand,
        display_emergency_reshuffle=display_emergency_reshuffle,
        display_final_results=display_final_results,
        get_bet=txt.get_bet_print,  # Your existing functions
        get_split_choice=txt.get_split_choice_print,
        get_hit_stand_dd=txt.get_hit_stand_dd_print
    )


# Here's what it says to do for play_round_cheat()
def play_cheat_round():
    # Get the cards from user input
    hand = get_hand_cheat()
    d_upcard = get_dealer_upcard_cheat()
    
    # Create a dummy second dealer card (required for the function)
    dealer_hand = [d_upcard[0], ('2', 'H')]  # Second card doesn't matter for strategy
    
    # Mock functions for cheat mode
    def mock_get_bet(cash):
        return 10  # Dummy bet
    
    def mock_display(text):
        pass  # Silent
    
    def mock_display_hand(hand, hidden=False):
        return str(hand)  # Simple display
    
    def mock_display_emergency_reshuffle():
        pass
    
    def mock_display_final_results(results):
        # Extract and display the recommendations
        for i, outcome in enumerate(results['outcomes']):
            if len(results['player_hands']) > 1:
                print(f"Hand {i+1}: {outcome}")
            else:
                print(f"Recommendation: {outcome}")
    
    # Create a dummy deck (won't be used since we're providing cards)
    dummy_deck = []
    
    return play_round(
        cash=1000,  # Dummy cash
        deck=dummy_deck,
        sleep=False,
        display=mock_display,
        display_hand=mock_display_hand,
        display_emergency_reshuffle=mock_display_emergency_reshuffle,
        display_final_results=mock_display_final_results,
        get_bet=mock_get_bet,
        get_split_choice=hc.get_split_choice_hardcode,  # Your cheat functions
        get_hit_stand_dd=hc.get_hit_stand_dd_hardcode,
        initial_hand=hand,  # Pre-provide the cards
        dealer_hand=dealer_hand
    )

# And I don't really understand this but this is a way it shows to simplify things
# Alternative: Even cleaner approach with strategy classes
class PlayStrategy:
    def get_bet(self, cash): raise NotImplementedError
    def get_split_choice(self, hand, dealer_hand): raise NotImplementedError  
    def get_hit_stand_dd(self, hand, dealer_hand, can_double): raise NotImplementedError

class HumanStrategy(PlayStrategy):
    def get_bet(self, cash): return txt.get_bet_print(cash)
    def get_split_choice(self, hand, dealer_hand): return txt.get_split_choice_print(hand, dealer_hand)
    def get_hit_stand_dd(self, hand, dealer_hand, can_double): return txt.get_hit_stand_dd_print(hand, dealer_hand, can_double)

class CheatStrategy(PlayStrategy):
    def get_bet(self, cash): return 10  # Dummy
    def get_split_choice(self, hand, dealer_hand): return hc.get_split_choice_hardcode(hand, dealer_hand)
    def get_hit_stand_dd(self, hand, dealer_hand, can_double): return hc.get_hit_stand_dd_hardcode(hand, dealer_hand, can_double)

# Usage would be:
# play_round_with_strategy(cash, deck, sleep, display_funcs, HumanStrategy())
# play_round_with_strategy(cash, deck, sleep, display_funcs, CheatStrategy(), 
#                         initial_hand=hand, dealer_hand=dealer_hand)



So right now I have the following ways to play: 
 - Text Mode (Full game)
 - Hardcode Mode (Full Game)
 - Hardcode Mode (Rounds) 

 - Performance Tracker (imit or hardcode)

 - Cheat Mode (Rounds only for now)


# Here's Gemini's idea for simplifying my dependency injection
class PlayStrategy:
    def get_bet(self, cash):
        raise NotImplementedError

    def get_split_choice(self, hand, dealer_hand):
        raise NotImplementedError

    def get_hit_stand_dd(self, hand, dealer_hand, can_double):
        raise NotImplementedError
    
    def get_card(self, deck, display_emergency_reshuffle):
        return deal_card(deck, display_emergency_reshuffle) # Default behavior

    def display(self, *args, **kwargs):
        print(*args, **kwargs)

    def display_hand(self, hand, hidden=False):
        return text.display_hand_print(hand, hidden)

    def display_emergency_reshuffle(self):
        text.display_emergency_reshuffle_print()

    def display_final_results(self, round_results):
        text.display_final_results_print(round_results)

########
class HumanStrategy(bj.PlayStrategy):
    def get_bet(self, cash):
        return txt.get_bet_print(cash)

    def get_split_choice(self, hand, dealer_hand):
        return txt.get_split_choice_print(hand, dealer_hand)

    def get_hit_stand_dd(self, hand, dealer_hand, can_double):
        return txt.get_hit_stand_dd_print(hand, dealer_hand, can_double)
    
class CheatStrategy(bj.PlayStrategy):
    def get_split_choice(self, player_hand, dealer_hand):
        return get_split_choice_cheat(player_hand, dealer_hand)

    def get_hit_stand_dd(self, player_hand, dealer_hand, can_double):
        return get_hit_stand_dd_cheat(player_hand, dealer_hand, can_double)

    def get_card(self, deck, display_emergency_reshuffle):
        return get_card_cheat(deck, display_emergency_reshuffle)
    
######
def play_round(
    # Game State
    cash,
    deck,
    sleep,
    strategy, # Pass the strategy object

    # Optional: Pre-dealt cards for cheat mode
    initial_hand = None,
    dealer_hand = None
    ):

    outcomes = [] 
    
    if initial_hand is None:
        initial_hand = [strategy.get_card(deck, strategy.display_emergency_reshuffle), strategy.get_card(deck, strategy.display_emergency_reshuffle)]
    if dealer_hand is None:
        dealer_hand = [strategy.get_card(deck, strategy.display_emergency_reshuffle), strategy.get_card(deck, strategy.display_emergency_reshuffle)]

    bet = strategy.get_bet(cash)

    strategy.display(f"\nPlayer hand: {strategy.display_hand(initial_hand)}")
    strategy.display(f"Dealer hand: {strategy.display_hand(dealer_hand, hidden=True)}")