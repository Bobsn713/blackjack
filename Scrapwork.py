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




cards_to_add = list(reversed([
    ('8', 'H'), ('8', 'D'),   # Player initial hand (8,8)
    ('8', 'C'), ('8', 'S'),   # Cards dealt to first and second hands (or to dealer if no split)
    ('8', 'H'), ('8', 'D'),   # Further split hands
    ('10', 'H'), ('7', 'D')   # Dealer cards
]))

deck.extend(cards_to_add)



print(bj.play_round(
    cash = 1000,
    deck = deck,
    sleep = False,
    get_bet = lambda cash: 1,
    get_split_choice = text.get_split_choice_print,
    display = print,
    get_hit_stand_dd = text.get_hit_stand_dd_print,
    display_hand = text.display_hand_print,
    display_emergency_reshuffle = text.display_emergency_reshuffle_print
))


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
{'cash_change': 3, 
 'cash_changes': [1, 2, 1, -1], 
 'player_hands': [[('8', 'H'), ('10', 'H')], 
                  [('8', 'H'), ('7', 'D'), ('A', 'H')], 
                  [('8', 'D'), ('J', 'S')], 
                  [('8', 'D'), ('J', 'C'), ('Q', 'S')]], 
'dealer_hand': [('8', 'C'), ('8', 'S'), ('9', 'H')], 
'outcomes': ['Dealer Bust', 'Dealer Bust', 'Dealer Bust', 'Player Bust']}