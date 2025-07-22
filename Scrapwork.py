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