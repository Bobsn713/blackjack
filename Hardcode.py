# Source for charts
# https://www.blackjackapprenticeship.com/blackjack-strategy-charts/
# Source for blackjack rules
# https://bicyclecards.com/how-to-play/blackjack 

# Import libraries
import random

# Help for formatting Printing
print("\n")

# Initialize some variables
cash = 1000
bet = 100

# Generate a deck of cards
suits = ['H', 'D', 'C', 'S']
ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
deck = [(rank, suit) for suit in suits for rank in ranks]

# At some point I will need to make this so it can use multiple decks
# Another good modification might be to allow for changing rules like if doubling down is allowed, etc. 
# For now we assume doubling down is allowed, as is splitting, and DAS, but no insurance or surrender (I think)
# and Dealer hits on a soft 17 (H17)


# Shuffle the deck
this_deck = deck.copy()
random.shuffle(this_deck)

# Deal two cards to the player and two to the dealer
player_hand = [this_deck.pop(), this_deck.pop()]
dealer_hand = [this_deck.pop(), this_deck.pop()]
dealer_upcard = dealer_hand[0]

# # Enter custom hand for testing
# player_hand = [('J', 'D'), ('10', 'S')]
# dealer_hand = [('Q', 'H'), ('7', 'D')]



# # Print the hand unformatted (so you can see the datatypes)
# print("Player's Hand:", player_hand)
# print("Dealer's Hand:", dealer_hand)

# Print the hands (what the player would be able to see)
print("Cash:", cash)
print("Bet:", bet)
print("Player's Hand (formatted):", f"{player_hand[0][0]}{player_hand[0][1]}, {player_hand[1][0]}{player_hand[1][1]}")
print("Dealer's Hand (formatted):", f"{dealer_upcard[0]}{dealer_upcard[1]}, X")

# A bit of "pre-processing" to just get the relevant information
def preprocessing(hand):
    processed_hand = []
    
    for card in hand:
        card_value = card[0]
        if card_value in ['J', 'Q', 'K']:
            processed_hand.append(10)
        elif card_value == 'A':
            processed_hand.append('A')
        else:
            processed_hand.append(int(card_value))

    return processed_hand

player_hand_processed = preprocessing(player_hand)
dealer_hand_processed = preprocessing(dealer_hand)
dealer_upcard_processed = dealer_hand_processed[0]

# Printing what the datatypes look like after processing
print("Player Hand (processed):", player_hand_processed)
print("Dealer Hand (procesed):", dealer_hand_processed)


# Trying to rewrite the hand value function in a way that makes more sense to me
# Basically if there are pairs or aces I'll do something special instead of making a total
# This will make it easier to sort into certain "Charts" below

# Note: Pass in the processed hands to this function
def chart_sorting_and_totals(hand):
    if hand[0] == hand[1]:
        return "Pair", hand[0]
    elif 'A' in hand:
        a_index = hand.index('A')
        other_index = a_index - 1 # In a two item list switches 1 to 0 and 0 to -1 (which is 1)

        if hand[other_index] == 10: #Checking for Blackjack off the deal
            return "Natural", None # Including None so the datatype is always a 2 value tuple
        else: 
            return "Soft", hand[other_index]
    else:
        return "Hard", hand[0] + hand [1]


player_chart = chart_sorting_and_totals(player_hand_processed)
dealer_chart = chart_sorting_and_totals(dealer_hand_processed)

# Printing the chart function for test
print("Player Chart:", player_chart)
print("Dealer Chart:", dealer_chart)

def check_naturals(player_chart, dealer_chart): # this may make sense to put inside of another function
    if player_chart[0] == "Natural":
            if dealer_chart[0] == "Natural":
                return "Push"
            else: 
                return "Blackjack"
    elif dealer_chart[0] == "Natural":
        return "Dealer Blackjack"
    else: 
        return "No Naturals"    # Maybe this should call another function, like the decision function
    
# Checking the Naturals function
print(check_naturals(player_chart, dealer_chart))

# Based on "Hard Totals" Chart
def hard_decision(player_chart, dealer_upcard_processed):
    player_total = player_chart[1]
    # Decision logic from chart
    if player_total >= 17:
        return "Stand"
    elif player_total >= 13 and dealer_upcard_processed in [2, 3, 4, 5, 6]:
        return "Stand"
    elif player_total == 12 and dealer_upcard_processed in [4, 5, 6]:
        return "Stand"
    elif player_total == 11:
        return "Double Down"
    elif player_total == 10 and dealer_upcard_processed not in [10, 'A']:
        return "Double Down"
    elif player_total == 9 and dealer_upcard_processed in [3, 4, 5, 6]:
        return "Double Down"
    else:
        return "Hit"
    
# Based on "Soft Totals" Chart
def soft_decision(player_chart, dealer_upcard_processed):
    non_a_card = player_chart[1]
    # Decision logic from chart
    if  non_a_card == 9:
        return "Stand"
    elif (non_a_card == 8) & (dealer_upcard_processed == 6):
        return "Double Down"
    elif non_a_card == 8:
        return "Stand"
    elif non_a_card == 7 & dealer_upcard_processed in [2, 3, 4, 5, 6]:  
        return "Double Down"
    elif non_a_card == 7 & dealer_upcard_processed in [7,8]:
        return "Stand"
    elif non_a_card == 6 & dealer_upcard_processed in [3, 4, 5, 6]:
        return "Double Down"
    elif (non_a_card in [4,5]) & (dealer_upcard_processed in [4, 5, 6]):
        return "Double Down"
    elif (non_a_card in [2,3]) & (dealer_upcard_processed in [5,6]):
        return "Double Down"
    else: 
        return "Hit"
    
# Based on Pair Splitting 
def split_decision(player_chart, dealer_upcard_processed):
    card = player_chart[1]
    if card == 'A' or card == 8:
        return "Split"
    elif card == 9 & (dealer_upcard_processed in [2, 3, 4, 5, 6, 8, 9]):
        return "Split"
    elif (card == 7 | card == 3 | card ==2) & dealer_upcard_processed >= 7:
        return "Split"
    elif card == 6 & dealer_upcard_processed >= 6:
        return "Split"
    elif card == 4 & dealer_upcard_processed in [5, 6]:
        return "Split"
    else: 
        print("Don't Split")  
        return hard_decision(('Hard', card*2), dealer_upcard_processed)
    
def decision(player_chart, dealer_upcard_processed):
    if player_chart[0] == "Pair": 
        return split_decision(player_chart, dealer_upcard_processed)
    elif player_chart[0] == "Soft":
        return soft_decision(player_chart, dealer_upcard_processed)
    elif player_chart[0] == "Hard":
        return hard_decision(player_chart, dealer_upcard_processed)
    else: 
        print("Something's gone wrong / Natural")
        
print(decision(player_chart, dealer_upcard_processed))


def stand(player_chart, dealer_chart):
    # Determining how many points the player has
    if player_chart[0] == "Hard":
        player_value = player_chart[1]
    elif player_chart[0] == "Soft":
        player_value = player_chart[1] + 11
        if player_value > 21:
            player_value -= 10
            #player_chart = "Hard", player_value # Is this right/ does it do what I need it to?
    
    # Determining how many points the dealer has
    if dealer_chart[0] == "Hard":
        dealer_value = dealer_chart[1]
    elif dealer_chart[0] == "Soft":
        dealer_value = dealer_chart[1] + 11
        if dealer_value > 21: 
            dealer_value -= 10

    print(f"Player: {player_value}, Dealer: {dealer_value}")
    # Determining what happens
    if player_value > 21:
        return "Bust"

    while dealer_value <= 17:
        dealer_hand.append(this_deck.pop())
        dealer_hand_processed = preprocessing(dealer_hand)
        print("Dealer Hit: ", dealer_hand_processed[-1])
        dealer_value += dealer_hand_processed[-1] # I think this will break if the dealer draws an Ace
    else: 
        if player_value > dealer_value: 
            print("Player Wins")
        elif dealer_value > 21:
            print("Dealer Busts")
            

    

print("Dealer's Hand (formatted):", f"{dealer_hand[0][0]}{dealer_hand[0][1]}, {dealer_hand[1][0]}{dealer_hand[1][1]}")

# Just a temporary solution
if decision(player_chart, dealer_upcard_processed) == "Stand":
    print(stand(player_chart, dealer_chart))

# More print formatting 
print("\n")