# This first version has no chatGPT input except for a few Copilot autocompletes at the beginning
# It's functionality is restricted to Hit and Stay logic. 

# Import libraries
import random

# Help for formatting Printing
print("\n")

# Generate a deck of cards
suits = ['H', 'D', 'C', 'S']
ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
deck = [(rank, suit) for suit in suits for rank in ranks]

this_deck = deck.copy()
random.shuffle(this_deck)

# Initialize some variables
cash = 1000
bet = 0
def get_bet(cash=cash):
    while True:
        print("How much would you like to bet?")
        bet = int(input(">>> "))

        if bet > cash:
            print("You don't have enough money for that bet. \n")#leaving secret way to beat game (negative bet, lose on purpose)
        else:
            cash -= bet
            return bet, cash

print("Cash: ", cash)
print()
bet, cash = get_bet()



# Deal two cards to the player and two to the dealer
player_hand = [this_deck.pop(), this_deck.pop()]
dealer_hand = [this_deck.pop(), this_deck.pop()]
dealer_upcard = dealer_hand[0]

# Print the hands (what the player would be able to see)

def print_status(cash, bet, player_hand, dealer_upcard, reveal = False):
    print("Cash:", cash)
    print("Bet:", bet)
    print()
    display_p_hand = [card[0]+card[1] for card in player_hand]
    display_d_hand = [card[0]+card[1] for card in dealer_hand]
    print("Player's Hand:", ', '.join(display_p_hand))
    if reveal == False: 
        print("Dealer's Hand:", f"{dealer_upcard[0]}{dealer_upcard[1]}, X")
    else: 
        print("Dealer's Hand: ", ', '.join(display_d_hand))

print_status(cash, bet, player_hand, dealer_upcard)

def preprocessing(hand):
    processed_hand = []
    
    for card in hand:
        if isinstance(card, tuple):
            card_value = card[0]
            if card_value in ['J', 'Q', 'K']:
                processed_hand.append(10)
            elif card_value == 'A':
                processed_hand.append('A')
            else:
                processed_hand.append(int(card_value))
        else:
            # This is to handle the case where the hand might already be processed
            processed_hand.append(card)
    return processed_hand


def ace_numerator(hand):
    processed_hand = preprocessing(hand)
    ace_counter = 0
    total = 0
    for card in processed_hand:
        if card == "A":
            total += 11
            ace_counter += 1
        else: 
            total += card
    
    return total, ace_counter

def ace_adjuster(ace_tuple):
    total, ace_counter = ace_tuple
    
    if total > 21:
        if ace_counter > 0:
            total -= 10
            ace_counter -= 1
            return ace_adjuster((total, ace_counter))
        else: 
            return "Bust"
    else: 
        return total

def is_bust(hand):
    return ace_adjuster(ace_numerator(hand))

def resolve(p_hand, d_hand):
    bust = is_bust(d_hand)
    if bust == "Bust":
        print("Dealer Busts")
        return "Player Wins"
    elif bust <= 17:
        dealer_hand.append(this_deck.pop())
        display_d_hand = [card[0]+card[1] for card in dealer_hand]
        print("Dealer Hits: ", ', '.join(display_d_hand))
        return resolve(p_hand, dealer_hand)
    else:
        print("Dealer Stays with: ", bust)

        if bust > is_bust(p_hand):
            return "Dealer Wins"
            
        elif bust < is_bust(p_hand):
            return "Player Wins"
        else: 
            return "Push"

def hit_or_stay():
    print("Hit or Stay?")
    h_o_s = input(">>> ")
    print()
    if h_o_s == "Hit":
        player_hand.append(this_deck.pop())
        print_status(cash, bet, player_hand, dealer_upcard)
        return "Bust" if is_bust(player_hand) == "Bust" else hit_or_stay()
    elif h_o_s == "Stay":
        print_status(cash, bet, player_hand, dealer_upcard, reveal = True)
        return resolve(preprocessing(player_hand), preprocessing(dealer_hand))
    else:
        print("Type either \"Hit\" or \"Stay\"")
        return hit_or_stay()


print()
#print(hit_or_stay())

def cash_update(cash, bet):
    result = hit_or_stay()

    if result == "Bust":
        print("Bust")
        bet = 0
        return cash, bet 
    elif result == "Dealer Wins":
        print("Dealer Wins")
        bet = 0
        return cash, bet
    elif result == "Player Wins":
        print("Player Wins")
        cash += bet * 2
        bet = 0 
        return cash, bet
    elif result == "Push":
        print("Push")
        cash += bet
        bet = 0
        return cash, bet
    else: 
        return "Error"
    
cash, bet = cash_update(cash, bet)

print("\nCash: ", cash)
