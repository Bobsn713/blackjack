# This version is reorganized with ChatGPT input
# It can handle the logic of Hit, Stand, and Double Down100

# Import libraries
import random

# Help for formatting Printing
print("\n")

# Generate a deck of cards
suits = ['H', 'D', 'C', 'S']
ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
starting_cash = 1000

def create_deck():
    return [(rank, suit) for suit in suits for rank in ranks]

def shuffle_deck(deck):
    return random.shuffle(deck)

def deal_card(deck):
    return deck.pop()

def hand_value(hand):
    total = 0
    aces = 0 
    
    for rank, suit in hand: 
        if rank in ['J', 'Q', 'K']:
            total += 10
        elif rank == "A":
            total += 11
            aces += 1
        else:
            total += int(rank)

    while total > 21 and aces > 0:
        total -= 10
        aces -= 1

    return total

def display_hand(hand, hidden=False):
    if hidden == False:
        return ', '.join(f'{rank}{suit}' for rank, suit in hand)
    else:
        return f'{hand[0][0]}{hand[0][1]}, [X]'
    
def get_bet(cash):
    while True:
        try:
            print("Cash: ", cash)
            bet = int(input("Bet: "))
            if bet <= 0:
                print("Bet must be positive.")
            elif bet > cash:
                print("You don't have enough money for that bet.")
            else:
                return bet
        except ValueError:
            print("Please enter a valid number.")

def play_hand(cash):
    deck = create_deck()
    shuffle_deck(deck)

    player_hand = [deal_card(deck), deal_card(deck)]
    dealer_hand = [deal_card(deck), deal_card(deck)]

    #So each hand is in the form e.g. [('A', 'C'), ('5', 'S')]

    bet = get_bet(cash)

    print("Player hand:", display_hand(player_hand))
    print("Dealer hand:", display_hand(dealer_hand, hidden=True))

    #Check for Blackjack 
    if hand_value(player_hand) == 21: #Player only has 2 cards at this point so this is an effective check
        if hand_value(dealer_hand) == 21:
            print("Push")
            return cash
        else: 
            print("Blackjack!")
            print("Player Wins")
            return cash + int(1.5 * bet) # 3:2 payout
    elif hand_value(dealer_hand) == 21:
        print("Dealer Blackjack")
        print("Dealer Wins")
        return cash - bet
    

    #Player's Turn
    while True:
        #Check if player can double down
        can_double = len(player_hand) == 2 and cash >= 2 * bet

        if can_double:
            prompt = "\nDo you want to hit, stand, or double down? "
        else:
            prompt = "\nDo you want to hit or stand? "
        h_or_s = input(prompt).lower()

        if h_or_s == "hit":
            player_hand.append(deal_card(deck))

            print(f"\nYou drew: ", display_hand([player_hand[-1]]))
            print()
            print("Player Hand: ", display_hand(player_hand))
            print("Dealer Hand: ", display_hand(dealer_hand, True))
            

            total = hand_value(player_hand)
            if total > 21:
                print("Bust")
                print("Dealer Wins")
                return cash - bet 
        elif h_or_s == "stand":
            break #leaves the while loop
        elif h_or_s == "double down" and can_double:
            bet *= 2
            player_hand.append(deal_card(deck))

            print(f"\nYou doubled down and drew: {display_hand([player_hand[-1]])}")
            print("Player Hand:", display_hand(player_hand))
            print("Dealer Hand:", display_hand(dealer_hand, True))

            total = hand_value(player_hand)
            if total > 21:
                print("Bust")
                print("Dealer Wins")
                return cash - bet
            break  # Auto-stand after doubling
        else: 
            print("Invalid input")

    #Reveal dealer hand
    print("Dealer Hand: ", display_hand(dealer_hand))

    #Dealer's Turn
    while hand_value(dealer_hand) < 17:
        dealer_hand.append(deal_card(deck))
        print("\nDealer drew:", display_hand([dealer_hand[-1]]))
        
        if hand_value(dealer_hand) > 21:
            print("Dealer Busts")
            print("Player Wins")
            return cash + bet #leaves the function
    
    player_total = hand_value(player_hand)
    dealer_total = hand_value(dealer_hand)

    print("\nFinal Results:")
    print("Your hand: ", display_hand(player_hand))
    print("\nDealer's hand:", display_hand(dealer_hand))

    if player_total > dealer_total:
        print("Player Wins")
        return cash + bet 
    elif player_total < dealer_total:
        print("Dealer Wins")
        return cash - bet
    else: 
        print("Push")
        return cash


def play_game():
    cash = starting_cash

    while cash > 0:
        cash = play_hand(cash)
        print(f"\nCash: {cash}")
        if cash <= 0:
            print("Game over.")
            print("")
            break
        again = input("\nPlay another round? (y/n): ").lower()
        print("-" * 30)
        if again != 'y':
            print("Final Cash: ", cash)
            print()
            break

if __name__ == "__main__":
    play_game()