import random

# Constants
suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
          '8': 8, '9': 9, '10': 10, 'J': 10, 'Q': 10, 'K': 10, 'A': 11}

# Card and deck helpers
def create_deck():
    return [(rank, suit) for suit in suits for rank in ranks]

def shuffle_deck(deck):
    random.shuffle(deck)

def deal_card(deck):
    return deck.pop()

def hand_value(hand):
    total = 0
    aces = 0
    for card in hand:
        rank = card[0]
        total += values[rank]
        if rank == 'A':
            aces += 1
    # Adjust for aces
    while total > 21 and aces:
        total -= 10
        aces -= 1
    return total

def display_hand(hand, hidden=False):
    if hidden:
        print("Dealer's Hand: [??] " + ', '.join(f'{r} of {s}' for r, s in hand[1:]))
    else:
        print(', '.join(f'{r} of {s}' for r, s in hand))

def play_blackjack():
    deck = create_deck()
    shuffle_deck(deck)

    player_hand = [deal_card(deck), deal_card(deck)]
    dealer_hand = [deal_card(deck), deal_card(deck)]

    print("Your hand:")
    display_hand(player_hand)
    print(f"Total: {hand_value(player_hand)}\n")

    print("Dealer's hand:")
    display_hand(dealer_hand, hidden=True)
    print()

    # Player's turn
    while True:
        action = input("Do you want to Hit or Stand? (h/s): ").lower()
        if action == 'h':
            player_hand.append(deal_card(deck))
            print("\nYou drew:")
            display_hand([player_hand[-1]])
            total = hand_value(player_hand)
            print(f"Total: {total}\n")
            if total > 21:
                print("You busted! Dealer wins.")
                return
        elif action == 's':
            break
        else:
            print("Invalid input, please enter 'h' or 's'.")

    # Dealer's turn
    print("\nDealer's turn:")
    display_hand(dealer_hand)
    while hand_value(dealer_hand) < 17:
        dealer_hand.append(deal_card(deck))
        print("Dealer drew:")
        display_hand([dealer_hand[-1]])
        print(f"Dealer total: {hand_value(dealer_hand)}")

    player_total = hand_value(player_hand)
    dealer_total = hand_value(dealer_hand)

    print("\nFinal Results:")
    print("Your hand:")
    display_hand(player_hand)
    print(f"Your total: {player_total}")

    print("\nDealer's hand:")
    display_hand(dealer_hand)
    print(f"Dealer total: {dealer_total}")

    # Determine winner
    if dealer_total > 21 or player_total > dealer_total:
        print("\nYou win!")
    elif player_total < dealer_total:
        print("\nDealer wins!")
    else:
        print("\nIt's a tie!")

# Run the game
if __name__ == "__main__":
    while True:
        play_blackjack()
        again = input("\nPlay again? (y/n): ").lower()
        if again != 'y':
            print("Thanks for playing!")
            break
