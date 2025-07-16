# This is the version where I am attempting to incorporate a logic for splitting. 

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

def card_value(card):
    rank, _ = card
    if rank in ['J', 'Q', 'K']:
        return 10
    elif rank == 'A':
        return 11
    else:
        return int(rank)

def hand_value(hand):
    total = 0
    aces = 0 
    
    for card in hand: 
        total += card_value(card)

        #Check for aces
        if card[0] == 'A':
            aces +=1

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

def can_split(hand):
    """Check if hand can be split (two cards of same rank)"""
    if len(hand) != 2:
        return False
    return hand[0][0] == hand[1][0]  # Same rank

def play_individual_hand(hand, deck, bet, cash, dealer_hand, hand_num=1):
    """Play a single hand (could be original or split hand)"""
    print(f"\nPlaying Hand {hand_num}:")
    print("Player hand:", display_hand(hand))
    print("Dealer hand:", display_hand(dealer_hand, hidden=True))

    # Check for blackjack (only on initial 2-card hands that aren't from splits)
    if len(hand) == 2 and hand_num == 1 and hand_value(hand) == 21:
        print("Blackjack!")
        return hand, 1.5 * bet  # Return the profit, not total cash
    
    # Player's turn
    while True:
        # Check if player can double
        can_double = len(hand) == 2 and cash >= bet
        
        if can_double:
            prompt = "\nDo you want to hit, stand, or double? "
        else:
            prompt = "\nDo you want to hit or stand? "
        
        action = input(prompt).lower()

        if action == "hit":
            hand.append(deal_card(deck))
            print(f"\nYou drew: {display_hand([hand[-1]])}")
            print("Player Hand:", display_hand(hand))
            
            if hand_value(hand) > 21:
                print("Bust!")
                return hand, -bet  # Return the loss
                
        elif action == "stand":
            break
            
        elif action == "double" and can_double:
            hand.append(deal_card(deck))
            print(f"\nYou doubled down and drew: {display_hand([hand[-1]])}")
            print("Player Hand:", display_hand(hand))
            
            if hand_value(hand) > 21:
                print("Bust!")
                return hand, -2 * bet  # Return double loss
            
            # Compare with dealer for doubled hand
            return hand, 2 * bet if hand_value(hand) > hand_value(dealer_hand) else -2 * bet
            
        else:
            print("Invalid input")
    
    # Hand completed without busting
    return hand, 0  # We'll compare with dealer later

def play_round(cash):
    deck = create_deck()
    shuffle_deck(deck)

    initial_hand = [deal_card(deck), deal_card(deck)]
    dealer_hand = [deal_card(deck), deal_card(deck)]
    
    bet = get_bet(cash)
    
    print("Player hand:", display_hand(initial_hand))
    print("Dealer hand:", display_hand(dealer_hand, hidden=True))
    
    # Check for dealer blackjack first
    if hand_value(dealer_hand) == 21:
        print("Dealer Blackjack!")
        if hand_value(initial_hand) == 21:
            print("Push!")
            return cash
        else:
            print("Dealer Wins!")
            return cash - bet
    
    # Check for player blackjack
    if hand_value(initial_hand) == 21:
        print("Player Blackjack!")
        return cash + int(1.5 * bet)
    
    # Start with list of hands to play
    hands_to_play = [initial_hand]
    hand_bets = [bet]
    total_bet = bet
    
    # Check if player can split
    if can_split(initial_hand) and cash >= 2 * bet:
        split_choice = input("\nDo you want to split? (y/n): ").lower()
        if split_choice == 'y':
            # Create two new hands
            card1, card2 = initial_hand[0], initial_hand[1]
            hand1 = [card1, deal_card(deck)]
            hand2 = [card2, deal_card(deck)]
            
            hands_to_play = [hand1, hand2]
            hand_bets = [bet, bet]
            total_bet = 2 * bet
            
            print(f"\nSplit! You now have two hands:")
            print(f"Hand 1: {display_hand(hand1)}")
            print(f"Hand 2: {display_hand(hand2)}")
    
    # Play each hand
    final_hands = []
    cash_changes = []
    
    for i, hand in enumerate(hands_to_play):
        hand_result, cash_change = play_individual_hand(hand, deck, hand_bets[i], cash, dealer_hand, i+1)
        final_hands.append(hand_result)
        cash_changes.append(cash_change)
    
    # Now play dealer's hand
    print("\nDealer's turn:")
    print("Dealer hand:", display_hand(dealer_hand))
    
    while hand_value(dealer_hand) < 17:
        dealer_hand.append(deal_card(deck))
        print(f"Dealer drew: {display_hand([dealer_hand[-1]])}")
        print(f"Dealer hand: {display_hand(dealer_hand)}")
    
    dealer_total = hand_value(dealer_hand)
    dealer_busted = dealer_total > 21
    
    if dealer_busted:
        print("Dealer busts!")
    
    # Calculate final results
    total_cash_change = 0
    
    for i, (hand, bet_amount) in enumerate(zip(final_hands, hand_bets)):
        player_total = hand_value(hand)
        
        print(f"\nHand {i+1} Results:")
        print(f"Your hand: {display_hand(hand)} (Value: {player_total})")
        print(f"Dealer hand: {display_hand(dealer_hand)} (Value: {dealer_total})")
        
        # If we already calculated the result (bust, double, blackjack), use it
        if cash_changes[i] != 0:
            total_cash_change += cash_changes[i]
            if cash_changes[i] > 0:
                print(f"Hand {i+1}: WIN (+{cash_changes[i]})")
            else:
                print(f"Hand {i+1}: LOSS ({cash_changes[i]})")
        else:
            # Compare with dealer
            if dealer_busted or player_total > dealer_total:
                print(f"Hand {i+1}: WIN (+{bet_amount})")
                total_cash_change += bet_amount
            elif player_total < dealer_total:
                print(f"Hand {i+1}: LOSS (-{bet_amount})")
                total_cash_change -= bet_amount
            else:
                print(f"Hand {i+1}: PUSH (0)")
    
    return cash + total_cash_change

def play_game():
    cash = starting_cash

    while cash > 0:
        cash = play_round(cash)
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