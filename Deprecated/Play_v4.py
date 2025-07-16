# I worked with Claude for this version. 
# It is supposed to separate out the play_individual_hand() and play_round() functions a little bit
# I am getting more hands off
# Also trying to edit this so that you use the same deck across rounds
# For now, reshuffling happens when you are 75% of the way through the deck


# Import libraries
import random
import time 

# Help for formatting Printing
print("\n")

# Generate a deck of cards
suits = ['H', 'D', 'C', 'S']
ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
starting_cash = 1000

def create_deck(num_decks = 1):
    single_deck = [(rank, suit) for suit in suits for rank in ranks]
    return single_deck * num_decks

def shuffle_deck(deck):
    return random.shuffle(deck)

def deal_card(deck):
    if len(deck) == 0:
        # Emergency reshuffle
        deck.extend(create_deck())
        shuffle_deck(deck)
        print("Deck ran out, emergency reshuffle")
        print("(adding 1 new deck)")
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
    if len(hand) != 2:
        return False
    
    return hand[0][0] == hand[1][0] # True if same rank


def play_individual_hand(hand, deck, bet, cash, dealer_hand, is_split=False, hand_num=None):
    """
    Play a single hand and return the result without printing final outcomes.
    Returns a dictionary with hand state and result information.
    """
    
    # Only show hand number if we're in a split situation
    if is_split and hand_num:
        print(f"\nHand {hand_num}:")
    
    print("\nPlayer hand:", display_hand(hand))
    print("Dealer hand:", display_hand(dealer_hand, hidden=True))
    
    # Player's Turn
    while True:
        # Check if player can double
        can_double = len(hand) == 2 and cash >= 2 * bet

        if can_double:
            prompt = "\nDo you want to hit, stand, or double down? "
        else:
            prompt = "\nDo you want to hit or stand? "
        h_or_s = input(prompt).lower()

        if h_or_s == "hit":
            hand.append(deal_card(deck))
            print(f"\nYou drew: {display_hand([hand[-1]])}")
            print("Player Hand:", display_hand(hand))
            print("Dealer Hand:", display_hand(dealer_hand, True))

            if hand_value(hand) > 21:
                print("Bust!")
                return {
                    'hand': hand,
                    'result': 'bust',
                    'bet_multiplier': 1,
                    'final': True  # Hand is completely resolved
                }
            
        elif h_or_s == "stand":
            return {
                'hand': hand,
                'result': 'stand',
                'bet_multiplier': 1,
                'final': False  # Need to compare with dealer
            }

        elif h_or_s == "double down" and can_double:
            hand.append(deal_card(deck))
            print(f"\nYou doubled down and drew: {display_hand([hand[-1]])}")
            print("Player Hand:", display_hand(hand))

            if hand_value(hand) > 21:
                print("Bust!")
                return {
                    'hand': hand,
                    'result': 'bust',
                    'bet_multiplier': 2,
                    'final': True  # Hand is completely resolved
                }
            else:
                return {
                    'hand': hand,
                    'result': 'doubled',
                    'bet_multiplier': 2,
                    'final': False  # Need to compare with dealer
                }

        else: 
            print("Invalid Input")


def play_round(cash, deck, sleep = False):
    initial_hand = [deal_card(deck), deal_card(deck)]
    dealer_hand = [deal_card(deck), deal_card(deck)]

    bet = get_bet(cash)

    # Check for dealer blackjack first
    if hand_value(dealer_hand) == 21:
        print("Dealer Blackjack!")
        if hand_value(initial_hand) == 21:
            print("Push - both have blackjack")
            return cash
        else:
            print("Dealer Wins!")
            return cash - bet

    # Check for player blackjack
    if hand_value(initial_hand) == 21:
        print("Blackjack!")
        print("Player Wins!")
        return cash + int(1.5 * bet)

    # Determine if we're splitting
    hands_to_play = [initial_hand]
    is_split = False

    if can_split(initial_hand) and cash >= 2 * bet:
        split_choice = input("Do you want to split? (y/n) ").lower()
        if split_choice == 'y':
            # Create two new hands
            card1, card2 = initial_hand[0], initial_hand[1]
            hand1 = [card1, deal_card(deck)]
            hand2 = [card2, deal_card(deck)]

            hands_to_play = [hand1, hand2]
            is_split = True

            print(f"\nYou now have two hands:")
            print(f"Hand 1: {display_hand(hand1)}")
            print(f"Hand 2: {display_hand(hand2)}")

    # Play each hand and collect results
    hand_results = []
    for i, hand in enumerate(hands_to_play):
        hand_num = i + 1 if is_split else None
        result = play_individual_hand(hand, deck, bet, cash, dealer_hand, is_split, hand_num)
        hand_results.append(result)

    # Check if any hands need dealer comparison
    need_dealer = any(not result['final'] for result in hand_results)

    # Play dealer's hand only if needed
    if need_dealer:
        if sleep == True:
            time.sleep(1)

        print("\n" + "="*40)
        print("Dealer's turn:")
        print("="*40)

        print("\nDealer hand:", display_hand(dealer_hand))
        print()
        
        while hand_value(dealer_hand) < 17:
            dealer_hand.append(deal_card(deck))
            print(f"Dealer drew: {display_hand([dealer_hand[-1]])}")

        dealer_total = hand_value(dealer_hand)
        if dealer_total > 21:
            print("Dealer Busts!")
    else:
        dealer_total = hand_value(dealer_hand)


    # Calculate and display final results
    total_cash_change = 0
    
    if sleep == True:
        time.sleep(1)

    print("\n" + "="*40)
    print("FINAL RESULTS")
    print("="*40)

    for i, result in enumerate(hand_results):
        hand = result['hand']
        player_total = hand_value(hand)
        bet_amount = bet * result['bet_multiplier']
        
        # Determine hand label
        if is_split:
            hand_label = f"Hand {i+1}"
        else:
            hand_label = "Your hand"
            
        print(f"\n{hand_label}: {display_hand(hand)} (Total: {player_total})")
        
        # Calculate outcome
        if result['final']:  # Already resolved (bust)
            if result['result'] == 'bust':
                print(f"Result: BUST - LOSS (-${bet_amount})")
                total_cash_change -= bet_amount
        else:  # Compare with dealer
            print(f"Dealer hand: {display_hand(dealer_hand)} (Total: {hand_value(dealer_hand)})")
            
            if dealer_total > 21:
                print(f"Result: DEALER BUST - WIN (+${bet_amount})")
                total_cash_change += bet_amount
            elif player_total > dealer_total:
                print(f"Result: WIN (+${bet_amount})")
                total_cash_change += bet_amount
            elif player_total < dealer_total:
                print(f"Result: LOSS (-${bet_amount})")
                total_cash_change -= bet_amount
            else:
                print(f"Result: PUSH (+$0)")
    
    print(f"\nNet change: {'+' if total_cash_change >= 0 else ''}${total_cash_change}")
    return cash + total_cash_change


def play_game():
    cash = starting_cash
    deck = create_deck()
    shuffle_deck(deck)

    deck_len = len(deck)
    reshuffle_point = int(deck_len / 4)

    while cash > 0:
        cash = play_round(cash, deck, sleep=True)

        if len(deck) < reshuffle_point: 
            print(f"\nReshuffling... ({len(deck)} cards left)\n")
            deck = create_deck()
            shuffle_deck(deck)

        print(f"\nCash: ${cash}")
        if cash <= 0:
            print("Game over - out of money!\n")
            break
        again = input("\nPlay another round? (y/n): ").lower()
        print("-" * 50)
        if again != 'y':
            print(f"Final Cash: ${cash}")
            print("Thanks for playing!\n")
            break

if __name__ == "__main__":
    play_game()