# Rewriting Play_v5.py to be agnostic about display/output

# Import libraries
import random
import time 
import Text as text
import Hardcode as hc

# Generate a deck of cards
suits = ['H', 'D', 'C', 'S']
ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
starting_cash = 1000

def create_deck(num_decks = 1):
    single_deck = [(rank, suit) for suit in suits for rank in ranks]
    return single_deck * num_decks

def shuffle_deck(deck):
    return random.shuffle(deck)

def deal_card(deck, display_emergency_reshuffle):
    if len(deck) == 0:
        # Emergency reshuffle
        deck.extend(create_deck())
        shuffle_deck(deck)
        display_emergency_reshuffle()
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

def can_split(hand):    
    if len(hand) != 2:
        return False
    
    return hand[0][0] == hand[1][0] # True if same rank

def play_individual_hand(
    hand,
    deck,
    bet,
    cash,
    dealer_hand,
    get_hit_stand_dd,
    display, 
    display_emergency_reshuffle, 
    display_hand
):
    """
    Play a single hand and return the result without printing final outcomes.
    Returns a dictionary with hand state and result information.

    Splits are handled in the play_round function because they should all happen before gameplay.
    """
    # Player's Turn
    while True:
        can_double = len(hand) == 2 and cash >= 2 * bet 
        h_or_s = get_hit_stand_dd(hand, dealer_hand, can_double)

        if h_or_s == "hit":
            hand.append(deal_card(deck, display_emergency_reshuffle))
            display(f"\nYou drew: {display_hand([hand[-1]])}")
            display(f"Player Hand: {display_hand(hand)}")
            display(f"Dealer Hand: {display_hand(dealer_hand, True)}")

            if hand_value(hand) > 21:
                display("Bust!")
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
            hand.append(deal_card(deck, display_emergency_reshuffle))
            display(f"\nYou doubled down and drew: {display_hand([hand[-1]])}")
            display(f"Player Hand: {display_hand(hand)}")

            if hand_value(hand) > 21:
                display("Bust!")
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
            display("Invalid Input")


def play_round(
    cash,
    deck,
    sleep,
    get_bet,
    get_split_choice,
    display,
    get_hit_stand_dd,
    display_hand,
    display_emergency_reshuffle
):

    initial_hand = [deal_card(deck, display_emergency_reshuffle), deal_card(deck, display_emergency_reshuffle)]
    dealer_hand = [deal_card(deck, display_emergency_reshuffle), deal_card(deck, display_emergency_reshuffle)]

    bet = get_bet(cash)

    display(f"\nPlayer hand: {display_hand(initial_hand)}")
    display(f"Dealer hand: {display_hand(dealer_hand, hidden=True)}")

    # NOTE: THESE BLACKJACK INSTANCES DONT DO THE FULL RESULTS PRINTOUT 
    # because they return early.
    # it might be as easy as making a results dictionary and not returning anything 
    # but I need to check the logic in the function. 

    # Check for dealer blackjack first
    if hand_value(dealer_hand) == 21:
        display("Dealer Blackjack!")
        if hand_value(initial_hand) == 21:
            display("Push - both have blackjack")
            return cash
        else:
            display("Dealer Wins!")
            return cash - bet

    # Check for player blackjack
    if hand_value(initial_hand) == 21:
        display("Blackjack!")
        display("Player Wins!")
        return cash + int(1.5 * bet)

    # --- Refactored Split and Hand Preparation Logic ---
    player_hands_for_decision = [(initial_hand, bet)] # Hands waiting for split decision
    final_player_hands = [] # Hands ready for actual play

    MAX_HANDS = 4 # Allow up to 4 hands total (initial + 3 splits)

    # Phase 1: Handle all split decisions
    while player_hands_for_decision:
        current_hand, current_bet = player_hands_for_decision.pop(0) # Take the first hand to process

        # Check if splitting *this* hand would exceed MAX_HANDS
        # A split turns 1 hand into 2, so it adds 1 to the total count.
        if len(final_player_hands) + len(player_hands_for_decision) + 1 > MAX_HANDS:
            display(f"Cannot split {display_hand(current_hand)}. Maximum number of hands ({MAX_HANDS}) reached.")
            final_player_hands.append((current_hand, current_bet))
            continue # Move to the next hand in the decision queue

        # If it's not a pair, or player can't afford, or max hands reached, it's a final hand
        if not can_split(current_hand) or cash < (len(final_player_hands) + len(player_hands_for_decision) + 1) * bet or len(final_player_hands) + len(player_hands_for_decision) >= MAX_HANDS:
            final_player_hands.append((current_hand, current_bet))
            continue # Move to the next hand in the decision queue

        # It's a pair and can be split
        split_choice = get_split_choice(current_hand, dealer_hand)

        if split_choice == 'y':
            card1, card2 = current_hand[0], current_hand[1]
            
            new_hand1 = [card1, deal_card(deck, display_emergency_reshuffle)]
            new_hand2 = [card2, deal_card(deck, display_emergency_reshuffle)]
            
            # Special handling for split aces (rule: only one card after split)
            if card1[0] == 'A':
                display("Split aces detected — each hand gets one card only and cannot hit further.")
                # For aces, these hands are immediately considered final for decision making,
                # as they cannot be split again or hit.
                final_player_hands.append((new_hand1, current_bet))
                final_player_hands.append((new_hand2, current_bet))
            else:
                # Add new hands to the front of the decision queue to process them next
                player_hands_for_decision.insert(0, (new_hand2, current_bet))
                player_hands_for_decision.insert(0, (new_hand1, current_bet))
        else:
            final_player_hands.append((current_hand, current_bet))

    player_hands = final_player_hands # This will now be the correctly ordered list of hands to play
    is_split_game = len(player_hands) > 1

    # PLAY PHASE: Play out all prepared hands
    hand_results = []
    for i, (hand, hand_bet) in enumerate(player_hands):
        is_split_ace_initial = (hand[0][0] == 'A' and len(hand) == 2 and is_split_game and hand_value(hand) != 21)

        # Print the hand header BEFORE calling play_individual_hand
        if is_split_game:
            display(f"\n--- Playing Hand {i+1} ---") # More prominent header
        else:
            display("\n--- Playing Your Hand ---") # For non-split game

        # Display the hand for the player to see *before* they are prompted for action
        # I WANT TO FIX THIS. AS IT STANDS, IT JUST PRINTS THE HANDS TWICE. WHY WOULD WE WANT THAT?
        display(f"Player hand: {display_hand(hand)}")
        display(f"Dealer hand: {display_hand(dealer_hand, hidden=True)}")

        if is_split_ace_initial:
            display("Split Aces: This hand received one card and must stand.")
            result = {
                'hand': hand,
                'result': 'stand',
                'bet_multiplier': 1,
                'final': False
            }
        elif hand_value(hand) == 21 and is_split_game and len(hand) == 2 and hand[0][0] == 'A':
            display("Split Aces: You got 21 with your second card. You must stand.")
            result = {
                'hand': hand,
                'result': 'stand',
                'bet_multiplier': 1,
                'final': False
            }
        else:
            result = play_individual_hand(
                                            hand,
                                            deck,
                                            bet,
                                            cash,
                                            dealer_hand,
                                            get_hit_stand_dd,
                                            display, 
                                            display_emergency_reshuffle, 
                                            display_hand)
            
        hand_results.append(result)

    # Check if any hands need dealer comparison
    need_dealer = any(not result['final'] for result in hand_results)

    # Play dealer's hand only if needed
    if need_dealer:
        if sleep == True:
            time.sleep(1)

        display("\n" + "="*40)
        display("Dealer's turn:")
        display("="*40)

        display(f"\nDealer hand: {display_hand(dealer_hand)}")
        display("")
        
        while hand_value(dealer_hand) < 17:
            dealer_hand.append(deal_card(deck, display_emergency_reshuffle))
            display(f"Dealer drew: {display_hand([dealer_hand[-1]])}")

        dealer_total = hand_value(dealer_hand)
        if dealer_total > 21:
            display("Dealer Busts!")
    else:
        dealer_total = hand_value(dealer_hand)


    # Calculate and display final results
    total_cash_change = 0
    
    if sleep == True:
        time.sleep(1)

    display("\n" + "="*40)
    display("FINAL RESULTS")
    display("="*40)

    for i, result in enumerate(hand_results):
        hand = result['hand']
        player_total = hand_value(hand)
        bet_amount = bet * result['bet_multiplier'] # Use original bet for multiplier
        
        # Determine hand label
        if is_split_game:
            hand_label = f"Hand {i+1}"
        else:
            hand_label = "Your hand"
            
        display(f"\n{hand_label}: {display_hand(hand)} (Total: {player_total})")
        
        # Calculate outcome
        if result['final']:  # Already resolved (bust)
            if result['result'] == 'bust':
                display(f"Result: BUST - LOSS (-${bet_amount})")
                total_cash_change -= bet_amount
        else:  # Compare with dealer
            display(f"Dealer hand: {display_hand(dealer_hand)} (Total: {hand_value(dealer_hand)})")
            
            if dealer_total > 21:
                display(f"Result: DEALER BUST - WIN (+${bet_amount})")
                total_cash_change += bet_amount
            elif player_total > dealer_total:
                display(f"Result: WIN (+${bet_amount})")
                total_cash_change += bet_amount
            elif player_total < dealer_total:
                display(f"Result: LOSS (-${bet_amount})")
                total_cash_change -= bet_amount
            else:
                display(f"Result: PUSH (+$0)")
    
    display(f"\nNet change: {'+' if total_cash_change >= 0 else ''}${total_cash_change}")
    return cash + total_cash_change


def play_game(
    get_another_round,
    display,
    get_bet,
    get_split_choice,
    get_hit_stand_dd,
    display_hand,
    display_emergency_reshuffle,
    sleep
):

    cash = starting_cash
    deck = create_deck()
    shuffle_deck(deck)

    deck_len = len(deck)
    reshuffle_point = int(deck_len / 4)

    # ### SPECIAL TESTING CODE
    # cards_to_add = list(reversed([
    #     ('2', 'H'), ('2', 'D'),   # Player initial hand (8,8)
    #     ('K', 'C'), ('3', 'S'),   # Cards dealt to first and second hands (or to dealer if no split)
    #     ('8', 'H'), ('8', 'D'),   # Further split hands
    #     ('10', 'H'), ('7', 'D')   # Dealer cards
    # ]))

    # deck.extend(cards_to_add)
    # ###SPECIAL TESTING CODE ^^^^^^^

    #Just for formatting
    display("\n") #Do I need this?

    while cash > 0:
        cash = play_round(
                    cash,
                    deck,
                    sleep,
                    get_bet,
                    get_split_choice,
                    display,
                    get_hit_stand_dd,
                    display_hand,
                    display_emergency_reshuffle)

        if len(deck) < reshuffle_point: 
            display(f"\nReshuffling... ({len(deck)} cards left)\n")
            deck = create_deck()
            shuffle_deck(deck)

        display(f"\nCash: ${cash}")
        if cash <= 0:
            display("Game over - out of money!\n")
            break
        again = get_another_round()
        display("-" * 50)
        if again != 'y':
            display(f"Final Cash: ${cash}")
            display("Thanks for playing!\n")
            break


def run_text_mode():
    play_game(
        get_another_round            = text.get_another_round_print,
        display                      = print,
        get_bet                      = text.get_bet_print,
        get_split_choice             = text.get_split_choice_print,
        get_hit_stand_dd             = text.get_hit_stand_dd_print,
        display_hand                 = text.display_hand_print,
        display_emergency_reshuffle  = text.display_emergency_reshuffle_print,
        sleep                        = True
    )

def run_hardcode_mode(game_or_round):
    #I may want to add the option to play rounds and skip over the game functionality so I can
    #iterate a large number of rounds without risking running out of cash

    #I think the strategy would be to pull all these arguments out and make them variable definitions, 
    # and then have a play game and play round function
    get_another_round              = hc.get_another_round_hardcode
    display                        = print
    get_bet                        = hc.get_bet_hardcode
    get_split_choice               = hc.get_split_choice_hardcode
    get_hit_stand_dd               = hc.get_hit_stand_dd_hardcode
    display_hand                   = text.display_hand_print           # or hc.display_hand_hardcode
    display_emergency_reshuffle    = text.display_emergency_reshuffle_print    # or hc.emergency_reshuffle_hardcode
    sleep                          = False

    if game_or_round == 'game':
        play_game(
            get_another_round,              
            display,                        
            get_bet,                        
            get_split_choice,               
            get_hit_stand_dd,               
            display_hand,                   
            display_emergency_reshuffle,    
            sleep                          
        )
    elif game_or_round == 'round':
        deck = create_deck()
        shuffle_deck(deck)

        play_round(
            1000, #infinite cash relative to bet size
            deck, 
            sleep, 
            lambda cash: 1, #minimal bet size, the lambda is so its callable to avoid an error
            get_split_choice, 
            display, 
            get_hit_stand_dd, 
            display_hand, 
            display_emergency_reshuffle
        )
    else: 
        raise ValueError("Pass either 'game' or 'round as arguments to 'run_hardcode_mode()'")


if __name__ == "__main__":
    run_hardcode_mode('game')