# Import libraries
import random
import time 
import pyfiglet
import text as txt
import basic_strategy as hc
from base import GameState, GameInterface

# Generate a deck of cards
suits = ['H', 'D', 'C', 'S']
ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
starting_cash = 1000

def create_deck(state: GameState):
    single_deck = [(rank, suit) for suit in suits for rank in ranks]
    state.deck = single_deck * state.num_decks

    if state.card_counting: 
        state.cards_left = {rank: 4*state.num_decks for rank in ranks}

    random.shuffle(state.deck)

def get_card_deal(state: GameState, ui: GameInterface, msg = None):
    # I hate this and want to get rid of it but it has to do with cheat functionality so I will need to edit the function there as well
    if msg == 'phand': 
        card1 = get_card_deal(state, ui)
        card2 = get_card_deal(state, ui)
        return [card1, card2]
    
    if len(state.deck) == 0:
        # Emergency reshuffle
        create_deck(state)
        ui.display_emergency_reshuffle()

    card = state.deck.pop()
    state.cards_left[card[0]] -= 1

    return card

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

def play_individual_hand(state: GameState, ui: GameInterface, hand, bet): 
    """
    Play a single hand and return the result without printing final outcomes.
    Returns a dictionary with hand state and result information.

    Splits are handled in the play_round function because they should all happen before gameplay.
    """
    # Player's Turn
    while True:
        can_double = len(hand) == 2 and state.cash >= 2 * bet 
        
        h_or_s = ui.get_hit_stand_dd(hand, state.dealer_hand, can_double, ui)

        if h_or_s == "hit":
            new_card = ui.get_card(state, ui, msg = 'hit')
            hand.append(new_card)

            ui.display(f"\nYou Drew: {ui.display_hand([hand[-1]])}")
            ui.display()
            ui.display(f"Player Hand: {ui.display_hand(hand)}")
            ui.display(f"Dealer Hand: {ui.display_hand(state.dealer_hand, True)}")
            ui.display()

            if hand_value(hand) > 21:
                ui.display("Bust!")
                return {
                    'hand': hand,
                    'result': 'Player Bust',
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
            new_card = ui.get_card(state, ui, msg = 'dd')
            hand.append(new_card)

            ui.display(f"\nYou doubled down and drew: {ui.display_hand([new_card])}")
            ui.display(f"Player Hand: {ui.display_hand(hand)}")

            if hand_value(hand) > 21:
                ui.display("Bust!")
                return {
                    'hand': hand,
                    'result': 'Player Bust',
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
            ui.display("Invalid Input")


def play_round(state: GameState, ui: GameInterface):
    bet = ui.get_bet(state.cash)

    outcomes = [] 

    # Initial deal
    initial_hand = ui.get_card(state, ui, msg = 'phand')
    state.dealer_hand = [ui.get_card(state, ui, msg = 'dhand1'), ui.get_card(state, ui, msg = 'dhand?')]

    ui.display(f"\nPlayer Hand: {ui.display_hand(initial_hand)}")
    ui.display(f"Dealer Hand: {ui.display_hand(state.dealer_hand, hidden=True)}")

    # NOTE: THESE BLACKJACK INSTANCES DONT DO THE FULL RESULTS PRINTOUT 
    # because they return early.
    # it might be as easy as making a results dictionary and not returning anything 
    # but I need to check the logic in the function. 

    # Dealer Blackjack Check: Handle the "peek" for cheat mode
    if state.dealer_hand[1] == ('?', '?') and card_value(state.dealer_hand[0]) in [10,11]:
        ui.display("\nDealer is checking for Blackjack...\n")
        state.dealer_hand[1] = ui.get_card(state, ui, msg='bjcheck')

    # Check for dealer blackjack first
    if state.dealer_hand[1] != ('?', '?') and hand_value(state.dealer_hand) == 21:
        if hand_value(initial_hand) == 21: # PUSH: Double blackjack
            round_results = {
                    'cash_changes' : [0],
                    'player_hands' : [initial_hand],
                    'dealer_hand': state.dealer_hand, 
                    'outcomes' : ['Blackjack Push']}
            ui.display_final_results(round_results)
            return round_results
        else: # Dealer only blackjack
            round_results = { 
                    'cash_changes' : [-bet],
                    'player_hands' : [initial_hand],
                    'dealer_hand': state.dealer_hand, 
                    'outcomes' : ['Dealer Blackjack']}
            ui.display_final_results(round_results)
            return round_results

    # Check for player blackjack
    if hand_value(initial_hand) == 21: #Player only blackjack
        ui.display("\nBlackjack!\n")
        
        # Prompting for dealer's second card to keep track
        if state.dealer_hand[1] == ('?', '?'):
            state.dealer_hand[1] = ui.get_card(state, ui, msg = 'dhand2')

        round_results = { 
                'cash_changes': [int(1.5 * bet)],
                'player_hands' : [initial_hand],
                'dealer_hand': state.dealer_hand, 
                'outcomes' : ['Player Blackjack']}
        ui.display_final_results(round_results)
        return round_results
    
    MAX_HANDS = 4 # Allow up to 4 hands total (initial + 3 splits)

    # --- Refactored Split and Hand Preparation Logic ---
    player_hands_for_decision = [(initial_hand, bet)] # Hands waiting for split decision
    final_player_hands = [] # Hands ready for actual play


    # Phase 1: Handle all split decisions
    while player_hands_for_decision:
        current_hand, current_bet = player_hands_for_decision.pop(0) # Take the first hand to process

        # Check if splitting *this* hand would exceed MAX_HANDS
        # A split turns 1 hand into 2, so it adds 1 to the total count.
        if len(final_player_hands) + len(player_hands_for_decision) + 1 > MAX_HANDS: # made redundant by following?
            ui.display(f"Cannot split {ui.display_hand(current_hand)}. Maximum number of hands ({MAX_HANDS}) reached.")
            final_player_hands.append((current_hand, current_bet))
            continue # Move to the next hand in the decision queue

        # If it's not a pair, or player can't afford, or max hands reached, it's a final hand
        if not can_split(current_hand) or state.cash < (len(final_player_hands) + len(player_hands_for_decision) + 1) * bet or len(final_player_hands) + len(player_hands_for_decision) >= MAX_HANDS:
            final_player_hands.append((current_hand, current_bet))
            continue # Move to the next hand in the decision queue

        # It's a pair and can be split
        split_choice = ui.get_split_choice(current_hand, state.dealer_hand, ui)

        if split_choice == 'y':
            card1, card2 = current_hand[0], current_hand[1]
            
            new_hand1 = [card1, ui.get_card(state, ui, msg = 'splithand1')]
            new_hand2 = [card2, ui.get_card(state, ui, msg = 'splithand2')]
            
            # Special handling for split aces (rule: only one card after split)
            if card1[0] == 'A':
                ui.display("Split aces detected — each hand gets one card only and cannot hit further.")
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
            ui.display(f"\n--- Playing Hand {i+1} ---") # More prominent header
            ui.display(f"Player Hand: {ui.display_hand(hand)}")
            ui.display(f"Dealer Hand: {ui.display_hand(state.dealer_hand, hidden=True)}")

        if is_split_ace_initial:
            ui.display("Split Aces: This hand received one card and must stand.")
            hand_result = {
                'hand': hand,
                'result': 'stand',
                'bet_multiplier': 1,
                'final': False
            }
        elif hand_value(hand) == 21 and is_split_game and len(hand) == 2 and hand[0][0] == 'A': #redundant?
            ui.display("Split Aces: You got 21 with your second card. You must stand.")
            hand_result = {
                'hand': hand,
                'result': 'stand',
                'bet_multiplier': 1,
                'final': False
            }
        else:
            #is it getting called?
            hand_result = play_individual_hand(state, ui, hand, bet)
            
        hand_results.append(hand_result)

    # Check if any hands need dealer comparison
    need_dealer = any(not hand_result['final'] for hand_result in hand_results)

    # Play dealer's hand only if needed
    if need_dealer:
        ui.wait(1)

        ui.display("\n" + "="*40)
        ui.display("Dealer's turn:")
        ui.display("="*40)

        # Resolve dealer's hole card if it's a placeholder
        if state.dealer_hand[1] == ('?', '?'): 
            ui.display()
            state.dealer_hand[1] = ui.get_card(state, ui, msg='dhand2')

        ui.display(f"\nDealer Hand: {ui.display_hand(state.dealer_hand)}")
        ui.display("")
        
        while hand_value(state.dealer_hand) < 17:
            state.dealer_hand.append(ui.get_card(state, ui, msg = 'dhit'))
            ui.display(f"Dealer Drew: {ui.display_hand([state.dealer_hand[-1]])}")

         

        dealer_total = hand_value(state.dealer_hand)
        if dealer_total > 21:
            ui.display("Dealer Busts!")
            
            # for hand_result in hand_results:
            #     if result['final']:
            #         outcomes.append(result['result'])
            #     else:
            #         outcomes.append("Dealer Bust")

            round_results = { # I have to change this because its returning too early and not printing final results
                    'cash_changes': [bet], #is this bad because it can't handle split hands? 
                    'player_hands' : [d['hand'] for d in hand_results],
                    'dealer_hand': state.dealer_hand, 
                    'outcomes' : outcomes}
    
    ui.wait(3)

    cash_changes = []

    for i, hand_result in enumerate(hand_results):
        hand = hand_result['hand']
        player_total = hand_value(hand)
        bet_amount = bet * hand_result['bet_multiplier'] # Use original bet for multiplier
        
        # Calculate outcome
        if hand_result['final']:  # Already resolved (bust)
            if hand_result['result'] == 'Player Bust': #redundant?
                cash_changes.append(-bet_amount)
                outcomes.append('Player Bust')
        else:  # Compare with dealer
            if dealer_total > 21:
                cash_changes.append(bet_amount)
                outcomes.append('Dealer Bust')
            elif player_total > dealer_total:
                cash_changes.append(bet_amount)
                outcomes.append('Player Higher')
            elif player_total < dealer_total:
                cash_changes.append(-bet_amount)
                outcomes.append('Dealer Higher')
            else:
                cash_changes.append(0)
                outcomes.append('Push')
    
    round_results = { 
            'cash_changes': [int(cash_change) for cash_change in cash_changes], #list
            'player_hands' : [d['hand'] for d in hand_results], #list
            'dealer_hand': state.dealer_hand, 
            'outcomes' : outcomes #list
            }
    
    # Resolve dealer's hole card if it's a placeholder
    if state.dealer_hand[1] == ('?', '?'): 
        ui.display()
        state.dealer_hand[1] = ui.get_card(state, ui, msg='dhand2')

    ui.display_final_results(round_results)
    return round_results


def play_game(state: GameState, ui: GameInterface):
    txt.print_title_box(["STARTING NEW GAME..."])
    ui.display()
    state.cash = starting_cash
    create_deck(state)

    deck_len = len(state.deck)

    # TODO: Is this how actual blackjack games work? 
    reshuffle_size = int(deck_len / 4)

    ###TESTING CODE
    #
    # cards_to_add = list(reversed([
    #     ('8', 'H'), ('10', 'D'),   # Player initial hand (8,8)
    #     ('7', 'C'), ('10', 'S'),   # Cards dealt to first and second hands (or to dealer if no split)
    #     ('10', 'H'), ('10', 'D'),   # Further split hands
    #     ('10', 'H'), ('7', 'D')   # Dealer cards
    # ]))
    
    # state.deck.extend(cards_to_add)
    # #
    # ###TESTING CODE

    round_num = 0
    while state.cash > 0:
        round_num += 1
        message = f"Round {round_num}"
        buffer = int((80 - len(message))/2)
        ui.display('-'*80) if round_num != 1 else None
        ui.display(" " * buffer, message, " " * buffer)
        ui.display('-'*80)

        result_dict = play_round(state, ui)

        state.cash = state.cash + sum(result_dict['cash_changes'])

        if len(state.deck) < reshuffle_size: 
            ui.display(f"\nReshuffling... (with {len(state.deck)} cards left)\n")
            create_deck(state)

        ui.display(f"Cash: ${state.cash}")
        if state.cash <= 0:
            txt.print_title_box(["GAME OVER", "~ OUT OF MONEY! ~"])
            ui.display()
            break
        again = ui.get_another_round()
        if again != 'y':
            txt.print_title_box(["GAME OVER", f"~ FINAL CASH: ${state.cash} ~"])
            ui.display()
            ui.display("\033[3mThanks for Playing!\033[0m\n") # These are italics
            break


def run_text_mode():
    play_game(state, text_mode)

def run_hardcode_mode(game_or_round):
    if game_or_round == 'game':
        play_game(state, hardcode_mode)
    elif game_or_round == 'round':
        play_round(state, hardcode_mode)
    else: 
        raise ValueError("Pass either 'game' or 'round as arguments to 'run_hardcode_mode()'")


if __name__ == "__main__":
    # These are defined in config.py and only useful here for testing/to avoid circular imports
    state = GameState()
    text_mode = GameInterface(
        # Display functions
        display                     = print, 
        display_hand                = txt.display_hand_print,
        display_emergency_reshuffle = txt.display_emergency_reshuffle_print,
        display_final_results       = txt.display_final_results_print,

        # Get Functions
        get_bet                     = txt.get_bet_print,
        get_card                    = get_card_deal,
        get_split_choice            = txt.get_split_choice_print,
        get_hit_stand_dd            = txt.get_hit_stand_dd_print,
        get_another_round           = txt.get_another_round_print,

        # Other
        sleep = True)
    
    hardcode_mode = GameInterface(
        # Display functions
        display                     = print,
        display_hand                = txt.display_hand_print, # or hc.display_hand_hardcode
        display_emergency_reshuffle = txt.display_emergency_reshuffle_print, # or hc.emergency_reshuffle_hardcode
        display_final_results       = txt.display_final_results_print,

        # Get Functions
        get_bet                     = hc.get_bet_hardcode,
        get_card                    = get_card_deal,
        get_split_choice            = hc.get_split_choice_hardcode,
        get_hit_stand_dd            = hc.get_hit_stand_dd_hardcode,
        get_another_round           = hc.get_another_round_hardcode,

        # Other
        sleep = False)
    
    run_text_mode()
    #run_hardcode_mode('game')
    