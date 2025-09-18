### HELPER FUNCTIONS

#This is just a copy from Logic and shouldn't really be here but it is for now so I don't do circular imports
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


 ### DISPLAY FUNCTIONS

def display_hand_print(hand, hidden=False): #Should this really be a return or should it print??
    if hidden == False:
        return ', '.join(f'{rank}{suit}' for rank, suit in hand)
    else:
        return f'{hand[0][0]}{hand[0][1]}, [X]'

def display_emergency_reshuffle_print():
    print("Deck ran out, emergency reshuffle")
    print("(adding 1 new deck)")
    # Will matter for card-counting stuff
    
def display_final_results_print(round_results):
    print("\n" + "="*40)
    print("FINAL RESULTS")
    print("="*40)

    is_split = len(round_results['cash_changes']) != 1

    if round_results['outcomes'] == ['Blackjack Push']:
        print("Push - both have blackjack")
    elif round_results['outcomes'] == ['Dealer Blackjack']:
        print("Dealer Blackjack!")
        print("Dealer Wins!")
    elif round_results['outcomes'] == ['Player Blackjack']:
        print("Blackjack!")
        print("Player Wins!")

    for i, outcome in enumerate(round_results['outcomes']):
        hand_label = f"Player Hand {i+1}" if is_split else "Player Hand"
        w_l_p = "WIN" if round_results['cash_changes'][i] > 0 else ("PUSH" if round_results['cash_changes'][i] == 0 else "LOSS")
        cash_sign = "-" if round_results['cash_changes'][i] < 0 else "+"

        print(f"\n{hand_label}: {display_hand_print(round_results['player_hands'][i])} (Total: {hand_value(round_results['player_hands'][i])})")
        print(f"Dealer Hand: {display_hand_print(round_results['dealer_hand'])} (Total: {hand_value(round_results['dealer_hand'])})")
        print(f"Result: {outcome} - {w_l_p} ({cash_sign}${abs(round_results['cash_changes'][i])})")

    #Should I make this only if split?
    net_change = sum(round_results['cash_changes'])
    net_cash_sign = "-" if net_change < 0 else "+"
    print(f"\nNet Change: {net_cash_sign}${abs(net_change)}")


### GET FUNCTIONS

def get_another_round_print():
    return input("\nPlay another round? (y/n): ").lower()

def get_bet_print(cash, input = input, print = print):
    while True:
        try:
            print(f"Cash: {cash}")
            bet = int(input("Bet: "))
            if bet <= 0:
                print("\nBet must be positive.\n")
            elif bet > cash:
                print("\nYou don't have enough money for that bet.\n")
            else:
                return bet
        except ValueError:
            print("\nPlease enter a valid (whole) number.\n")

def get_split_choice_print(hand, dealer_hand): #dealer hand is just in there because the hardcode version of the file needs it
    print(f"\nCurrently considering: {display_hand_print(hand)}")
    return input("Do you want to split this hand? (y/n) ").lower()

def get_hit_stand_dd_print(hand, dealer_hand, can_double): #dealer hand is in hardcode so that's why its here
    if can_double:
        prompt = "\nDo you want to hit, stand, or double down? "
    else:
        prompt = "\nDo you want to hit or stand? "
    h_or_s = input(prompt).lower()

    return h_or_s




# #Testing
# test_dict = {
# 'cash_changes': [150], 
# 'player_hands': [[('A', 'H'), ('10', 'H')]], 
# 'dealer_hand': [('8', 'C'), ('8', 'S')], 
# 'outcomes': ['Player Blackjack']
# }

# display_final_results_print(test_dict)