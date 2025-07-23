#This file will be where functions are defined that interact with Logic.py
# and allow it to be played as a text based terminal game


# Potentially I should do all the error handling stuff in this instead of Logic.py
# e.g. "You must enter a number greater than 1"

def display_emergency_reshuffle_print():
    print("Deck ran out, emergency reshuffle")
    print("(adding 1 new deck)")

        # Will matter for card-counting stuff

def display_hand_print(hand, hidden=False): #Should this really be a return or should it print??
    if hidden == False:
        return ', '.join(f'{rank}{suit}' for rank, suit in hand)
    else:
        return f'{hand[0][0]}{hand[0][1]}, [X]'
    

def get_bet_print(cash, input = input, print = print):
    while True:
        try:
            print(f"Cash: {cash}")
            bet = int(input("Bet: "))
            if bet <= 0:
                print("Bet must be positive.")
            elif bet > cash:
                print("You don't have enough money for that bet.")
            else:
                return bet
        except ValueError:
            print("Please enter a valid number.")

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

def get_another_round_print():
    return input("\nPlay another round? (y/n): ").lower()
