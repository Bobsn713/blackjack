#This file will be where functions are defined that interact with Logic.py
# and allow it to be played as a text based terminal game


### -------------------------------------------------------- ###
def display(msg): #This is redundant I'm just putting it here now so I don't have to edit everything I'm copying in
    print(msg)

def get_input(msg): #As above
    return input(msg)

### -------------------------------------------------------- ###

def emergency_reshuffle():
        display("Deck ran out, emergency reshuffle")
        display("(adding 1 new deck)")

        # Will matter for card-counting stuff

def display_hand(hand, hidden=False): #Should this really be a return or should it print??
    if hidden == False:
        return ', '.join(f'{rank}{suit}' for rank, suit in hand)
    else:
        return f'{hand[0][0]}{hand[0][1]}, [X]'
    

def get_bet(cash, get_input = input, display = print):
    while True:
        try:
            display(f"Cash: {cash}")
            bet = int(get_input("Bet: "))
            if bet <= 0:
                display("Bet must be positive.")
            elif bet > cash:
                display("You don't have enough money for that bet.")
            else:
                return bet
        except ValueError:
            display("Please enter a valid number.")

def get_hit_stand_dd(hand, cash, bet):
    can_double = len(hand) == 2 and cash >= 2 * bet

    if can_double:
        prompt = "\nDo you want to hit, stand, or double down? "
    else:
        prompt = "\nDo you want to hit or stand? "
    h_or_s = get_input(prompt).lower()
