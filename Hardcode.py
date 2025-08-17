import Logic as bj

### STRATEGY FUNCTIONS

def hard_decision(player_hand, dealer_hand, can_double):
    player_total = bj.hand_value(player_hand)
    dealer_upcard_value = bj.card_value(dealer_hand[0])

    # Decision logic from chart
    if player_total >= 17:
        return "stand"
    elif player_total >= 13 and dealer_upcard_value in [2, 3, 4, 5, 6]:
        return "stand"
    elif player_total == 12 and dealer_upcard_value in [4, 5, 6]:
        return "stand"
    elif player_total == 11:
        return "double down" if can_double else "hit"
    elif player_total == 10 and dealer_upcard_value not in [10, 11]:
        return "double down" if can_double else "hit"
    elif player_total == 9 and dealer_upcard_value in [3, 4, 5, 6]:
        return "double down" if can_double else "hit"
    else:
        return "hit"
    
def soft_decision(player_hand, dealer_hand, can_double, soft_total):
    dealer_upcard_value = bj.card_value(dealer_hand[0])

    # Decision logic from chart
    if soft_total in [9, 10]:
        return "stand"
    elif soft_total == 8 and dealer_upcard_value == 6:
        return "double down" if can_double else "stand"
    elif soft_total == 8:
        return "stand"
    elif soft_total == 7 and dealer_upcard_value in [2, 3, 4, 5, 6]:
        return "double down" if can_double else "stand"
    elif soft_total == 7 and dealer_upcard_value in [7, 8]:
        return "stand"
    elif soft_total == 6 and dealer_upcard_value in [3, 4, 5, 6]:
        return "double down" if can_double else "hit"
    elif soft_total in [4, 5] and dealer_upcard_value in [4, 5, 6]:
        return "double down" if can_double else "hit"
    elif soft_total in [2, 3] and dealer_upcard_value in [5, 6]:
        return "double down" if can_double else "hit"
    else:
        return "hit"
    
def is_soft_total(hand): #it's possible this makes more sense to put in the Logic file
    #Also, this is so similar to hand_value it's probably redundant somehow but this is the easiest fix
    """Returns a tuple (is_soft, total) where total is either soft or hard depending on is_soft"""
    hand_ranks = []
    for card in hand:
        hand_ranks.append(bj.card_value(card))

    for i, card in enumerate(hand_ranks): 
        if sum(hand_ranks) > 21 and card == 11:
            hand_ranks[i] = 1

    if 11 in hand_ranks:
        is_soft = True
        total = sum(hand_ranks) - 11
    else: 
        is_soft = False
        total = sum(hand_ranks) #this is just the hard total
    
    return is_soft, total


# DISPLAY FUNCTIONS
def display_nothing_hardcode(*args, **kwargs): #Maybe worth renaming for generality, but it fits my current sytem
    pass #just doesn't print anything! That's it!

# GET FUNCTIONS

def get_another_round_hardcode():
    return input("\nPlay another round? (y/n): ").lower() #Keeping this for now, could make it like a for loop or something

def get_bet_hardcode(cash):
    return 100 #Automatically set the bet amount to 100

def get_split_choice_hardcode(player_hand, dealer_hand): 
    card = player_hand[0][0] #Either hand would do
    dealer_upcard_value = bj.card_value(dealer_hand[0]) #Is it confusing that this is the only place we do value instead of dealer upcard as a string? 

    # Decision logic from chart
    if card == 'A' or card == '8':
        return "y"
    elif card == '9' and (dealer_upcard_value in [2, 3, 4, 5, 6, 8, 9]):
        return "y"
    elif card in ['7', '3', '2'] and dealer_upcard_value <= 7:
        return "y"
    elif card == '6' and dealer_upcard_value <= 6:
        return "y"
    elif card == '4' and dealer_upcard_value in [5, 6]:
        return "y"
    else: 
        return "n"
    
def get_hit_stand_dd_hardcode(player_hand, dealer_hand, can_double):
    is_soft, soft_total = is_soft_total(player_hand)

    if is_soft:
        return soft_decision(player_hand, dealer_hand, can_double, soft_total)
    else:
        return hard_decision(player_hand, dealer_hand, can_double)

