import Logic as bj

def chart_sorting(player_hand, dealer_hand):
    if player_hand[0][0] == player_hand[1][0]:
        return split_decision(player_hand, dealer_hand)
    elif player_hand[0][0] == 'A' or player_hand[1][0] == 'A':
        return soft_decision(player_hand, dealer_hand)  # Do I need to check for naturals here?
    else:
        return hard_decision(player_hand, dealer_hand)
    

#I think it's fine but it may be more like technically fair if these functions only get
# passed the dealer's upcard and not his whole hand (maybe more relevant for NN?)

def hard_decision(player_hand, dealer_hand, can_double):
    player_total = bj.hand_value(player_hand)
    dealer_upcard_rank = dealer_hand[0][0]

    # Decision logic from chart
    if player_total >= 17:
        return "stand"
    elif player_total >= 13 and dealer_upcard_rank in [2, 3, 4, 5, 6]:
        return "stand"
    elif player_total == 12 and dealer_upcard_rank in [4, 5, 6]:
        return "stand"
    elif player_total == 11:
        return "double down" if can_double else "hit"
    elif player_total == 10 and dealer_upcard_rank not in [10, 'A']:
        return "double down" if can_double else "hit"
    elif player_total == 9 and dealer_upcard_rank in [3, 4, 5, 6]:
        return "double down" if can_double else "hit"
    else:
        return "hit"
    
def soft_decision(player_hand, dealer_hand, can_double, soft_total):
    dealer_upcard_rank = dealer_hand[0][0]
    print(f"Soft Total: {soft_total}")
    # Decision logic from chart
    if soft_total in [9, 10]:
        return "stand"
    elif soft_total == 8 and dealer_upcard_rank == '6':
        return "double down" if can_double else "stand"
    elif soft_total == 8:
        return "stand"
    elif soft_total == 7 and dealer_upcard_rank in ['2', '3', '4', '5', '6']:
        return "double down" if can_double else "stand"
    elif soft_total == 7 and dealer_upcard_rank in ['7', '8']:
        return "stand"
    elif soft_total == 6 and dealer_upcard_rank in ['3', '4', '5', '6']:
        return "double down" if can_double else "hit"
    elif soft_total in [4, 5] and dealer_upcard_rank in ['4', '5', '6']:
        return "double down" if can_double else "hit"
    elif soft_total in [2, 3] and dealer_upcard_rank in ['5', '6']:
        return "double down" if can_double else "hit"
    else:
        return "hit"
    
def split_decision(player_hand, dealer_hand):
    card = player_hand[0][0] #Either hand would do
    dealer_upcard_rank = dealer_hand[0][0]

    # Decision logic from chart
    if card == 'A' or card == '8':
        return "Split"
    elif card == '9' and (dealer_upcard_rank in ['2', '3', '4', '5', '6', '8', '9']):
        return "Split"
    elif card in ['7', '3', '2'] and int(dealer_upcard_rank) >= 7:
        return "Split"
    elif card == '6' and int(dealer_upcard_rank) <= 6:
        return "Split"
    elif card == '4' and dealer_upcard_rank in ['5', '6']:
        return "Split"
    else: 
        return "Don't Split"
    

# # Some basic testing
# print(chart_sorting([['10','H'],['6','H']], [['7','H'],['2','H']]))

#Defining input functions

def get_bet_hardcode(cash):
    return 100 #Automatically set the bet amount to 100

def get_split_choice_hardcode(player_hand, dealer_hand): 
    card = player_hand[0][0] #Either hand would do
    dealer_upcard_rank = dealer_hand[0][0]

    # Decision logic from chart
    if card == 'A' or card == '8':
        return "y"
    elif card == '9' and (dealer_upcard_rank in ['2', '3', '4', '5', '6', '8', '9']):
        return "y"
    elif card in ['7', '3', '2'] and int(dealer_upcard_rank) >= 7:
        return "y"
    elif card == '6' and int(dealer_upcard_rank) <= 6:
        return "y"
    elif card == '4' and dealer_upcard_rank in ['5', '6']:
        return "y"
    else: 
        return "n"
    
def is_soft_total(hand): #it's possible this makes more sense to put in the Logic file
    #Also, this is so similar to hand_value it's probably redundant somehow but this is the easiest fix
    total = bj.hand_value(hand)
    aces = 0

    for card in hand:
        if card[0] == 'A':
            aces += 1
    
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1

    return (aces > 0, total - 11)

    
def get_hit_stand_dd_hardcode(player_hand, dealer_hand, can_double):
    is_soft, soft_total = is_soft_total(player_hand)

    if is_soft:
        return soft_decision(player_hand, dealer_hand, can_double, soft_total)
    else:
        return hard_decision(player_hand, dealer_hand, can_double)
    
def get_another_round_hardcode():
    return input("\nPlay another round? (y/n): ").lower() #Keeping this for now, could make it like a for loop or something

