import Play_v5 as bj

def chart_sorting(player_hand, dealer_hand):
    if player_hand[0][0] == player_hand[1][0]:
        return split_decision(player_hand, dealer_hand)
    elif player_hand[0][0] == 'A' or player_hand[1][0] == 'A':
        return soft_decision(player_hand, dealer_hand)  # Do I need to check for naturals here?
    else:
        return hard_decision(player_hand, dealer_hand)
    

#I think it's fine but it may be more like technically fair if these functions only get
# passed the dealer's upcard and not his whole hand (maybe more relevant for NN?)

def hard_decision(player_hand, dealer_hand):
    player_total = bj.hand_value(player_hand)
    dealer_upcard_rank = dealer_hand[0][0]

    # Decision logic from chart
    if player_total >= 17:
        return "Stand"
    elif player_total >= 13 and dealer_upcard_rank in [2, 3, 4, 5, 6]:
        return "Stand"
    elif player_total == 12 and dealer_upcard_rank in [4, 5, 6]:
        return "Stand"
    elif player_total == 11:
        return "Double Down"
    elif player_total == 10 and dealer_upcard_rank not in [10, 'A']:
        return "Double Down"
    elif player_total == 9 and dealer_upcard_rank in [3, 4, 5, 6]:
        return "Double Down"
    else:
        return "Hit"
    
def soft_decision(player_hand, dealer_hand):
    # This logic should work because pairs have already been sorted out
    player_ranks = [player_hand[0][0], player_hand[1][0]]
    a_index = player_ranks.index('A') 
    other_index = a_index - 1 # In a two item list switches 1 to 0 and 0 to -1 (which is 1)
    non_a_rank = player_hand[other_index][0]
    #I'm not sure if this needs to handle 3+ card hands, but if it does, I think I just add all the non As?
    #Don't know what to do with multiple Aces

    dealer_upcard_rank = dealer_hand[0][0]

    # Decision logic from chart
    if non_a_rank == '9':
        return "Stand"
    elif non_a_rank == '8' and dealer_upcard_rank == '6':
        return "Double Down"
    elif non_a_rank == '8':
        return "Stand"
    elif non_a_rank == '7' and dealer_upcard_rank in ['2', '3', '4', '5', '6']:
        return "Double Down"
    elif non_a_rank == '7' and dealer_upcard_rank in ['7', '8']:
        return "Stand"
    elif non_a_rank == '6' and dealer_upcard_rank in ['3', '4', '5', '6']:
        return "Double Down"
    elif non_a_rank in ['4', '5'] and dealer_upcard_rank in ['4', '5', '6']:
        return "Double Down"
    elif non_a_rank in ['2', '3'] and dealer_upcard_rank in ['5', '6']:
        return "Double Down"
    else:
        return "Hit"
    
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
        print("Don't Split")  
        return hard_decision(player_hand, dealer_hand)
    

# # Some basic testing
# print(chart_sorting([['10','H'],['6','H']], [['7','H'],['2','H']]))