
### 3 FUNCTIONS TO CHECK FOR BUSTS 
            # These also work as total calculators, because if you don't go bust then they return a total
            # Maybe I should change them so theyre just totals and deal with the busts elsewhere. 
def ace_numerator(hand):
    ace_counter = 0
    total = 0
    for card in hand:
        if card == "A":
            total += 11
            ace_counter += 1
        else: 
            total += card
    
    return total, ace_counter

def ace_adjuster(ace_tuple):
    total, ace_counter = ace_tuple
    
    if total > 21:
        if ace_counter > 0:
            total -= 10
            ace_counter -= 1
            return ace_adjuster((total, ace_counter))
        else: 
            return "Bust"
    else: 
        return total

def is_bust(hand):
    return ace_adjuster(ace_numerator(hand))




# RESOLVE

# #When the resolve function is called, the player will not hit again, and the player has not busted

def resolve(p_hand, d_hand):
    print(d_hand)
    bust = is_bust(d_hand)
    print(bust)
    if bust == "Bust":
        print("Dealer Busts")
    elif bust <= 17:
        print("Dealer Hits")
        d_hand.append(5) ###### EDIT THIS! THIS SHOULD NOT STAY AS JUST A 5!!!!
        resolve(p_hand, d_hand) 
    else: 
        print("Dealer Stays with: ", bust)


# Originally everything above this was a different function, "dealer", and "resolve" was going to be
# what's below this, but the easiest thing I saw was combining them. Maybe worth changing eventually?
        if bust > is_bust(p_hand):
            print("Dealer Wins")
        elif bust < is_bust(p_hand):
            print("Player Wins")
        else: 
            print("Push")

