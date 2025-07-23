import Logic as bj
import Hardcode as hc


def split_test():
    deck_1 = [(rank, 'H') for rank in bj.ranks]
    #deck_2 = bj.create_deck()

    deck_3 = [(rank, 'H') for rank in bj.ranks]
    #deck_4 = bj.create_deck()

    for card_1 in deck_1:
        #for card_2 in deck_2:
        for card_3 in deck_3:
            #for card_4 in deck_4:
            player_hand = [card_1, card_1]
            dealer_hand = [card_3, card_3]

            result = hc.get_split_choice_hardcode(player_hand, dealer_hand)

            print(f"Player: {player_hand[0][0]}  Dealer: {card_3[0]}   Result: {result} ")

# split_test()
# SPLIT WORKS

def soft_test():
    deck_1 = [(rank, 'H') for rank in bj.ranks]
    deck_2 = [(rank, 'H') for rank in bj.ranks]

    for card_1 in deck_1:
        player_hand = [card_1, ('A', 'H')]
        for card_2 in deck_2:
            dealer_hand = [card_2, card_2]

            is_soft, soft_total = hc.is_soft_total(player_hand)

            result = hc.soft_decision(player_hand, dealer_hand, False, soft_total)

            print(f"Player: {card_1[0]}, A ({soft_total})  Dealer: {card_2[0]}   Result: {result}")

# Soft Test with two card hands and can_split = True WORKS
# Soft Test with two card hands and can_split = False WORKS
#soft_test()

# DEPRECATED
# def soft_test_2():
#     deck_1 = [(rank, 'H') for rank in bj.ranks]
#     deck_2 = [(rank, 'H') for rank in bj.ranks]
#     deck_3 = [(rank, 'H') for rank in bj.ranks]

#     for card_1 in deck_1:
#         for card_2 in deck_2:
#             player_hand = [('A', 'H'), card_1, card_2]

#             for card_3 in deck_3:
#                 dealer_hand = [card_3, card_3]

#                 is_soft, soft_total = hc.is_soft_total(player_hand)


#                 result = hc.soft_decision(player_hand, dealer_hand, True, soft_total)

#                 print(f"Player: {'A'}, {card_1[0]}, {card_2[0]} ({soft_total}, {is_soft}) Dealer {card_3[0]}  Result: {result}")

# soft_test_2()

def soft_test_3():
    player_hand = [('A', 'H'), ('2', 'H'), ('3', 'H'), ('A', 'H')]
    
    deck_1 = [(rank, 'H') for rank in bj.ranks]
    for card_1 in deck_1:
        dealer_hand = [card_1, card_1]

        is_soft, soft_total = hc.is_soft_total(player_hand)

        result = hc.soft_decision(player_hand, dealer_hand, True, soft_total)
        print(f"Player: {player_hand[0][0], player_hand[1][0], player_hand[2][0]} {soft_total, is_soft}  Dealer: {card_1[0]}   Result: {result}")

soft_test_3()

# So this seems like its working on multi-card hands that should be soft,
# But multi-card hands that should be hard are being labeled as soft, so I need to test "is_soft"

#SOFT DECISION SEEMS TO BE WORKING (I havent tested every possible thing but the logic seems right)

# print(hc.is_soft_total([('A', 'H'), ('2', 'H'), ('3', 'H'), ('A', 'H')]))
# print(hc.is_soft_total([('A', 'H'), ('2', 'H'), ('10', 'H'), ('A', 'H')]))
# IS_SOFT is WORKING, now I can go back to other tests


#I don't really want to test hardcode right now I think it works