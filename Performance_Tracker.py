# I was getting import circularity errors so I'm making a separate Performance tracker file
import Logic as bj
import Hardcode as hc
import Text as text
import matplotlib.pyplot as plt

def performance_tracker():
    iterations = 10000
    cash = 1000
    outcomes = []

    for i in range(iterations):     
        deck = bj.create_deck()
        bj.shuffle_deck(deck)

        outcome_raw = bj.play_round(
            cash = cash, #infinite cash relative to bet size
            deck = deck, 
            sleep = False, 
            get_bet = lambda cash: 1, #minimal bet size, the lambda is so its callable to avoid an error
            get_split_choice = hc.get_split_choice_hardcode, 
            display = hc.display_hardcode, 
            get_hit_stand_dd = hc.get_hit_stand_dd_hardcode, 
            display_hand = text.display_hand_print, # I think as long as the display function is empty this shouldn't print
            display_emergency_reshuffle = text.display_emergency_reshuffle_print #Ditto
        )

        outcome = outcome_raw - cash
        outcomes.append(outcome)

    cumulative_outcome = sum(outcomes)
    expected_return_per_hand = cumulative_outcome / iterations

    won = 0
    push = 0
    lost = 0
    blackjack = 0
    doubled_down = 0 #this will be crude because split hands will be in here too
    doubled_down_won = 0
    doubled_down_lost = 0


    for i, outcome in enumerate(outcomes):
        if i == 0:
            running_total = [outcome]
        else: 
            running_total.append(running_total[i-1] + outcome) 
        if outcome > 0:
            won += 1

            if outcome == 1.5:
                blackjack += 1
            elif outcome == 2:
                doubled_down += 1
                doubled_down_won += 1
        elif outcome == 0:
            push += 1
        else: 
            lost += 1

            if outcome == -2:
                doubled_down += 1
                doubled_down_lost += 1

    doubled_down_push = doubled_down - doubled_down_won - doubled_down_lost

    perc_won = round((won / iterations) * 100, 2)
    perc_push = round((push / iterations) * 100, 2)
    perc_lost = round((lost / iterations) * 100, 2)

    perc_blackjack = round((blackjack / iterations) * 100, 2)
    perc_doubled_down = round((doubled_down / iterations) * 100, 2)
    perc_doubled_down_won = round((doubled_down_won / doubled_down) * 100, 2)
    perc_doubled_down_lost = round((doubled_down_lost / doubled_down) * 100, 2)

    print(f"Results of {iterations} iterations...")
    print(f"Cumulative Outcome: {cumulative_outcome}")
    print(f"Expected Return (per hand): {expected_return_per_hand}")
    print(f"Won: {won}   Push: {push}   Lost: {lost}")
    print(f"BJ: {blackjack}   DD: {doubled_down}   DD Won: {doubled_down_won}   DD Lost: {doubled_down_lost}")
    print()
    print("Percentages:")
    print(f"Won: {perc_won}%   Push: {perc_push}%   Lost: {perc_lost}%")
    print(f"BJ: {perc_blackjack}%   DD: {perc_doubled_down}%   DD Won: {perc_doubled_down_won}%   DD Lost: {perc_doubled_down_lost}%")

    print(f"\n First 100 Outcomes: {outcomes[:100]}")
    print(f"\n First 100 Running Total: {running_total[:100]}")

    plt.plot(running_total)
    plt.show()

performance_tracker()