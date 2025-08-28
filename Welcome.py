# This is supposed to help a user unfamiliar with the code to run it. 
import pyfiglet
import time
import Logic as bj
import Text as txt
import Performance_Tracker as pt
import Cheat as ch

def clear():
    print("\n" * 30)

clear()
time.sleep(0.5)
pyfiglet.print_figlet("Welcome", font = "speed")
time.sleep(.5)
pyfiglet.print_figlet("to...", font = 'speed')
time.sleep(.5)
pyfiglet.print_figlet("BLACK JACK!", font = 'speed')
time.sleep(1)
print("-"*80)

keep_playing = True

while keep_playing: 
    print("\nWould you like to play, cheat, simulate, or train?")
    print("(type 'help' for more information)")
    mode_choice = input(">>> ").lower()
    print("\n")
    if mode_choice in ['help', 'h']:
        print("Type 'play' or 'p' to play a text based game of blackjack, where the computer is the dealer.")
        print("Type 'cheat' or 'c' to input your current game state and have the computer recommend to you the best move.")
        print("Type 'simulate' or 's' to iteratively run one of our blackjack play models, and analyze its performance.")
        print("Type 'train' to train a neural network to play blackjack.")
        print("Type 'quit' or 'q' to leave the program")
    elif mode_choice in ['play', 'p']:
        print("="*80)
        print("New Game")
        print("="*80)
        print()
        bj.play_game(
            get_another_round            = txt.get_another_round_print,
            display                      = print,
            get_bet                      = txt.get_bet_print,
            get_split_choice             = txt.get_split_choice_print,
            get_hit_stand_dd             = txt.get_hit_stand_dd_print,
            get_card                     = bj.deal_card,
            display_hand                 = txt.display_hand_print,
            display_emergency_reshuffle  = txt.display_emergency_reshuffle_print,
            sleep                        = True,
            display_final_results        = txt.display_final_results_print
        )
    elif mode_choice in ['cheat', 'c']:
        run_again = True
        while run_again:
            ch.primitive_play_round_cheat()
            time.sleep(1)
            another_round = txt.get_another_round_print()
            print()
            if another_round != 'y':
                print("Thanks for playing!")
                run_again = False

        # INCOMPLETE
        # Should the player be able to pick between hand by hand or whole game cheat cycles? 
        # Print Prettier
        # Celebrate Blackjack?
        # Handle input busts?
        # Print totals?
        # Make splits and hits cumulative instead of starting from scratch
    elif mode_choice in ['simulate', 's']:
        model = input("Model: ") # Make input handling robust
        iterations = int(input("# of Iterations: ")) # Make input handling robust
        pt.performance_tracker(model,iterations)
    elif mode_choice in ['train', 't']:
        message = "Sorry, this functionality isn't quite ready yet."
        len_mes = len(message)
        buffer = int((80 - len(message))/2)
        print("#"*80)
        print(" " * buffer, message, " " * buffer)
        print("#"*80)
    elif mode_choice in ['quit', 'q']:
        keep_playing = False
    else: 
        print("Please enter a valid choice. Type 'help' for more information.")

print('\n'*6)
print("Thanks for playing!")
print('\n'*6)
time.sleep(2)
        
### Should I turn this into a function and add something like this? 
# if __name__ == "__main__":
#     run_text_mode()