# This is supposed to help a user unfamiliar with the code to run it. 
import pyfiglet
import time
import Logic as l
import Text as txt
import Performance_Tracker as pt

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
        l.play_game(
            get_another_round            = txt.get_another_round_print,
            display                      = print,
            get_bet                      = txt.get_bet_print,
            get_split_choice             = txt.get_split_choice_print,
            get_hit_stand_dd             = txt.get_hit_stand_dd_print,
            display_hand                 = txt.display_hand_print,
            display_emergency_reshuffle  = txt.display_emergency_reshuffle_print,
            sleep                        = True,
            display_final_results        = txt.display_final_results_print
        )
    elif mode_choice in ['cheat', 'c']:
        print("I'll come back to this shortly")
    elif mode_choice in ['simulate', 's']:
        # Placeholders...
        print("Model: ")
        print("# of Iterations: ")
        # Look into tqdm progress bar
        #pt.performance_tracker()
    elif mode_choice in ['train', 't']:
        print("I'll come back to this shortly")
    elif mode_choice in ['quit', 'q']:
        keep_playing = False
    else: 
        print("Please enter a valid choice. Type 'help' for more information.")

print("Thanks for playing!")
clear()
        
