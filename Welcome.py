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
time.sleep(.75)
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
        print("Type 'train'  or 't' to train a neural network to play blackjack.")
        print("Type 'quit' or 'q' to leave the program")
    elif mode_choice in ['play', 'p']:
        bj.play_game(
            sleep                        = True,

            display                      = print,
            display_hand                 = txt.display_hand_print,
            display_emergency_reshuffle  = txt.display_emergency_reshuffle_print,
            display_final_results        = txt.display_final_results_print,

            get_another_round            = txt.get_another_round_print,
            get_bet                      = txt.get_bet_print,
            get_split_choice             = txt.get_split_choice_print,
            get_hit_stand_dd             = txt.get_hit_stand_dd_print,
            get_card                     = bj.get_card_deal,

        )
    elif mode_choice in ['cheat', 'c']:
        game_or_round = None
        while True:
            print("Would you like to track bets and play with a consistent deck? (y/n)")
            game_or_round_input = input(">>> ").lower()
            print()

            if game_or_round_input in ['y', 'yes']:
                game_or_round = 'g'
                break
            elif game_or_round_input in ['n', 'no']: 
                game_or_round = 'r'
                break
            else: 
                print("Invalid Input")
                print()

        if game_or_round == 'g': 
            ch.play_game_cheat()
        elif game_or_round == 'r':
            run_again = True
            while run_again:
                ch.play_round_cheat()
                time.sleep(1)
                another_round = txt.get_another_round_print()
                print()
                if another_round != 'y':
                    print("Thanks for playing!")
                    run_again = False
        else: 
            raise NameError("Game or round somehow not selected")

    elif mode_choice in ['simulate', 's']:

        while True: 
            print("Which model would you like to use?")
            print("('hardcode' or 'imitation')")
            model = input(">>> ").lower() 
            print()

            if model in ['imit', 'imitation', 'hc', 'hardcode']:
                break
            else: 
                print("Please enter one of the options listed.")
                print()

        while True: 
            print("How many hands would you like to simulate?")
            try: 
                iterations = int(input(">>> "))
                print()
                break
            except: 
                print()
                print("Please enter an integer.")
                print()

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
pyfiglet.print_figlet("Thanks for playing!", font="speed")
time.sleep(2)
        
### Should I turn this into a function and add something like this? 
# if __name__ == "__main__":
#     run_text_mode()