# This is the most user friendly way to run the code here
import pyfiglet
import time
import Logic as bj
import Text as txt
import Performance_Tracker as pt
import Cheat as ch
import Train as tr

def clear():
        print("\n" * 30)

def main_loop():
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
            bj.play_game()
        elif mode_choice in ['cheat', 'c']:
            ch.play_game_cheat()
        elif mode_choice in ['simulate', 's']:
            txt.print_title_box(["ENTERING SIMULATION MODE..."])
            print('\n')
            while True: 
                print("Which model would you like to use?")
                print("Defaults: hardcode, imitation")
                custom_models = tr.get_models()
                display_names = [name.split('.')[0] for name in custom_models]
                names_string = ", ".join(display_names)
                print(f"Custom Models: {names_string}")
                model = input(">>> ").lower() 
                print()

                if model in ['imit', 'imitation', 'hc', 'hardcode']:
                    break
                elif model in custom_models:
                    break
                elif model in display_names:
                    model = f"{model}.pt"
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

            print()
            print()
            print('\033[3mSimulation Complete.\033[0m')
            print('-'*80)
            print()
        elif mode_choice in ['train', 't']:
            txt.print_title_box(["ENTERING TRAINING MODE..."])
            tr.train_model()

            # message = "Sorry, this functionality isn't quite ready yet."
            # len_mes = len(message)
            # buffer = int((80 - len(message))/2)
            # print("#"*80)
            # print(" " * buffer, message, " " * buffer)
            # print("#"*80)
        elif mode_choice in ['quit', 'q']:
            keep_playing = False
        else: 
            print("Please enter a valid choice. Type 'help' for more information.")

    print('\n'*6)
    pyfiglet.print_figlet("Thanks for playing!", font="speed")
    time.sleep(2)
        
if __name__ == "__main__":
    main_loop()