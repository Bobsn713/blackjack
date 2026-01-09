# This is the most user friendly way to run the code here
import pyfiglet
import time
import game_logic as bj
import text1 as txt
import performance_tracker1 as pt
import cheat1 as ch
import train_nn as tr
from config import text_mode, hardcode_mode, cheat_mode
from base import GameState

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
            state = GameState()
            bj.play_game(state, text_mode)
        elif mode_choice in ['cheat', 'c']:
            state = GameState()
            bj.play_game(state, cheat_mode)
        elif mode_choice in ['simulate', 's']:
            txt.print_title_box(["ENTERING SIMULATION MODE..."])
            pt.choose_model()
        elif mode_choice in ['train', 't']:
            txt.print_title_box(["ENTERING TRAINING MODE..."])
            tr.train_model()
        elif mode_choice in ['quit', 'q']:
            keep_playing = False
        else: 
            print("Please enter a valid choice. Type 'help' for more information.")

    print('\n'*6)
    pyfiglet.print_figlet("Thanks for playing!", font="speed")
    time.sleep(2)
        
if __name__ == "__main__":
    main_loop()