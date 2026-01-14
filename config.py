# Since I can't fit all my configs in here, I probably will delete it and put everything in its own parent file

import game_logic as gl
import text as tx
import basic_strategy as bs
import cheat as ch
from base import GameState, GameInterface

text_mode = GameInterface(
    # Display functions
    display                     = print, 
    display_hand                = tx.display_hand_print,
    display_emergency_reshuffle = tx.display_emergency_reshuffle_print,
    display_final_results       = tx.display_final_results_print,

    # Get Functions
    get_bet                     = tx.get_bet_print,
    get_card                    = gl.get_card_deal,
    get_split_choice            = tx.get_split_choice_print,
    get_hit_stand_dd            = tx.get_hit_stand_dd_print,
    get_another_round           = tx.get_another_round_print,

    # Other
    sleep = True
)

hardcode_mode = GameInterface(
    # Display functions
    display                     = print,
    display_hand                = tx.display_hand_print, # or bs.display_hand_hardcode
    display_emergency_reshuffle = tx.display_emergency_reshuffle_print, # or bs.emergency_reshuffle_hardcode
    display_final_results       = tx.display_final_results_print,

    # Get Functions
    get_bet                     = bs.get_bet_hardcode,
    get_card                    = gl.get_card_deal,
    get_split_choice            = bs.get_split_choice_hardcode,
    get_hit_stand_dd            = bs.get_hit_stand_dd_hardcode,
    get_another_round           = bs.get_another_round_hardcode,

    # Other
    sleep = False
)

cheat_mode = GameInterface(
    # Display functions
    display                     = print,
    display_hand                = tx.display_hand_print,
    display_emergency_reshuffle = tx.display_emergency_reshuffle_print,
    display_final_results       = tx.display_final_results_print,

    # Get Functions
    get_bet                      = tx.get_bet_print,
    get_card                     = ch.get_card_cheat,
    get_split_choice             = ch.get_split_choice_cheat,
    get_hit_stand_dd             = ch.get_hit_stand_dd_cheat,
    get_another_round            = tx.get_another_round_print,

    # Other
    sleep = True
)