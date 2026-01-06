# Since I can't fit all my configs in here, I probably will delete it and put everything in its own parent file

import Logic as bj
import Text as txt
import Hardcode as hc
import Cheat as ch
from base import GameState, GameInterface

text_mode = GameInterface(
    # Display functions
    display                     = print, 
    display_hand                = txt.display_hand_print,
    display_emergency_reshuffle = txt.display_emergency_reshuffle_print,
    display_final_results       = txt.display_final_results_print,

    # Get Functions
    get_bet                     = txt.get_bet_print,
    get_card                    = bj.get_card_deal,
    get_split_choice            = txt.get_split_choice_print,
    get_hit_stand_dd            = txt.get_hit_stand_dd_print,
    get_another_round           = txt.get_another_round_print,

    # Other
    sleep = True
)

hardcode_mode = GameInterface(
    # Display functions
    display                     = print,
    display_hand                = txt.display_hand_print, # or hc.display_hand_hardcode
    display_emergency_reshuffle = txt.display_emergency_reshuffle_print, # or hc.emergency_reshuffle_hardcode
    display_final_results       = txt.display_final_results_print,

    # Get Functions
    get_bet                     = hc.get_bet_hardcode,
    get_card                    = bj.get_card_deal,
    get_split_choice            = hc.get_split_choice_hardcode,
    get_hit_stand_dd            = hc.get_hit_stand_dd_hardcode,
    get_another_round           = hc.get_another_round_hardcode,

    # Other
    sleep = False
)

cheat_mode = GameInterface(
    # Display functions
    display                     = print,
    display_hand                = txt.display_hand_print,
    display_emergency_reshuffle = txt.display_emergency_reshuffle_print,
    display_final_results       = txt.display_final_results_print,

    # Get Functions
    get_bet                      = txt.get_bet_print,
    get_card                     = ch.get_card_cheat,
    get_split_choice             = ch.get_split_choice_cheat,
    get_hit_stand_dd             = ch.get_hit_stand_dd_cheat,
    get_another_round            = txt.get_another_round_print,

    # Other
    sleep = True
)