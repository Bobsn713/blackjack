from dataclasses import dataclass, field
from typing import Callable, List, Tuple

@dataclass
class GameState: 
    # Global Config
    cash: int = 1000
    num_decks: int = 4
    card_counting: bool = True

    # Persistent State
    deck: List[Tuple[str,str]] = field(default_factory=list)
    cards_left: dict[str, int] = field(default_factory=dict)

    # Round Specific State
    player_hand: List[Tuple[str,str]] = field(default_factory=list)
    dealer_hand: List[Tuple[str,str]] = field(default_factory=list)
    bet: int = 1

@dataclass
class GameInterface: 
    display: Callable = print
    display_hand: Callable = None
    display_emergency_reshuffle: Callable = None
    display_final_results: Callable = None
    
    get_bet: Callable = None
    get_card: Callable = None
    get_split_choice: Callable = None
    get_hit_stand_dd: Callable = None
    get_another_round: Callable = None

    sleep: bool = True