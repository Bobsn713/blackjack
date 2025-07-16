import sys
import os
import builtins

# Add parent directory to path so we can import Play_v4
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import Deprecated.Play_v4 as blackjack

def test_multiple_splits(monkeypatch):
    # Mock a deck designed to force 3 splits (8,8,8,8...)
    test_deck = list(reversed([
        ('8', 'H'), ('8', 'D'),   # Player initial hand (8,8)
        ('8', 'C'), ('8', 'S'),   # Cards dealt to first and second hands
        ('8', 'H'), ('8', 'D'),   # Further split hands
        ('10', 'H'), ('7', 'D')   # Dealer cards
    ]))
    
    # Monkeypatch deck creation and shuffle
    monkeypatch.setattr(blackjack, 'create_deck', lambda num_decks=1: test_deck.copy())
    monkeypatch.setattr(blackjack, 'shuffle_deck', lambda deck: None)

    # Predefined responses to simulate:
    # - Bet: 10
    # - Split? y
    # - Each hand: stand
    # - Play another round? n
    inputs = iter([
        '10',   # Bet
        'y',    # Split
        'stand', 'stand', 'stand', 'stand',  # For up to 4 split hands
        'n'     # End game
    ])
    monkeypatch.setattr(builtins, 'input', lambda _: next(inputs))

    # Call the function under test
    starting_cash = blackjack.starting_cash
    deck = blackjack.create_deck()
    
    final_cash = blackjack.play_round(starting_cash, deck, sleep=False)

    print("\nSTARTING CASH:", starting_cash)
    print("ENDING CASH:", final_cash)
    print("CHANGE:", final_cash - starting_cash)

    # You could also assert something like:
    assert final_cash >= starting_cash - 40  # No total blowout
