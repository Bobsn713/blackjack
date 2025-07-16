import pytest
import sys
import os
from unittest.mock import patch, MagicMock
from io import StringIO
import io
import contextlib

# Add the directory containing your blackjack code to the path
# Adjust this path based on where your blackjack.py file is located
sys.path.insert(0, os.path.dirname("/Users/brantwilson/Python/Personal Projects/Blackjack/"))

# Import your blackjack functions
# Replace 'blackjack' with the actual name of your Python file
from Deprecated.Play_v4 import (
    create_deck, shuffle_deck, deal_card, card_value, hand_value, 
    display_hand, can_split, play_individual_hand, get_bet
)

class TestDeckFunctions:
    """Test deck creation, shuffling, and card dealing"""
    
    def test_create_deck_single(self):
        """Test creating a single deck"""
        deck = create_deck(1)
        assert len(deck) == 52
        # Check all suits and ranks are present
        suits = set(card[1] for card in deck)
        ranks = set(card[0] for card in deck)
        assert suits == {'H', 'D', 'C', 'S'}
        assert ranks == {'A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'}
    
    def test_create_deck_multiple(self):
        """Test creating multiple decks"""
        deck = create_deck(2)
        assert len(deck) == 104
        # Should have 8 aces (2 decks * 4 suits)
        aces = [card for card in deck if card[0] == 'A']
        assert len(aces) == 8
    
    def test_deal_card(self):
        """Test dealing cards from deck"""
        deck = create_deck(1)
        original_length = len(deck)
        
        card = deal_card(deck)
        assert len(deck) == original_length - 1
        assert len(card) == 2  # Should be a tuple (rank, suit)
        assert card[0] in ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        assert card[1] in ['H', 'D', 'C', 'S']
    
    def test_deal_card_empty_deck(self):
        """Test dealing from an empty deck triggers emergency reshuffle"""
        empty_deck = []
        with patch('builtins.print') as mock_print, \
             patch('Play_v4.create_deck', return_value=[('A', 'S'), ('K', 'D')]*26) as mock_create_deck:
            card = deal_card(empty_deck)
            # Should print emergency reshuffle message
            mock_print.assert_called()
            mock_create_deck.assert_called()
        
        # Deck should now have cards after emergency reshuffle
        assert len(empty_deck) > 0

class TestCardValues:
    """Test card value calculations"""
    
    def test_card_value_numbers(self):
        """Test numeric card values"""
        assert card_value(('2', 'H')) == 2
        assert card_value(('5', 'D')) == 5
        assert card_value(('10', 'C')) == 10
    
    def test_card_value_face_cards(self):
        """Test face card values"""
        assert card_value(('J', 'H')) == 10
        assert card_value(('Q', 'D')) == 10
        assert card_value(('K', 'C')) == 10
    
    def test_card_value_ace(self):
        """Test ace value (initially 11)"""
        assert card_value(('A', 'S')) == 11
    
    def test_hand_value_simple(self):
        """Test simple hand values without aces"""
        hand = [('5', 'H'), ('7', 'D')]
        assert hand_value(hand) == 12
        
        hand = [('K', 'H'), ('Q', 'D')]
        assert hand_value(hand) == 20
    
    def test_hand_value_with_aces(self):
        """Test hand values with aces"""
        # Ace + 5 = 16 (ace as 11)
        hand = [('A', 'H'), ('5', 'D')]
        assert hand_value(hand) == 16
        
        # Ace + King = 21 (blackjack)
        hand = [('A', 'H'), ('K', 'D')]
        assert hand_value(hand) == 21
        
        # Ace + 6 + 9 = 16 (ace as 1)
        hand = [('A', 'H'), ('6', 'D'), ('9', 'C')]
        assert hand_value(hand) == 16
    
    def test_hand_value_multiple_aces(self):
        """Test hand values with multiple aces"""
        # Two aces = 12 (one as 11, one as 1)
        hand = [('A', 'H'), ('A', 'D')]
        assert hand_value(hand) == 12
        
        # Three aces = 13 (one as 11, two as 1)
        hand = [('A', 'H'), ('A', 'D'), ('A', 'C')]
        assert hand_value(hand) == 13

class TestDisplayFunctions:
    """Test display and formatting functions"""
    
    def test_display_hand_normal(self):
        """Test normal hand display"""
        hand = [('A', 'H'), ('K', 'D')]
        result = display_hand(hand)
        assert result == 'AH, KD'
    
    def test_display_hand_hidden(self):
        """Test hidden hand display (for dealer)"""
        hand = [('A', 'H'), ('K', 'D')]
        result = display_hand(hand, hidden=True)
        assert result == 'AH, [X]'
    
    def test_can_split_same_rank(self):
        """Test splitting with same rank cards"""
        hand = [('8', 'H'), ('8', 'D')]
        assert can_split(hand) == True
        
        hand = [('K', 'H'), ('K', 'D')]
        assert can_split(hand) == True
    
    def test_can_split_different_ranks(self):
        """Test splitting with different rank cards"""
        hand = [('8', 'H'), ('9', 'D')]
        assert can_split(hand) == False
    
    def test_can_split_wrong_hand_size(self):
        """Test splitting with wrong hand size"""
        hand = [('8', 'H'), ('8', 'D'), ('5', 'C')]
        assert can_split(hand) == False
        
        hand = [('8', 'H')]
        assert can_split(hand) == False

class TestBettingFunction:
    """Test betting input validation"""
    
    @patch('builtins.input')
    def test_get_bet_valid(self, mock_input):
        """Test valid bet input"""
        mock_input.return_value = '50'
        cash = 100
        bet = get_bet(cash)
        assert bet == 50
    
    @patch('builtins.input')
    @patch('builtins.print')
    def test_get_bet_too_high(self, mock_print, mock_input):
        """Test bet higher than available cash"""
        mock_input.side_effect = ['150', '50']  # First too high, then valid
        cash = 100
        bet = get_bet(cash)
        assert bet == 50
        # Should print error message
        mock_print.assert_any_call("You don't have enough money for that bet.")
    
    @patch('builtins.input')
    @patch('builtins.print')
    def test_get_bet_negative(self, mock_print, mock_input):
        """Test negative bet"""
        mock_input.side_effect = ['-10', '50']  # First negative, then valid
        cash = 100
        bet = get_bet(cash)
        assert bet == 50
        mock_print.assert_any_call("Bet must be positive.")
    
    @patch('builtins.input')
    @patch('builtins.print')
    def test_get_bet_invalid_input(self, mock_print, mock_input):
        """Test invalid input (not a number)"""
        mock_input.side_effect = ['abc', '50']  # First invalid, then valid
        cash = 100
        bet = get_bet(cash)
        assert bet == 50
        mock_print.assert_any_call("Please enter a valid number.")

class TestGameLogic:
    """Test game logic functions"""
    
    def test_play_individual_hand_bust(self):
        """Test individual hand that busts"""
        hand = [('K', 'H'), ('Q', 'D')]  # 20
        deck = create_deck(1)
        bet = 10
        cash = 100
        dealer_hand = [('5', 'H'), ('6', 'D')]
        
        # Mock input to hit and get a card that busts
        with patch('builtins.input', return_value='hit'):
            with patch('builtins.print'):  # Suppress output
                # Force a bust by adding a high card
                hand.append(('5', 'C'))  # This would make it 25, busting
                # We'll need to modify the test to work with the actual function
                # This is a simplified test - in practice you'd need to mock deal_card
                pass
    
    def test_play_individual_hand_stand(self):
        """Test individual hand that stands"""
        hand = [('K', 'H'), ('Q', 'D')]  # 20
        deck = create_deck(1)
        bet = 10
        cash = 100
        dealer_hand = [('5', 'H'), ('6', 'D')]
        
        with patch('builtins.input', return_value='stand'):
            with patch('builtins.print'):
                result = play_individual_hand(hand, deck, bet, cash, dealer_hand)
                assert result['result'] == 'stand'
                assert result['bet_multiplier'] == 1
                assert result['final'] == False

# Integration tests
class TestIntegration:
    """Integration tests for multiple functions working together"""
    
    def test_full_deck_cycle(self):
        """Test that we can deal all cards from a deck"""
        deck = create_deck(1)
        dealt_cards = []
        
        for _ in range(52):
            card = deal_card(deck)
            dealt_cards.append(card)
        
        assert len(dealt_cards) == 52
        assert len(deck) == 0
        
        # All cards should be unique
        assert len(set(dealt_cards)) == 52
    
    def test_blackjack_detection(self):
        """Test detecting blackjack hands"""
        # Ace + 10-value card = blackjack
        blackjack_hands = [
            [('A', 'H'), ('K', 'D')],
            [('A', 'C'), ('Q', 'H')],
            [('A', 'S'), ('J', 'C')],
            [('A', 'D'), ('10', 'H')]
        ]
        
        for hand in blackjack_hands:
            assert hand_value(hand) == 21
    
    def test_soft_seventeen(self):
        """Test soft 17 scenarios"""
        # Ace + 6 = soft 17
        hand = [('A', 'H'), ('6', 'D')]
        assert hand_value(hand) == 17
        
        # Ace + 6 + 5 = 12 (ace becomes 1)
        hand = [('A', 'H'), ('6', 'D'), ('5', 'C')]
        assert hand_value(hand) == 12

class TestSplitting:
    #Copilot wrote this test and I don't think it works at all. Hopefully that doesn't mean all the tests are messed up. 
    """Test splitting logic, including multiple splits in a row"""
    def test_multiple_splits(self):
        from Deprecated.Play_v4 import play_round, create_deck
        # The last items in the list are dealt first (deck.pop()), so put the initial hands at the end
        deck = [
            # Cards for splits and hits
            ('10', 'S'), ('10', 'C'), ('10', 'D'), ('10', 'H'),
            ('8', 'S'), ('8', 'C'),
            # Dealer hand
            ('9', 'C'), ('5', 'S'),
            # Player initial hand (dealt first)
            ('8', 'D'), ('8', 'H')
        ]
        deck += create_deck(1)
        user_inputs = iter(['100', 'y', 'y', 'y', 'stand', 'stand', 'stand', 'stand'])
        def mock_input(prompt):
            return next(user_inputs)
        with patch('builtins.input', mock_input):
            cash = 1000
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                result_cash = play_round(cash, deck, sleep=False)
                # Print the first four cards actually dealt for debugging
                print('DEBUG: First four cards dealt:', deck[-4:])
            output = f.getvalue()
            print(output)  # Will show output if you run pytest -s
            assert isinstance(result_cash, int)

if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])