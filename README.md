# README

## Blackjack Engine Overview
There are 4 ways to use this Blackjack Engine. 
1. It can be used to play a text-based blackjack game, with the computer serving as dealer and scorekeeper. 
2. It can advise you on how to play any given hand. 
3. It can simulate one of the pre-trained models playing an arbitrary number of rounds, tracking various performance metrics. 
4. Eventually, it will allow a user to train a new model themselves, which they can then simulate. 

I developed this engine as an excuse to learn to build neural networks, and built up the other functionality along the way. 

## Instructions for Use and Organization
The most user friendly way to use this engine is by running the `Welcome.py` file. From here, users can choose which of the 4 functionalities they would like to use. 

If you want to get further under the hood, here is how the rest of the project is structured: 
* `Logic.py` is where the primary functions relating to gameplay are defined. It was first designed around the text-based gameplay, and still reflects this organization, though I have attempted to make it more general and the other programs run on it with modfications through dependency injection
* `Text.py` defines functions related to the text-based gameplay
* `Hardcode.py` defines functions related to perfect play, related to the advice/cheat functionality
* `Imitation.py` defines and runs a neural network trained not by reinforcement learning but by "copying" the perfect player defined in `Hardcode.py`. The work to generate and train this family of models is in the `/Build_Imitation` folder
* `Performance_Tracker.py` defines a function to simulate iterated performance of a model and report performance metrics
* `Scrapwork.py` is just a place for me to think through and try things out

## Rules
It is not yet possible to customize the ruleset. The current rules are:

* doubling down is allowed,
* splitting is allowed,
* maximum 3 splits (4 hands),
* doubling down after splitting is allowed,
* if aces are split, only one card is dealt to each hand
* no insurance or surrender,
* dealer hits on a soft 17 (H17),

## To-Do's
As of right now there are some more short-term and more long-term to-do's. 

**Short Term**
1. Improve `play_round_cheat()` (currently called `test_play_round()`) 
    * URGENT: Dealer card is asked for before ==Dealer's Turn== Screen (There is no good time to ask for a dealer's card if it's not already given (this is true for the Blackjack as well), maybe that should be handled in the the display final results function?)
    * Print Totals along the way? 
    * Play with what needs to happen in `Welcome.py` vs the `test_play_round()` function and also with shifting to a `play_game()` instead of `play_round()` model
    * Longer term, this should be integrated into a play game function
2. Standardize and improve the visuals for both the text-based gameplay and the cheat gameplay
    * I think the best way to think of this is as individual frames, separated by something like headers or horizontal lines and stalled on with time.sleeps
3. Add a loading bar to the perfomance tracker
4. Improve error handling for model entry in 'simulate' functionality. 
5. Add p-values/measures of variance to performance tracker
6. Simplify dependency injection
7. Add betting strategies to the simulation
8. Consider making sleep parameter accept sleep times rather than being boolean
9. Better cosmetics for the outtro
    
**Long Term**
0. Semi-long term, building out the model training functionality. 
1. Integrate card counting strategies. This will include alowing for dynamically changing some functions from a play_round model to a play_game model. 
    * Note: This includes implementing deck tracking for the 'cheat' gamemode
2. Make rules customizable. 
3. Refactoring `Logic.py` to make it less _default textbased_
4. Maybe making a GUI. 

## Sources

Source for charts:
<https://www.blackjackapprenticeship.com/blackjack-strategy-charts/>

Source for blackjack rules:
<https://bicyclecards.com/how-to-play/blackjack>
