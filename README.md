# README

## Goal

The goal of this project is first of all to "hardcode" a program that can tell me what to do in blackjack, and second of all to train a neural net to learn on its own.

## Structure

**Logic.py** is the brains of the game, where the fundamental gameplay mechanics are defined and stored. 

**Text.py** will be where the text based terminal game can be played.

**Hardcode.py** will be where the brains are that can play each hand perfectly.

**Neural Net.py** will be, predictably, where I train the neural net.

The tests in the **Tests** folder helped make sure the logic was correct, but may already be somewhat outdated.

The most likely way for all of this to change is that in editing the "play" file so that it can be used for the Hardcode and Neural Net files, it may stop being the playable game, and just the "brains".

There may also be two versions of the "Hardcode" file, one which plays games against the computer iteratively (as a proof of concept), and another that can be used by a player who inputs gameplay information and gets told how to play.

Finally, at this stage I do not integrate card counting strategies, and the rules of the game are not customizable. These are the final things that would be worth changing.

### Sources

Source for charts
<https://www.blackjackapprenticeship.com/blackjack-strategy-charts/>
Source for blackjack rules
<https://bicyclecards.com/how-to-play/blackjack>
