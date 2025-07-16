# README

## Goal

The goal of this project is first of all to "hardcode" a program that can tell me what to do in blackjack, and second of all to train a neural net to learn on its own.

## Structure
As this has evolved, the first task has been to create a text based, playable Blackjack game to ensure that the game logic functions as it should. That is more or less completed now as Play_v5.py.

Hardcode.py will be where the brains are that can play each hand perfectly.

After that, I'll make a third file to develop the neural net.

The tests in the Tests folder helped make sure the logic was correct, but may already be somewhat outdated.

The most likely way for all of this to change is that in editing the "play" file so that it can be used for the Hardcode and Neural Net files, it may stop being the playable game, and just the "brains".

There may also be two versions of the "Hardcode" file, one which plays games against the computer iteratively (as a proof of concept), and another that can be used by a player who inputs gameplay information and gets told how to play.

Finally, at this stage I do not integrate card counting strategies, and the rules of the game are not customizable. These are the final things that would be worth changing.