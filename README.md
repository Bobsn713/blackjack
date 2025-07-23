# README

## Goal

The goal of this project is first of all to "hardcode" a program that can tell me what to do in blackjack, and second of all to train a neural net to learn to play well on its own.

## Structure

**Logic.py** is the brains of the game, where the fundamental gameplay mechanics are defined and stored.

**Text.py** will be where functions for the text based terminal game are stored.

**Hardcode.py** will be where the brains are that can play each hand perfectly.

**Neural Net.py** will be, predictably, where I train the neural net.

**Performance_Tracker.py** Collects and displays game statistics. 

### Temporary Files

---
**Scrapwork.py** is a place to work out an alternative way to store data, which I may use but I am still not sure.

### Notes

---
Eventually there may also be an **Advice.py** file that can be used by a player who inputs gameplay information and gets told how to play.

At this stage I do not integrate card counting strategies, and the rules of the game are not customizable. These will be worth implementing eventually.

## Rules

For now the rules are:

* doubling down is allowed,
* splitting is allowed,
* maximum 3 splits (4 hands),
* doubling down after splitting is allowed,
* if aces are split, only one card is dealt to each hand
* no insurance or surrender,
* dealer hits on a soft 17 (H17),

## Sources

Source for charts
<https://www.blackjackapprenticeship.com/blackjack-strategy-charts/>

Source for blackjack rules
<https://bicyclecards.com/how-to-play/blackjack>
