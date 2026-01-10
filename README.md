# Blackjack Engine
```
___________________________________ __    _____________________________ _______
___  __ )__  /___    |_  ____/__  //_/    ______  /__    |_  ____/__  //_/__  /
__  __  |_  / __  /| |  /    __  ,<       ___ _  /__  /| |  /    __  ,<  __  / 
_  /_/ /_  /___  ___ / /___  _  /| |      / /_/ / _  ___ / /___  _  /| |  /_/  
/_____/ /_____/_/  |_\____/  /_/ |_|      \____/  /_/  |_\____/  /_/ |_| (_)  
```
This codebase is a blackjack game engine/application where you can build your own neural network and train it to learn basic blackjack strategy. 

## Blackjack Engine Overview
There are 4 ways to use this Blackjack Engine without touching any code. 
1. It can be used to play a text-based blackjack game, with the computer serving as dealer and scorekeeper. 
2. It can advise you on how to play any given hand based on basic-strategy. 
3. It can simulate an arbitrary number of hands, using either a basic strategy model or a neural network.
4. It allows users to choose hyperparameters and train their own neural network, which they can then use in the simulator. 

I developed this engine as an excuse to learn to build neural networks, and built up the other functionality along the way. 

## Install
The easiest way to use the blackjack engine is to type the following command in your terminal, with [python](https://www.python.org/downloads/) and [uv](https://docs.astral.sh/uv/getting-started/installation/) installed. 

```bash
uv run main.py
```

(If you have an older computer you may need older versions of numpy and pytorch, but `uv` should handle this automatically for many users. There is also a `requirements.txt` file for users more comfortable with other methods of dependency management.)

## Rules
It is not yet possible to customize the ruleset. The current rules are:

* doubling down is allowed,
* splitting is allowed,
* maximum 3 splits (4 hands),
* doubling down after splitting is allowed,
* if aces are split, only one card is dealt to each hand
* no insurance or surrender,
* dealer hits on a soft 17 (H17),

## Further Development
Ideas to build on this program further include: 
* Allow the user to simulate different betting strategies. 
* Go beyond basic strategy to include card-counting capabilities. 
* Make the ruleset customizable. 
* Switch from the current "imitation" model of training the neural network to a true RL framework.
* Build a GUI. 

## Sources
Source for charts:
<https://www.blackjackapprenticeship.com/blackjack-strategy-charts/>

Source for blackjack rules:
<https://bicyclecards.com/how-to-play/blackjack>

### License