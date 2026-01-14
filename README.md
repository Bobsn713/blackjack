# Blackjack Engine
```
___________________________________ __    _____________________________ _______
___  __ )__  /___    |_  ____/__  //_/    ______  /__    |_  ____/__  //_/__  /
__  __  |_  / __  /| |  /    __  ,<       ___ _  /__  /| |  /    __  ,<  __  / 
_  /_/ /_  /___  ___ / /___  _  /| |      / /_/ / _  ___ / /___  _  /| |  /_/  
/_____/ /_____/_/  |_\____/  /_/ |_|      \____/  /_/  |_\____/  /_/ |_| (_)  
```
This codebase is a command-line blackjack game-engine/application where you can design your own neural network and train it to learn basic blackjack strategy. 

## Blackjack Engine Overview
There are 4 ways to use this Blackjack Engine without touching any code. 
1. It can be used to play a text-based blackjack game, with the computer serving as dealer and scorekeeper. 
2. It can advise you on how to play any given hand you're playing in real life based on basic-strategy. 
3. It can simulate an arbitrary number of hands, using either a basic strategy model or a neural network.
4. It allows users to choose hyperparameters and train their own neural network, which they can then use in the simulator. 

I developed this engine as an excuse to learn to build neural networks, and built up the other functionality along the way. 

## Install and Use
The easiest way to use the blackjack engine is to download the code, navigate to the directory you just downloaded, and type the following command in your terminal (with [python](https://www.python.org/downloads/) and [uv](https://docs.astral.sh/uv/getting-started/installation/) installed). 

```bash
uv run main.py
```
For those unsure where to start when training a neural network, some relatively reasonable default parameters are: 
```
Layers: 8
Neurons per Layer: 128
Learning Rate: .02
Epochs: 10
Batch Size: 64
```

## Organization
For those interested in digging deeper into the code, here's the basic outline: 

* `main.py` is the user-friendly entry point, which coordinates between the different functions of the engine from the command line. 
* `game_logic.py` is where the underlying logic of the game is defined. To make different game-modes (text-based play along, getting recommended moves, simulating thousands of hands) that use the same game logic, turn-structure, etc., the structure is built around the use of dependency injection, and calls different functions based on which functions are passed as arguments in the `GameState` and `GameInterface` objects. 
* Those objects are defined in `base.py`, and a few common configurations are stored in `config.py`. 
* Most of the functions for each game-mode, which are either passed into the main functions of `game_logic.py` or called by `main.py`, are defined in their own file. That's what `text.py`, `basic_strategy.py`, `cheat.py`, `train_nn.py`, and `performance_tracker.py` do. 

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
Ideas to build on this program further include, in rough order of priority: 
* Allow users to use trained models in "cheat" mode. 
* Calculate odds in different positions using the engine instead of relying on established basic-strategy rules. 
* Go beyond basic strategy to include card-counting capabilities. 
* Make the ruleset customizable. 
* Allow newer machines to use newer versions of `pytorch`/`numpy` and use their GPUs. 
* Allow users to see the hyperparameters of their various models. 
* Allow users to customize the activation functions used in their models. 
* Allow the user to simulate different betting strategies. 
* Switch from the current "imitation" model of training the neural network to a true RL framework.
* Build a database of hyperparameter combinations to map out the hyperparameter "loss landscape". 
* Make a web-based version for less technical people. 
* Build a GUI, either terminal based or not?

## Sources
Source for charts:
<https://www.blackjackapprenticeship.com/blackjack-strategy-charts/>

Source for blackjack rules:
<https://bicyclecards.com/how-to-play/blackjack>

## License
Distributed under the [MIT](https://github.com/Bobsn713/blackjack/blob/main/LICENSE) License. 