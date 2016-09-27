# Train a Smartcab How to Drive

This project uses the pygame environment to navigate a car sprite across a grid network. The goal is for the car to reach the goal safely, before the time limit expires. The project implements q-learning to update q-values of the car's state in order to reach its goal quickly.

The state of the car is based on the current state of the traffic light and the direction it intends to go. Its reward, and q-value, increases if the car follows the law and moves toward the intended direction.

## Install

This project requires Python 2.7 with the pygame library installed:

https://www.pygame.org/wiki/GettingStarted

## Code

Open `smartcab/agent.py` and implement `LearningAgent`. Follow `TODO`s for further instructions.

## Run

Make sure you are in the top-level project directory `smartcab/` (that contains this README). Then run:

```python smartcab/agent.py```

OR:

```python -m smartcab.agent```
