# Advanced Artificial Intelligence 
CS6955: Advanced Artificial Intelligence, University of Utah, Spring 2026.  

## Installation
```bash
...
```

## Structure
```bash
.
├── algorithm   # algorithm implementations
│   └── config      # configuration files for algorithms
├── docs        # reports
├── environment # environment setup
├── model       # model implementations (mlp, cnn, etc.)
├── script      # scripts for running tasks
└── util        # utility and helper functions
```

## Algorithms
| # | Abbr | Algorithm |
|---|------|--------------------------------------|
| 1 | bc   | [Behavioral Cloning](https://arxiv.org/abs/1604.06778) |
| 2 | bco  | [Behavioral Cloning from Observation](https://arxiv.org/abs/1805.01954) |
| 3 | q    | [Q-Learning](https://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf) |
| 4 | dqn  | [Deep Q-Network](https://www.nature.com/articles/nature14236) |
| 5 | vpg  | [Vanilla Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/vpg.html) |
| 6 | ppo  | [PPO](https://arxiv.org/abs/1707.06347) |

## Usage
```python
# Run algorithms on Gym environments
python run.py
# Manually play Gym environments
python play.py
```
## References
#### 
- [Class Website](https://dsbrown1331.github.io/advanced-ai-26/)
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/ebook/)
- [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/index.html)
#### Homeworks
1. [Imitation Learning](https://github.com/dsbrown1331/imitation_learning)
2. [Advanced BC and Interactive Imitation Learning](https://docs.google.com/document/d/1lhKB93JxxqcIZnbi9tuVh9KRNobdrsjNuK7l0NhhmnA/edit?tab=t.0)
3. [Multi-Armed Bandits](https://docs.google.com/document/d/1HsvielgBPUZA-MiO3Y2m4pEE5iXndgOi7TWyEZyvPpE/edit?tab=t.0)
4. [Q-Learning and DQN](https://github.com/dsbrown1331/q-learning-homework)
5. [Policy Gradient](https://github.com/dsbrown1331/policy_gradient_homework/)
