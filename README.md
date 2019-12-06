# alea

alea provides a way to create random variables that model real-world, probablistic 
experiments and then perform operations on those random variables. You can construct
new random variables, calculate a random variable's mean, variance, and moments, and
determine covariances between random variables, among other actions. See below
for a full features list.

## Why Model Random Variables?

The concept of a random variable is central to the field of probability theory; 
informally, a random variable is a mapping from the possible outcomes of a 
probabilistic phenomenon to real numbers. Random variables are important because 
they allow arbitrary values to be assigned to the results of an
uncertain action, and the interpretation of these values can influence whether
or not that action is considered desirable. Observe the following scenario.

> Say you are a hedge fund or institutional investor making bets on whether a
> stock will rise or not. You determine that Stock A will rise to a certain 
> level with 40% probability, stay the same with 30% probability, and fall with
> 30% probability. Stock B follows the same distribution and will rise and fall
> independently of Stock A. You're offered separate but identical options contracts 
> for A and B, where if the stock rises, you make $1000, but if not, you pay $150,
> the premium on the contract. If you were to buy these contracts repeatedly on
> different days, on average, how much money would you expect to make?

With alea, you could model this scenario pretty easily:

```
from alea.discrete import RootDiscreteRandVar

A = RootDiscreteRandVar({1000, -150}, lambda x : 0.4 if x == 1000 else 0.6)
B = RootDiscreteRandVar({1000, -150}, lambda x : 0.4 if x == 1000 else 0.6)
earnings = A + B
print(earnings.mean())
```

What if the scenario changes?

> Now, the hedge fund is levered. If a fund is levered, this means that 
> it is borrowing money from a bank or another fund at a certain ratio. For
> example, if a fund is levered 6:1, it is borrowing $6 for every $1 it owns.
> Leverage allows a fund to potentially multiply gains or losses.
> Say that your hedge fund on any day is levered either at a 4:1, 5:1, or 6:1
> ratio independently of any positions you are taking. What's your earnings 
> average now?

```
from alea.discrete import UniformDiscreteRandVar

L = UniformDiscreteRandVar({4, 5, 6})
levered_earnings = (L + 1) * earnings
print(levered_earnings.mean())
```

In this sense, alea is very powerful tool because it simplifies calculations
of important concepts tied to random variables. You can construct random
variables using simple, natural queries without having to worry about the
mathematics behind them.

## Features List

TODO

## Usage

alea will be made available on PyPI sometime in the future when
it is more mature. API documentation will also be included.

## Testing

To test alea locally, ensure that the following tools are installed:

* Python 3.7+
* pipenv
* pytest

After cloning this repository, run the following commands:

```
cd alea
pipenv install
pipenv run pytest
```

This will run all unit tests in the tests directory.
