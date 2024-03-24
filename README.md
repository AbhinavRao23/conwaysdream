# Deep-O-nets for learning Conway's Game transfer operator

Attempts to train a model that can learn the transfer operator in Conway's game of life, using JAX. 

## Contents:

1. Conway: Contains a toroidal simulator completely written in `jax` while being `jit` friendly.

2. DeepOnet: Contains a basic `MLP` based deepOnet to approximate the transfer operator. 

## Results:
Real:
![alt text](<assets/Game of Life.gif>)


Dream:
![alt text](<assets/Dream of Life.gif>)



## Todo

1. Try CNN-based trunk.
2. Try MLP-mixer trunk.

