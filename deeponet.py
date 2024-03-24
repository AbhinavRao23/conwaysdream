from functools import partial
import os

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from conway import *

ASSET_DIR = 'assets'
HEIGHT = 50
WIDTH = 50
P = 0.25

_at = lambda arr, i,j : jax.lax.dynamic_slice(arr, (i,j), (1,1))[0][0]

def evolve_grids(grid0s, till_generations, max_gen):
    def last_gen(grid0, till_generation, max_gen): 
        return simulate(grid0, max_gen)[till_generation] #why does this work
    return jax.vmap(last_gen, in_axes=(0,0,None))(grid0s, till_generations, max_gen)


# def evaluate_deeponet_ij(model, i, j, grid):
#     x, y = i/HEIGHT, j/WIDTH
#     latent_b = model['b'](grid.flatten())
#     latent_t = model['t'](jnp.array((x,y)))
#     return jax.nn.sigmoid(jnp.dot(latent_b, latent_t))


def evaluate_deeponet_ij(model, i, j, grid):
    val = _at(grid, i, j)
    neighbors = n_live_neighbors(i, j, grid)
    latent_b = model['b'](grid.flatten())
    latent_t = model['t'](jnp.array((val, neighbors)))
    return jax.nn.sigmoid(jnp.dot(latent_b, latent_t))


@partial(jax.vmap, in_axes = (None, 0, 0, 0))
@partial(jax.vmap, in_axes = (None, 0, 0, None))
def bceloss(model, i, j, grid):
    p = evaluate_deeponet_ij(model, i, j, grid)
    y = conway_ij(i, j, grid)
    return -(y*jnp.log(p) + (1-y)*jnp.log(1-p))


def loss_func(model, rows, cols, grids):
    return jnp.mean(bceloss(model, rows, cols, grids))


@eqx.filter_jit
def train_step(model, key, n_grids, n_cells, max_gen, optim, opt_state):
    grid_key, row_key, col_key, gen_key = jax.random.split(key, 4)
    till_generations = jax.random.choice(gen_key, max_gen, (n_grids,))
    grids = jax.random.choice(grid_key, jnp.array([1.,0.],dtype=jnp.float32),
                              (n_grids, HEIGHT, WIDTH), p=jnp.array([(1-P), P]))
    grids = evolve_grids(grids, till_generations, max_gen)
    rows = jax.random.choice(row_key, HEIGHT, (n_grids, n_cells))
    cols = jax.random.choice(col_key, HEIGHT, (n_grids, n_cells))
    loss,grads = eqx.filter_value_and_grad(loss_func)(model, rows, cols, grids)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, loss, opt_state


_evaluate_deeponet = jax.vmap(
    jax.vmap(evaluate_deeponet_ij,in_axes=(None, None, 0, None)),
    in_axes=(None, 0, None, None)) 


def evaluate_deeponet(deeponet, grid):
    rows = jnp.arange(HEIGHT, dtype=jnp.int32)
    cols = jnp.arange(WIDTH, dtype=jnp.int32)
    return _evaluate_deeponet(deeponet, rows, cols, grid)


def simulate_deeponet(deeponet, grid0, generations):
    def scan(grid, _):
        return evaluate_deeponet(deeponet, grid), grid
    return jax.lax.scan(scan, grid0, None, generations)[1]


if __name__ == '__main__':
    ...