import os

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import jax
import jax.numpy as jnp

HEIGHT = 50
WIDTH = 50
P = 0.25
ASSET_DIR = 'assets'

_at = lambda arr, i,j : jax.lax.dynamic_slice(arr, (i,j), (1,1))[0][0]

def n_live_neighbors(i, j, grid):
    height, width = grid.shape
    return _at(grid, i-1, j-1)\
                  +  _at(grid, i, j-1)\
                  +  _at(grid, (i+1)%height, j-1)\
                  +  _at(grid, i-1 , j)\
                  +  _at(grid, (i+1)%height, j)\
                  +  _at(grid, i-1, (j+1)%width)\
                  +  _at(grid, i, (j+1)%width)\
                  +  _at(grid, (i+1)%height, (j+1)%width)


def conway_ij(i, j, grid):
    live_neighbors = n_live_neighbors(i, j, grid)
    return jax.lax.cond((_at(grid, i, j) == 1.),
                        lambda x: jax.lax.cond(((live_neighbors<2.) | (live_neighbors>3.)),
                                               lambda y: jnp.array(0.), 
                                               lambda y: y,
                                               x),  
                        lambda x: jax.lax.cond((live_neighbors==3),
                                               lambda y: jnp.array(1.), 
                                               lambda y: y,
                                               x),
                        _at(grid, i, j))


_conway = jax.vmap(jax.vmap(conway_ij, in_axes=(None, 0, None)), 
                   in_axes=(0, None, None))


def conway(grid):
    height, width = grid.shape
    rows = jnp.arange(height, dtype=jnp.int32)
    cols = jnp.arange(width, dtype=jnp.int32)
    return _conway(rows, cols, grid)


def simulate(grid0, generations):
    def scan(grid, _):
        return conway(grid), grid
    return jax.lax.scan(scan, grid0, None, generations)[1]


def animate_life(grids, title='game of life'):
    fig, ax = plt.subplots()
    ax.set_title(title)

    def update(frame):
        ax.clear()
        ax.imshow(grids[frame], cmap = 'binary')
        ax.set_title(f'Generation {frame}')

    anim = FuncAnimation(fig, update, frames=len(grids), interval=250)
    anim.save(os.path.join(ASSET_DIR, title+'.gif'), writer='pillow')


if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    generations = 100
    key, grid_key = jax.random.split(key)
    grid0 = jax.random.choice(grid_key, jnp.array([1.,0.],dtype=jnp.float32), 
                              shape=(HEIGHT,WIDTH), p=jnp.array((P, 1-P)))
    grids = simulate(grid0, generations)
    animate_life(grids, 'Game of Life')