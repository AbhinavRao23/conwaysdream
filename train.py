import os

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from conway import *
from deeponet import *

ASSET_DIR = 'assets'
HEIGHT = 50
WIDTH = 50
P = 0.25
key = jax.random.PRNGKey(0)

latent_size = 64
key, t_key, b_key = jax.random.split(key, 3)
trunk = eqx.nn.MLP(2, latent_size, 128, 3, key=t_key)
branch = eqx.nn.MLP(HEIGHT*WIDTH, latent_size, 128, 3, key=b_key)
deeponet = {'t':trunk,'b':branch}

lr = 0.00001
n = 100 #per curriculum
n_grids = 32
n_cells = 16
max_gen = 10
n_curriculum = 5
update_max_gen_by = 10
reset_optim = True

optim = optax.adam(lr)
opt_state = optim.init(eqx.filter(deeponet, eqx.is_array))
losses = []

for curriculum in range(n_curriculum):
    for i in range(n):
        key, train_key = jax.random.split(key)
        deeponet, loss, opt_state = train_step(
            deeponet, train_key, n_grids, n_cells, max_gen, optim, opt_state)
        losses.append(loss)
    
    if reset_optim:
        optim = optax.adam(lr)
        opt_state = optim.init(eqx.filter(deeponet, eqx.is_array))

    max_gen += update_max_gen_by

fig, ax = plt.subplots()
ax.plot(losses)
ax.set(title = 'training losses v iters', xlabel = 'iter', ylabel = 'losss')
fig.savefig(os.path.join(ASSET_DIR, 'traininglosses'))

generations = 100
key, grid_key = jax.random.split(key)
grid0 = jax.random.choice(grid_key, jnp.array([1.,0.],dtype=jnp.float32), 
                            shape=(HEIGHT, WIDTH), p=jnp.array((P, 1-P)))

deep_life = simulate_deeponet(deeponet, 
                              grid0, generations)
animate_life(deep_life, 'Dream of Life')