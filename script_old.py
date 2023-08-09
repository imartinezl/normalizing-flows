# %%

# %%
import matplotlib.pyplot as plt
import sklearn.datasets as datasets

import torch
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal, ConditionalDiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.nn.nets import ResidualNet

# %%
x, y = datasets.make_moons(500, noise=.1)
plt.scatter(x[:, 0], x[:, 1], c=y)
# %%

num_layers = 5
base_dist = StandardNormal(shape=[2])

transforms = []
for _ in range(num_layers):
    transforms.append(ReversePermutation(features=2))
    transforms.append(MaskedAffineAutoregressiveTransform(features=2, 
                                                          hidden_features=4))
transform = CompositeTransform(transforms)

flow = Flow(transform, base_dist)
optimizer = torch.optim.Adam(flow.parameters())

# %%

num_iter = 5000

for i in range(num_iter):
    x, y = datasets.make_moons(128, noise=.1)
    x = torch.tensor(x, dtype=torch.float32)
    optimizer.zero_grad()
    loss = -flow.log_prob(inputs=x).mean()
    loss.backward()
    optimizer.step()
    
    if (i + 1) % 500 == 0:
        xline = torch.linspace(-1.5, 2.5, 20)
        yline = torch.linspace(-.75, 1.25, 20)
        xgrid, ygrid = torch.meshgrid(xline, yline)
        xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

        with torch.no_grad():
            zgrid = flow.log_prob(xyinput).exp().reshape(100, 100)

        plt.contourf(xgrid.numpy(), ygrid.numpy(), zgrid.numpy())
        plt.title('iteration {}'.format(i + 1))
        plt.show()
# %%

num_layers = 5
base_dist = ConditionalDiagonalNormal(shape=[2], 
                                      context_encoder=torch.nn.Linear(1, 4))

transforms = []
for _ in range(num_layers):
    transforms.append(ReversePermutation(features=2))
    transforms.append(MaskedAffineAutoregressiveTransform(features=2, 
                                                          hidden_features=4, 
                                                          context_features=1))
transform = CompositeTransform(transforms)

flow = Flow(transform, base_dist)
optimizer = torch.optim.Adam(flow.parameters())
# %%


num_iter = 5000
for i in range(num_iter):
    x, y = datasets.make_moons(128, noise=.1)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    optimizer.zero_grad()
    loss = -flow.log_prob(inputs=x, context=y).mean()
    loss.backward()
    optimizer.step()
    
    if (i + 1) % 500 == 0:
        fig, ax = plt.subplots(1, 2)
        xline = torch.linspace(-1.5, 2.5)
        yline = torch.linspace(-.75, 1.25)
        xgrid, ygrid = torch.meshgrid(xline, yline)
        xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

        with torch.no_grad():
            zgrid0 = flow.log_prob(xyinput, torch.zeros(10000, 1)).exp().reshape(100, 100)
            zgrid1 = flow.log_prob(xyinput, torch.ones(10000, 1)).exp().reshape(100, 100)

        ax[0].contourf(xgrid.numpy(), ygrid.numpy(), zgrid0.numpy())
        ax[1].contourf(xgrid.numpy(), ygrid.numpy(), zgrid1.numpy())
        plt.title('iteration {}'.format(i + 1))
        plt.show()

# %%

import jax
import jax.numpy as jnp
from jax import grad
from jax import lax
import matplotlib.pyplot as plt


def get_cell(x):
    c = jnp.floor(x*nc)
    c = jnp.clip(c, 0, nc-1).astype(jnp.int32)
    return c

def get_velocity(x):
    c = get_cell(x)
    ac = a[c]
    bc = b[c]
    return ac*x + bc

def left_boundary(c):
    return c / nc

def right_boundary(c):
    return (c+1)/nc

def get_psi(x, t):
    c = get_cell(x)
    ac = a[c]
    bc = b[c]
    eta = jnp.exp(t*ac)
    return eta * x + (bc / ac) * (eta - 1.0)

def get_hit_time(x):
    c = get_cell(x)
    v = get_velocity(x)
    xc = lax.cond(v >= 0, right_boundary, left_boundary, c)

    ac = a[c]
    bc = b[c]

    return jnp.log((xc + bc / ac) / (x + bc / ac)) / ac

def get_time(x, t):
    return t - get_hit_time(x)

def get_x(psi, left, right):
    return jnp.clip(psi, left, right)

def f(x, t):
    c = get_cell(x)
    cont = 0
    while True:
        left = left_boundary(c)
        right = right_boundary(c)
        v = get_velocity(x)
        psi = get_psi(x, t)

        g = grad(get_psi, 0)(x,t)
        print("dpsi_dx", g)

        g = grad(get_psi, 1)(x,t)
        print("dpsi_dt", g)

        cond1 = jnp.logical_and(left <= psi, psi <= right)
        cond2 = jnp.logical_and(v >= 0, c == nc - 1)
        cond3 = jnp.logical_and(v <= 0, c == 0)
        valid = jnp.any(jnp.array([cond1, cond2, cond3]))

        # if valid:
        #     print("DONE")
        #     return psi

        g = grad(get_time, 0)(x,t)
        print("dt_dx", g)

        g = grad(get_time, 1)(x,t)
        print("dt_dt", g)
        # t -= get_hit_time(x)
        # t = tm
        t = get_time(x, t)
        # x = jnp.clip(psi, left, right)
        x = get_x(psi, left, right)
        g = grad(get_x, 2)(psi, left, right)
        print("dx_dpsi", g)
        c = lax.cond(v >= 0, lambda c: c+1, lambda c: c-1, c) 

        # REMOVE>>>>>>>>>>>>>>>>>>>>>>>>
        left = left_boundary(c)
        right = right_boundary(c)
        v = get_velocity(x)
        psi = get_psi(x, t)

        g = grad(get_psi, 0)(x,t)
        print("dpsi_dx", g)

        g = grad(get_psi, 1)(x,t)
        print("dpsi_dt", g)

        return psi
        # REMOVE<<<<<<<<<<<<<<<<<<<<<<<<<

        cont += 1
        # if cont > nc:
        #     raise BaseException

    return None

def f2(x, t):
    c = get_cell(x)
    init_val = jnp.array([1.0, x, t, c, x])

    def cond_fun(val):
        return val[0] == 1.0

    def body_fun(val):
        x = val[1]
        t = val[2]
        c = val[3]

        left = left_boundary(c)
        right = right_boundary(c)
        v = get_velocity(x)
        psi = get_psi(x, t)

        cond1 = jnp.logical_and(left <= psi, psi <= right)
        cond2 = jnp.logical_and(v >= 0, c == nc - 1)
        cond3 = jnp.logical_and(v <= 0, c == 0)
        valid = jnp.any(jnp.array([cond1, cond2, cond3]))

        t -= get_hit_time(x)
        x = jnp.clip(psi, left, right)
        c = lax.cond(v >= 0, lambda c: c+1, lambda c: c-1, c) 

        val = jnp.array([~valid, x, t, c, psi])

        return val

    val = lax.while_loop(cond_fun, body_fun, init_val)
    return val[4]

k = 0.57735027

def fdebug(x, t):
    # c = get_cell(x)
    c = 0

    # left = left_boundary(c)
    left = 0
    # right = right_boundary(c)
    right = 0.5

    # v = get_velocity(x)
    # v = k*x
    # psi = get_psi(x, t)
    eta = jnp.exp(t*k)
    psi = eta * x

    return psi

    # t = get_time(x, t)
    xc = 0.5
    t = t - jnp.log(xc / x) / k
    # x = get_x(psi, left, right)
    x = right
    # c = lax.cond(v >= 0, lambda c: c+1, lambda c: c-1, c)
    c = 1

    # return psi
    
    # REMOVE>>>>>>>>>>>>>>>>>>>>>>>>
    # left = left_boundary(c)
    left = 0.5
    # right = right_boundary(c)
    right = 1.0
    # psi2 = get_psi(x, t)
    eta = jnp.exp(t*(-k))
    psi2 = eta * x + (k / (-k)) * (eta - 1.0)


    return psi2
    # REMOVE<<<<<<<<<<<<<<<<<<<<<<<<<

    return None



import graphviz
def hlo_graph(f, *args, **kwargs):
    comp = jax.xla_computation(f)(*args, **kwargs)
    graph = graphviz.Source(comp.as_hlo_dot_graph())
    return graph

# hlo_graph(grad(fdebug), x, t)

nc = 2
x = 0.2
# x = jnp.linspace(0,1)
a = jnp.array([0.57735027,-0.57735027])
b = jnp.array([0.0, 0.57735027])
t = 1.0

xm = 0.5
tm = 0.6135036187893925
# x, f(x, t).item()
# grad(f)(x, t).item()
# grad(get_hit_time)(x).item(), grad(get_psi)(xm,tm).item()

# jax.make_jaxpr(f)(x,t)
jax.make_jaxpr(grad(fdebug))(x,t)
grad(fdebug)(x,t)
# %%
y, g = [], []
grid = jnp.linspace(0, 1.0, 100, endpoint=False)
for x in grid:
    y.append(f(x,t))
    g.append(grad(f)(x,t))
y = np.array(y)
g = np.array(g)

# %%
plt.figure()
plt.plot(grid,y)
plt.axline((0,0),(1,1), c='k', ls='--')

plt.figure()
plt.plot(grid,g)
ge = np.gradient(y, grid)
plt.plot(grid,ge)