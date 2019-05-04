#! -*- coding: utf-8 -*-
# https://kexue.fm/archives/6583

# Vanilla GAN

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


T = 100
h = 0.1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def F(t, x):
    varphi1, varphi2, theta1, theta2 = x
    p = sigmoid(varphi1 * theta1 + varphi2 * theta2)
    return [-p * theta1, -p * theta2, (1 - p) * varphi1, (1 - p) * varphi2]


ts = np.arange(0, T, h)
xs = solve_ivp(F, (0, T), [0.1, 0.2, 0.2, 0.1], method='RK45', t_eval=ts)
varphi1, varphi2, theta1, theta2 = xs.y


plt.figure(figsize=(6, 6))
plt.clf()
plt.plot(theta1, theta2)
plt.scatter(0, 0, 30, color='red')
plt.xlabel('$\\theta_1$')
plt.ylabel('$\\theta_2$')
plt.savefig('test.pdf')


# Vanilla GAN 动画

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import animation


T = 100
h = 0.1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def F(t, x):
    varphi1, varphi2, theta1, theta2 = x
    p = sigmoid(varphi1 * theta1 + varphi2 * theta2)
    return [-p * theta1, -p * theta2, (1 - p) * varphi1, (1 - p) * varphi2]


ts = np.arange(0, T, h)
xs = solve_ivp(F, (0, T), [0.1, 0.2, 0.2, 0.1], method='RK45', t_eval=ts)
varphi1, varphi2, theta1, theta2 = xs.y


fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(0, 0, 30, color='red')
ax.set_xlim(-0.3, 0.3)
ax.set_ylim(-0.3, 0.3)
ax.set_xlabel('$\\theta_1$')
ax.set_ylabel('$\\theta_2$')
line, = plt.plot(theta1[:2], theta2[:2], color='blue')

def animate(i):
    line.set_data(theta1[:i], theta2[:i])
    return line,

anim = animation.FuncAnimation(fig=fig,
                               func=animate,
                               frames=range(12, len(theta1), 20),
                               interval=100)

anim.save('line.gif', writer='imagemagick', dpi=80)





# WGAN

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


T = 10
h = 0.01


def F(t, x):
    varphi1, varphi2, theta1, theta2 = x
    norm = np.sqrt(varphi1**2 + varphi2**2)
    dot = varphi1 * theta1 + varphi2 * theta2
    return [- theta1 / norm + dot * varphi1 / norm**3,
            - theta2 / norm + dot * varphi2 / norm**3,
            varphi1 / norm,
            varphi2 / norm
            ]


ts = np.arange(0, T, h)
xs = solve_ivp(F, (0, T), [0.1, 0.2, 0.2, 0.1], method='RK45', t_eval=ts)
varphi1, varphi2, theta1, theta2 = xs.y


plt.figure(figsize=(6, 6))
plt.clf()
plt.plot(theta1, theta2)
plt.scatter(0, 0, 30, color='red')
plt.xlabel('$\\theta_1$')
plt.ylabel('$\\theta_2$')
plt.savefig('test.pdf')




# WGAN 动画

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import animation


T = 10
h = 0.01


def F(t, x):
    varphi1, varphi2, theta1, theta2 = x
    norm = np.sqrt(varphi1**2 + varphi2**2)
    dot = varphi1 * theta1 + varphi2 * theta2
    return [- theta1 / norm + dot * varphi1 / norm**3,
            - theta2 / norm + dot * varphi2 / norm**3,
            varphi1 / norm,
            varphi2 / norm
            ]


ts = np.arange(0, T, h)
xs = solve_ivp(F, (0, T), [0.1, 0.2, 0.2, 0.1], method='RK45', t_eval=ts)
varphi1, varphi2, theta1, theta2 = xs.y


fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(0, 0, 30, color='red')
ax.set_xlim(-0.4, 0.4)
ax.set_ylim(-0.4, 0.4)
ax.set_xlabel('$\\theta_1$')
ax.set_ylabel('$\\theta_2$')
line, = plt.plot(theta1[:2], theta2[:2], color='blue')

def animate(i):
    line.set_data(theta1[:i], theta2[:i])
    return line,

anim = animation.FuncAnimation(fig=fig,
                               func=animate,
                               frames=range(12, len(theta1), 20),
                               interval=100)

anim.save('line.gif', writer='imagemagick', dpi=80)






# WGAN-GP

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


T = 40
h = 0.1
c = 1
lamb = 0.1


def F(t, x):
    varphi1, varphi2, theta1, theta2 = x
    norm = np.sqrt(varphi1**2 + varphi2**2)
    dot = varphi1 * theta1 + varphi2 * theta2
    return [- theta1 - 2 * lamb * (1 - c / norm) * varphi1,
            - theta2 - 2 * lamb * (1 - c / norm) * varphi2,
            varphi1,
            varphi2
            ]


ts = np.arange(0, T, h)
xs = solve_ivp(F, (0, T), [0.1, 0.2, 0.2, 0.1], method='RK45', t_eval=ts)
varphi1, varphi2, theta1, theta2 = xs.y


plt.figure(figsize=(6, 6))
plt.clf()
plt.plot(theta1, theta2)
plt.scatter(0, 0, 30, color='red')
plt.xlabel('$\\theta_1$')
plt.ylabel('$\\theta_2$')
plt.savefig('test.pdf')




# WGAN-GP 动画 (c=0)

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import animation


T = 40
h = 0.1
c = 0
lamb = 0.1


def F(t, x):
    varphi1, varphi2, theta1, theta2 = x
    norm = np.sqrt(varphi1**2 + varphi2**2)
    dot = varphi1 * theta1 + varphi2 * theta2
    return [- theta1 - 2 * lamb * (1 - c / norm) * varphi1,
            - theta2 - 2 * lamb * (1 - c / norm) * varphi2,
            varphi1,
            varphi2
            ]


ts = np.arange(0, T, h)
xs = solve_ivp(F, (0, T), [0.1, 0.2, 0.2, 0.1], method='RK45', t_eval=ts)
varphi1, varphi2, theta1, theta2 = xs.y


fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(0, 0, 30, color='red')
ax.set_xlim(-0.3, 0.3)
ax.set_ylim(-0.3, 0.3)
ax.set_xlabel('$\\theta_1$')
ax.set_ylabel('$\\theta_2$')
line, = plt.plot(theta1[:2], theta2[:2], color='blue')

def animate(i):
    line.set_data(theta1[:i], theta2[:i])
    return line,

anim = animation.FuncAnimation(fig=fig,
                               func=animate,
                               frames=range(12, len(theta1), 8),
                               interval=100)

anim.save('line.gif', writer='imagemagick', dpi=80)



# WGAN-GP 动画 (c=1)


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import animation


T = 40
h = 0.1
c = 1
lamb = 0.1


def F(t, x):
    varphi1, varphi2, theta1, theta2 = x
    norm = np.sqrt(varphi1**2 + varphi2**2)
    dot = varphi1 * theta1 + varphi2 * theta2
    return [- theta1 - 2 * lamb * (1 - c / norm) * varphi1,
            - theta2 - 2 * lamb * (1 - c / norm) * varphi2,
            varphi1,
            varphi2
            ]


ts = np.arange(0, T, h)
xs = solve_ivp(F, (0, T), [0.1, 0.2, 0.2, 0.1], method='RK45', t_eval=ts)
varphi1, varphi2, theta1, theta2 = xs.y


fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(0, 0, 30, color='red')
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_xlabel('$\\theta_1$')
ax.set_ylabel('$\\theta_2$')
line, = plt.plot(theta1[:2], theta2[:2], color='blue')

def animate(i):
    line.set_data(theta1[:i], theta2[:i])
    return line,

anim = animation.FuncAnimation(fig=fig,
                               func=animate,
                               frames=range(12, len(theta1), 8),
                               interval=100)

anim.save('line.gif', writer='imagemagick', dpi=80)






# GAN-QP

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


T = 40
h = 0.1
lamb = 1


def F(t, x):
    varphi1, varphi2, theta1, theta2 = x
    norm = np.sqrt(varphi1**2 + varphi2**2)
    dot = varphi1 * theta1 + varphi2 * theta2
    return [- theta1 - theta1 * dot / norm / lamb,
            - theta2 - theta2 * dot / norm / lamb,
            varphi1,
            varphi2
            ]


ts = np.arange(0, T, h)
xs = solve_ivp(F, (0, T), [0.1, 0.2, 0.2, 0.1], method='RK45', t_eval=ts)
varphi1, varphi2, theta1, theta2 = xs.y


plt.figure(figsize=(6, 6))
plt.clf()
plt.plot(theta1, theta2)
plt.scatter(0, 0, 30, color='red')
plt.xlabel('$\\theta_1$')
plt.ylabel('$\\theta_2$')
plt.savefig('test.pdf')




# GAN-QP 动画

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import animation


T = 40
h = 0.1
lamb = 1


def F(t, x):
    varphi1, varphi2, theta1, theta2 = x
    norm = np.sqrt(varphi1**2 + varphi2**2)
    dot = varphi1 * theta1 + varphi2 * theta2
    return [- theta1 - theta1 * dot / norm / lamb,
            - theta2 - theta2 * dot / norm / lamb,
            varphi1,
            varphi2
            ]


ts = np.arange(0, T, h)
xs = solve_ivp(F, (0, T), [0.1, 0.2, 0.2, 0.1], method='RK45', t_eval=ts)
varphi1, varphi2, theta1, theta2 = xs.y


fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(0, 0, 30, color='red')
ax.set_xlim(-0.3, 0.3)
ax.set_ylim(-0.3, 0.3)
ax.set_xlabel('$\\theta_1$')
ax.set_ylabel('$\\theta_2$')
line, = plt.plot(theta1[:2], theta2[:2], color='blue')

def animate(i):
    line.set_data(theta1[:i], theta2[:i])
    return line,

anim = animation.FuncAnimation(fig=fig,
                               func=animate,
                               frames=range(12, len(theta1), 8),
                               interval=100)

anim.save('line.gif', writer='imagemagick', dpi=80)



# GAN-QP带L2正则 动画

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import animation


T = 70
h = 0.1
lamb = 1


def F(t, x):
    varphi1, varphi2, theta1, theta2 = x
    norm = np.sqrt(varphi1**2 + varphi2**2)
    dot = varphi1 * theta1 + varphi2 * theta2
    return [- theta1 - theta1 * dot / norm / lamb - 0.1 * varphi1,
            - theta2 - theta2 * dot / norm / lamb - 0.1 * varphi2,
            varphi1,
            varphi2
            ]


ts = np.arange(0, T, h)
xs = solve_ivp(F, (0, T), [0.1, 0.2, 0.2, 0.1], method='RK45', t_eval=ts)
varphi1, varphi2, theta1, theta2 = xs.y


fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(0, 0, 30, color='red')
ax.set_xlim(-0.3, 0.3)
ax.set_ylim(-0.3, 0.3)
ax.set_xlabel('$\\theta_1$')
ax.set_ylabel('$\\theta_2$')
line, = plt.plot(theta1[:2], theta2[:2], color='blue')

def animate(i):
    line.set_data(theta1[:i], theta2[:i])
    return line,

anim = animation.FuncAnimation(fig=fig,
                               func=animate,
                               frames=range(12, len(theta1), 9),
                               interval=100)

anim.save('line.gif', writer='imagemagick', dpi=80)
