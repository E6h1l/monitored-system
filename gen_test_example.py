import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.animation as animation
import matplotlib
from scipy import signal
from sympy.abc import x, y, t
from sympy import *
import sympy as sp
import json

fps = 10 # frame per sec
frn = 50
k=1

a, b = 0, 1
c, d = 1, 2
T = 3

params = {}
params["a"] = a
params["b"] = b
params["c"] = c
params["d"] = d
params["T"] = T
params["L0"] = 1
params["R0"] = 3
params["Lg"] = 1
params["Rg"] = 3
params["k"] = 1

params["x0"] = np.array([(a+b)/2, a*0.1 + 0.9*b, a*0.7+b*0.3])
params["y0"] = np.array([(c+d)/2, c*0.1 + 0.9*d, c*0.7+d*0.3])

params["xg"] = np.array([a, b, a*0.7+b*0.3])
params["yg"] = np.array([(c+d)/2, c*0.1 + 0.9*d, c])
params["tg"] = np.array([0.1*T, 0.4*T, 0.7*T])

params["L0"] = ["f"]
params["Lg"] = ["f"]
params["f"] = "sin(3*x)*sin(4*y)*sin(5*t)"

#G = sp.Heaviside(t)/(4*k*sp.pi*t)*sp.exp(-(x**2+y**2)/4/t/k)
f = S(params["f"])

L = lambda f: eval("diff(f, t)-(diff(f, x, x)+diff(f, y, y))")
u = L(f)

params["u"] = str(u)

f = sp.lambdify((x, y, t), f, "numpy")

params["Y0"] = f(params["x0"], params["y0"], 0).reshape((1, 3))
params["Yg"] = f(params["xg"], params["yg"], params["tg"]).reshape((1, 3))

for key in ("x0", "y0", "xg", "yg", "tg", "Y0", "Yg"):
    params[key] = params[key].tolist()

with open("test_params.json", "w") as ff:
    json.dump(params, ff, indent=2)

num_points = 50
X = np.linspace(a, b, num_points)
Y = np.linspace(c, d, num_points)
T_ = np.linspace(0, T, num_points)


def update_plot(frame_number, zarray, plot):
    plot[0].remove()
    plot[0] = ax.plot_surface(x, y, zarray[:,:,frame_number], cmap="magma")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

phi = f(X[:, None, None], Y[None, :, None], T_[None, None, :])
x, y = np.meshgrid(X, Y)
plot = [ax.plot_surface(x, y, phi[:,:,0], rstride=1, cstride=1)]
ax.set_zlim(-1.1,1.1)
ani = animation.FuncAnimation(fig, update_plot, num_points, fargs=(phi, plot), interval=1000/fps)
ani.save("test_example.gif", fps=fps)
plt.show()
