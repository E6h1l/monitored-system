import sympy as sp
from sympy import *
from sympy.abc import x, y, t
import numpy as np
from scipy import signal
import json
import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

import warnings
warnings.filterwarnings("ignore")


class Solver:
    """
    Class for solving dynamic system modeling problem
    """
    def __init__(self, a, b, c, d, T, k=1, num_points=80):
        self.G_sp = Heaviside(t) / (4*k*pi*t) * exp(-(x**2+y**2)/(4*t*k))
        # self.G_np = lambdify((x, y, t), self.G_sp, [{'Heaviside': lambda x: np.heaviside(x, 1)}, 'numpy'])
        self.G_np = lambdify((x, y, t), self.G_sp, 'numpy')
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.T = T
        self.num_points = num_points

        self.hx = (b - a) / (num_points - 1)
        self.hy = (d - c) / (num_points - 1)
        self.ht = T / (num_points - 1)

        self.X = np.linspace(a, b, num_points)[:, None, None]
        self.Y = np.linspace(c, d, num_points)[None, :, None]
        self.T_ = np.linspace(0, T, num_points)[None, None, :]

        self.XX = np.linspace(a-b, b-a, 2*num_points-1)[:, None, None]
        self.YY = np.linspace(c-d, d-c, 2*num_points-1)[None, :, None]
        self.TT = np.linspace(-T, T, 2*num_points-1)[None, None, :]

        num_points = 80
        hx = (self.b - self.a) / (num_points - 1)
        hy = (self.d - self.c) / (num_points - 1)
        ht = self.T / (num_points - 1)

        X = np.linspace(self.a, self.b, num_points)[:, None, None]
        Y = np.linspace(self.c, self.d, num_points)[None, :, None]
        T = np.linspace(0, self.T, num_points)[None, None, :]

        self.integrate_params = [X, Y, T, hx, hy, ht]

    def convolve_green_inf(self, u):
        """
        Compute convolution of response function u and
        Green's function over region S.
        Return values of state function on net of S
        called S'

        :param u: np.function
        :return: np.ndarray of size (num_points, num_points, num_points)
        """
        self.update_pbar("Computing of phi_inf")
        
        u_S = u(self.X, self.Y, self.T_)*self.hx
        G_SS = self.G_np(self.XX, self.YY, self.TT)*self.hy
        G_SS[np.isnan(G_SS)] = 0

        phi = signal.convolve(G_SS, u_S, mode="valid")*self.ht

        return phi

    def convolve_green_0(self, u0):
        """
        Compute convolution of response function u0 and
        Green's function over region S0.
        Return values of state function on net of S
        called S'

        :param u0: np.function
        :return: np.ndarray of size (num_points, num_points, num_points)
        """
        self.update_pbar("Computing of phi_0")

        print()
        print("X", self.X.shape)
        print("Y", self.Y.shape)
        print("T_", (self.T_-self.T).shape)

        u_S0 = u0(self.X, self.Y, self.T_-self.T) * self.hx
        G_SS0 = self.G_np(self.XX, self.YY, self.TT+self.T) * self.hy
        G_SS0[np.isnan(G_SS0)] = 0

        print(u_S0)

        phi = signal.convolve(G_SS0, u_S0, mode="valid") * self.ht

        return phi

    def convolve_green_g(self, ug):
        """
        Compute convolution of response function ug and
        Green's function over region Sg.
        Return values of state function on net of S
        called S'

        :param ug: np.function
        :return: np.ndarray of size (num_points, num_points, num_points)
        """
        self.update_pbar("Computing of phi_g")
        phi = 0

        for dx in (self.a - self.b, 0, self.b - self.a):
            for dy in (self.c - self.d, 0, self.d - self.c):
                if dx == 0 and dy == 0:
                    continue
                self.update_pbar()
                u_Sg = ug(self.X+dx, self.Y+dy, self.T_) * self.hx
                G_SSg = self.G_np(self.XX-dx, self.YY-dy, self.TT) * self.hy
                G_SSg[np.isnan(G_SSg)] = 0

                phi += signal.convolve(G_SSg, u_Sg, mode="valid") * self.ht

        return phi

    def dx(self, f):
        """
        Apply discrete differentiation by x to f
        :param f: np.ndarray of dim 3
        :return: nd.ndarray
        """
        return np.gradient(f, self.hx, axis=0, edge_order=2)

    def dy(self, f):
        """
        Apply discrete differentiation by y to f
        :param f: np.ndarray of dim 3
        :return: nd.ndarray
        """
        return np.gradient(f, self.hy, axis=1, edge_order=2)

    def dt(self, f):
        """
        Apply discrete differentiation by t to f
        :param f: np.ndarray of dim 3
        :return: nd.ndarray
        """
        return np.gradient(f, self.ht, axis=2, edge_order=2)

    def interpolate(self, F):
        """
        Interpolate function by its values on region S

        :param F: np.ndarray
        :return: np.function
        """
        @np.vectorize
        def f(x, y, t):
            a, b, c, d, T = self.a, self.b, self.c, self.d, self.T
            num_points = self.num_points
            
            assert a <= x <= b, f"x = {x} is out of [{a}, {b}]"
            assert c <= y <= d, f"y = {y} is out of [{c}, {d}]"
            assert 0 <= t <= T, f"t = {t} is out of [{0}, {T}]"

            t0 = 0
            i = int((x - a) / (b - a) * (num_points - 1))
            j = int((y - c) / (d - c) * (num_points - 1))
            k = int((t - t0) / (T - t0) * (num_points - 1))

            if i == num_points - 1: i -= 1
            if j == num_points - 1: j -= 1
            if k == num_points - 1: k -= 1

            res = 0
            s = 0

            # return F[i, j, k]

            for idx in (i, i + 1):
                for idy in (j, j + 1):
                    for idt in (k, k + 1):
                        x_, y_, t_ = self.X[idx, 0, 0], self.Y[0, idy, 0], self.T_[0, 0, idt]
                        r = ((x - x_) ** 2 + (y - y_) ** 2 + (t - t_) ** 2) ** 0.5

                        s += r
                        res += F[idx, idy, idt] * r

            return res / s

        return f

    def nquad_0(self, f):
        """
        Compute triple integral of function f
        over region S0

        :param f: np.function
        :return: float
        """
        X, Y, T, hx, hy, ht = self.integrate_params
        F = f(X, Y, T-self.T)
        F[np.isnan(F)] = 0
        return np.sum(F*hx*hy*ht)

        """
        opts = {"epsrel": 1e-4}
        return nquad(f, [(self.a, self.b),
                         (self.c, self.d),
                         (-self.T, 0)], opts=[opts, opts, opts],
                    full_output=False)[0]
        """

    def nquad_g(self, f):
        """
        Compute triple integral of function f
        over region Sg

        :param f: np.function
        :return: float
        """
        res = 0
        X, Y, T, hx, hy, ht = self.integrate_params

        opts = {"epsrel": 1e-4}
        for dx in (self.a - self.b, 0, self.b - self.a):
            for dy in (self.c - self.d, 0, self.d - self.c):
                if dx == 0 and dy == 0:
                    continue

                F = f(X+dx, Y+dy, T)
                F[np.isnan(F)] = 0

                res += np.sum(F*hx*hy*ht)

                """
                res += nquad(f, [(self.a+dx, self.b+dx),
                                 (self.c-dy, self.d-dy),
                                 (0, self.T)], opts=[opts, opts, opts])[0]
                """

        return res

    def nquad_matrix(self, A):
        """
        Compute integral of matrix function A
        over its domain

        :param A: sp.Matrix of sp.Function
        :return:
        """
        n, m = A.shape
        P = np.empty((n, m))

        for i in range(n):
            for j in range(m):
                self.update_pbar()
                # f = lambdify((x, y, t), A[i, j], [{'Heaviside': lambda x: np.heaviside(x, 1)}, 'numpy'])
                f = lambdify((x, y, t), A[i, j], 'numpy')
                
                P[i, j] = self.nquad_0(f) + self.nquad_g(f)

        return P

    def compute_P(self, As):
        """
        Compute matrix P from function matrix A1, A2

        :param As: tuple. (A1, A2)
        :return: np.ndarray
        """
        self.update_pbar(f"Computing of matrix P")
        p = [[None, None],
             [None, None]]

        for i in range(2):
            for j in range(2):
                p[i][j] = self.nquad_matrix(As[i]*As[j].T)

        p1 = np.hstack(p[0])
        p2 = np.hstack(p[1])

        res = np.vstack([p1, p2])
        
        res[np.isnan(res)] = 0

        return res

    def compute_Y(self, phi_inf, params):
        """
        Compute vector Y based on bounded conditions
        and precomputed phi_inf
        :param phi_inf: np.ndarray
        :param params: dict
        :return: np.ndarray
        """
        self.update_pbar("Computing of vector Y")
        Y0 = np.array(params["Y0"])
        Yg = np.array(params["Yg"])
        dL0 = params["dL0"]
        dLg = params["dLg"]

        x0 = np.array(params["x0"])
        y0 = np.array(params["y0"])
        xg = np.array(params["xg"])
        yg = np.array(params["yg"])
        tg = np.array(params["tg"])

        n, m = Y0.shape
        for i in range(n):
            L_phi = self.interpolate(dL0[i](phi_inf))
            Y0[i] -= L_phi(x0, y0, 0)

        n, m = Yg.shape
        for i in range(n):
            L_phi = self.interpolate(dLg[i](phi_inf))
            Yg[i] -= L_phi(xg, yg, tg)

        Y0 = Y0.reshape((-1, 1))
        Yg = Yg.reshape((-1, 1))

        return np.vstack([Y0, Yg])

    def compute_As(self, params):
        """
        Compute matrices A1, A2 based on params

        :param params: dict
        :return: tuple of sympy expressions
        """
        self.update_pbar("Computing of matrices A1, A2")
        x_, y_, t_ = symbols("x_, y_, t_")
        Y0 = np.array(params["Y0"])
        Yg = np.array(params["Yg"])
        L0 = params["L0"]
        Lg = params["Lg"]

        x0 = np.array(params["x0"])
        y0 = np.array(params["y0"])
        xg = np.array(params["xg"])
        yg = np.array(params["yg"])
        tg = np.array(params["tg"])

        G = self.G_sp
        G = G.subs([(x, x-x_), (y, y-y_), (t, t-t_)])

        n, m = Y0.shape
        A1 = zeros(n, m)
        for i in range(n):
            LG = L0[i](G)
            for j in range(m):
                A1[i, j] = LG.subs([(x, x0[j]), (y, y0[j]), (t, 1e-16)])

        A1 = A1.subs([(x_, x), (y_, y), (t_, t)])
        A1 = A1.reshape(n*m, 1)

        n, m = Yg.shape
        A2 = zeros(n, m)

        for i in range(n):
            LG = Lg[i](G)
            for j in range(m):
                A2[i, j] = LG.subs([(x, xg[j]), (y, yg[j]), (t, tg[j])])

        A2 = A2.subs([(x_, x), (y_, y), (t_, t)])
        A2 = A2.reshape(n*m, 1)

        return A1, A2

    def update_pbar(self, desc=None):
        self.pbar.update(1)

        if desc is not None:
            self.pbar.set_description(desc)

    def solve(self, params):
        """
        Solve boundary value problem and find state function

        :param params: dict
        :return: np.ndarray
        """
        L0, R0, Lg, Rg = len(params["L0"]), params["R0"], len(params["Lg"]), params["Rg"]
        self.pbar = tqdm.tqdm(range((L0*R0+Lg*Rg)*(L0*R0+Lg*Rg+1) + 16))

        u = lambdify((x, y, t), S(params["u"]))
        phi_inf = self.convolve_green_inf(u)

        params["L0"] = [lambda f: f]
        params["Lg"] = [lambda f: f]
        params["dL0"] = [lambda f: f]
        params["dLg"] = [lambda f: f]

        Y = self.compute_Y(phi_inf, params)
        A1, A2 = self.compute_As(params)
        print("A1")
        print(A1)

        print("A2")
        print(A2)

        P = self.compute_P((A1, A2))

        print("P:")
        print(P)

        self.update_pbar("Computing of u0, ug")
        P_inv = np.linalg.pinv(P)

        # TODO: nan values
        P_inv[np.isnan(P_inv)] = 0

        A = A2.T.col_insert(0, A1.T)

        v0 = eval(params["u"])
        vg = eval(params["u"])

        Av0 = np.array(self.nquad_matrix(A1*Matrix([v0])).tolist())
        Avg = np.array(self.nquad_matrix(A2*Matrix([vg])).tolist())

        Av = np.vstack([Av0, Avg])

        print("Pinv:")
        print(P_inv)

        u0 = (A @ P_inv @ (Y - Av))[0, 0] + v0
        # u0 = lambdify((x, y, t), u0, [{'Heaviside': lambda x: np.heaviside(x, 1)}, 'numpy'])
        u0 = lambdify((x, y, t), u0, 'numpy')
        

        ug = (A @ P_inv @ (Y-Av))[0, 0] + vg
        # ug = lambdify((x, y, t), ug, [{'Heaviside': lambda x: np.heaviside(x, 1)}, 'numpy'])
        ug = lambdify((x, y, t), ug, 'numpy')

        phi_0 = self.convolve_green_0(u0)
        phi_g = self.convolve_green_g(ug)   

        phi = phi_inf + phi_0 + phi_g
        self.update_pbar("End")
        self.pbar.close()
        eps = abs(Y.T@Y - Y.T@P@P_inv@Y)
        print(f"\nEps: {eps[0, 0]}")

        return phi
    



def update_scene_2d(frame_index, phi, phi_real, patch1, patch2, ax):
    patch1.set_data(phi[:, :, frame_index])
    patch2.set_data(phi_real[:, :, frame_index])

    # ax[1].set_xlim(0, frame_index+1)
    # ax[1].set_ylim(np.min(selected_rewards)-1, np.max(selected_rewards)+1)
    return (patch1, patch2)


def animate_2d(phi, phi_real, figsize=(15, 5), repeat=False, interval=40):
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    
    patch1 = ax[0].imshow(phi[:, : ,0], cmap="magma")
    ax[0].set_axis_off()
    
    patch2 = ax[1].imshow(phi_real[:, : ,0], cmap="magma")
    ax[1].set_axis_off()

    ax[0].set_title("Phi")
    ax[1].set_title("Phi real")
    
    anim = animation.FuncAnimation(
        fig=fig,
        func=update_scene_2d,
        frames=phi.shape[2],
        fargs=(phi, phi_real, patch1, patch2, ax),
        repeat=repeat,
        interval=interval
    )
    plt.close()
    return anim


def update_scene_3d(frame_index, phi, phi_real, plot, plot_real, ax1, ax2, X_, Y_):
    plot[0].remove()
    plot[0] = ax1.plot_surface(X_, Y_, phi[:, :, frame_index], cmap="magma")

    plot_real[0].remove()
    plot_real[0] = ax2.plot_surface(X_, Y_, phi_real[:, :, frame_index], cmap="magma")


def animate_3d(phi, phi_real, X, Y, figsize=(15, 5), repeat=False, interval=40):
    
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    X_, Y_ = np.meshgrid(X, Y)

    plot = [ax1.plot_surface(X_, Y_, phi[:, :, 0], rstride=1, cstride=1)]
    plot_real = [ax1.plot_surface(X_, Y_, phi_real[:, :, 0], rstride=1, cstride=1)]

    ax1.set_zlim(-1.1, 1.1)
    ax2.set_zlim(-1.1, 1.1)

    ax1.set_title("Phi")
    ax2.set_title("Phi real")
    
    anim = animation.FuncAnimation(
        fig=fig,
        func=update_scene_3d,
        frames=phi.shape[2],
        fargs=(phi, phi_real, plot, plot_real, ax1, ax2, X_, Y_),
        repeat=repeat,
        interval=interval
    )
    plt.close()
    return anim


def get_real_phi(params, num_points=80):
    a = params["a"]
    b = params["b"]
    c = params["c"]
    d = params["d"]
    T = params["T"]
    x, y, t = symbols("x, y, t")

    f = S(params["f"])

    L = lambda f: eval("diff(f, t)-(diff(f, x, x)+diff(f, y, y))")
    u = L(f)

    f = sp.lambdify((x, y, t), f, "numpy")

    X = np.linspace(a, b, num_points)
    Y = np.linspace(c, d, num_points)
    T_ = np.linspace(0, T, num_points)

    phi = f(X[:, None, None], Y[None, :, None], T_[None, None, :])

    return phi 


def main():
    with open("test_params_jeka.json") as f:
        params = json.load(f)
    
    a, b = params["a"], params["b"]
    c, d = params["c"], params["d"]
    T = params["T"]

    slv = Solver(a, b, c, d, T)
    num_points = slv.num_points

    fps = 10 * num_points / 50

    # u = 25*sin(5*t)*sin(3*x)*sin(4*y) + 5*sin(3*x)*sin(4*y)*cos(5*t)
    u = S(params["u"])
    u = lambdify((x, y, t), u, "numpy")

    phi = slv.solve(params)
    phi_real = get_real_phi(params, num_points=num_points)

    X = np.linspace(a, b, num_points)
    Y = np.linspace(c, d, num_points)
    
    ani = animate_3d(phi, phi_real, X, Y, figsize=(15, 5), repeat=True)
    ani.save("phi.gif", fps=fps)


if __name__ == "__main__":
    main()
