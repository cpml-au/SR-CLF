import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sympy import symbols, lambdify, diff, Matrix, sin, cos, tanh, exp, log, sqrt, simplify


def plot_contour(
    X1,
    X2,
    Z,
    title,
    filename,
    directory,
    show=True,
):
    """
    Single contour plot with fixed settings.
    """

    plt.figure(figsize=(6, 4))
    
    # contour = plt.contourf(X1, X2, Z, levels=50, cmap=cm.coolwarm)
    # plt.colorbar(contour)
    
    contour_filled = plt.contourf(X1, X2, Z, levels=200, cmap=cm.coolwarm)
    # Contour lines
    contour_lines = plt.contour(X1, X2, Z, levels=50, colors='black', linewidths=0.5)
    # Add color bar and labels
    plt.colorbar(contour_filled)

    plt.title(title)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.grid()

    if filename is not None:
        os.makedirs(directory, exist_ok=True)
        plt.savefig(
            os.path.join(directory, filename),
            dpi=100,
            bbox_inches="tight",
        )

    if show:
        plt.show()
    else:
        plt.close()


def plot_states_and_path(
    *,
    directory: str,
    filename_states: str,
    filename_path: str,
    x0,
    xylabel_font_size=22,
    axis_font_size=20,
    legend_font_size=20,
    dpi=300,
    show=True,
    # --- curves over time (list of dicts) ---
    # each item: {"t": t_vals, "x": x_vals, "label1": "...", "label2": "...",
    #            "color": "...", "ls1": "--", "ls2": ":", "lw": 4.0,
    #            optional marker settings: "marker1"/"marker2"/"ms"/"mfc"/"step"}
    time_series=None,
    # --- curves in state space (list of dicts) ---
    # each item: {"x": x_vals, "label": "...", "color": "...", "lw": 4.0,
    #            optional marker settings: "marker"/"ms"/"mfc"/"step"}
    path_series=None,
    # --- xtick spacing ---
    n_tick=5,
):
    """
    One function to generate:
      1) States over time plot
      2) 2D trajectory plot (x1 vs x2)

    Call multiple times with different series sets (NN / VF markers / FLF / etc.).
    """

    if time_series is None:
        time_series = []
    if path_series is None:
        path_series = []

    os.makedirs(directory, exist_ok=True)

    # ------------------ Plot states over time ------------------
    plt.figure(figsize=(8, 7))

    # plot each controller/model (LQR/NN/CLF/VF/FLF/etc.)
    for s in time_series:
        t = s["t"]
        x = s["x"]  # expects x[0], x[1]

        lw = s.get("lw", 4.0)
        color = s.get("color", "black")
        ls1 = s.get("ls1", "--")
        ls2 = s.get("ls2", ":")
        step = s.get("step", None)

        # optional marker mode (for VF-style sparse markers)
        marker1 = s.get("marker1", None)
        marker2 = s.get("marker2", None)
        ms = s.get("ms", 10)
        mfc = s.get("mfc", "none")

        if step is None:
            plt.plot(t, x[0], label=s["label1"], color=color, linestyle=ls1, linewidth=lw)
            plt.plot(t, x[1], label=s["label2"], color=color, linestyle=ls2, linewidth=lw)
        else:
            idx = np.arange(0, len(t), step)
            # x1 markers
            plt.plot(
                t[idx], x[0][idx],
                label=s["label1"],
                color=color,
                linestyle="None",
                marker=marker1 if marker1 is not None else "o",
                markerfacecolor=mfc,
                markersize=ms,
                linewidth=lw,
            )
            # x2 markers
            plt.plot(
                t[idx], x[1][idx],
                label=s["label2"],
                color=color,
                linestyle="None",
                marker=marker2 if marker2 is not None else "s",
                markerfacecolor=mfc,
                markersize=ms,
                linewidth=lw,
            )

    plt.xlabel(r"$t$", fontsize=xylabel_font_size)
    plt.ylabel("State Variables", fontsize=xylabel_font_size)

    if len(time_series) > 0:
        tmax = max(np.max(s["t"]) for s in time_series)
        xt = np.arange(0, np.floor(tmax / n_tick) * n_tick + n_tick, n_tick)
        plt.xticks(xt, fontsize=axis_font_size)
    else:
        plt.xticks(fontsize=axis_font_size)

    plt.yticks(fontsize=axis_font_size)
    plt.legend(fontsize=legend_font_size)

    plt.savefig(os.path.join(directory, filename_states), dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

    # ------------------ Plot 2D trajectory ------------------
    plt.figure(figsize=(8, 7))

    for s in path_series:
        x = s["x"]  # expects x[0], x[1]
        lw = s.get("lw", 4.0)
        color = s.get("color", "black")
        step = s.get("step", None)

        marker = s.get("marker", None)
        ms = s.get("ms", 10)
        mfc = s.get("mfc", "none")

        if step is None:
            plt.plot(x[0], x[1], label=s["label"], color=color, linewidth=lw)
        else:
            idx = np.arange(0, x.shape[1], step) if hasattr(x, "shape") else np.arange(0, len(x[0]), step)
            plt.plot(
                np.asarray(x[0])[idx],
                np.asarray(x[1])[idx],
                label=s["label"],
                color=color,
                linestyle="None",
                marker=marker if marker is not None else "o",
                markerfacecolor=mfc,
                markersize=ms,
                linewidth=lw,
            )

    # start/end points (exactly like you did)
    plt.scatter(x0[0], x0[1], color="green", label="Start", zorder=5)
    plt.scatter(0, 0, color="red", label="End", zorder=5)

    plt.xlabel(r"$x_1$", fontsize=xylabel_font_size)
    plt.ylabel(r"$x_2$", fontsize=xylabel_font_size)
    plt.xticks(fontsize=axis_font_size)
    plt.yticks(fontsize=axis_font_size)
    plt.legend(fontsize=legend_font_size)

    plt.savefig(os.path.join(directory, filename_path), dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_u_over_time(
    *,
    directory: str,
    filename: str,
    xylabel_font_size=22,
    axis_font_size=20,
    legend_font_size=20,
    dpi=300,
    show=True,
    n_tick=5,
    # list of dicts describing each curve
    # each item:
    #   {"t": t_vals, "u": u_vals, "label": "...", "color": "...", "lw": 4.0,
    #    optional marker-mode: "step": 30, "marker": "o", "ms": 10, "mfc": "none"}
    series=None,
):
    """
    One function to plot control policy u(t) with fixed style.
    Call multiple times with different 'series' lists.
    """
    if series is None:
        series = []

    os.makedirs(directory, exist_ok=True)

    plt.figure(figsize=(8, 7))

    for s in series:
        t = s["t"]
        u = s["u"]  # can be (N,) or (N,2) etc.
        label = s["label"]
        color = s.get("color", "black")
        lw = s.get("lw", 4.0)

        # choose u[:,1] if 2D else u
        u1 = u[:, 1] if hasattr(u, "ndim") and u.ndim > 1 else u

        step = s.get("step", None)
        if step is None:
            plt.plot(t, u1, label=label, color=color, linewidth=lw)
        else:
            idx = np.arange(0, len(t), step)
            marker = s.get("marker", "o")
            ms = s.get("ms", 10)
            mfc = s.get("mfc", "none")
            plt.plot(
                t[idx],
                u1[idx],
                label=label,
                color=color,
                linestyle="None",
                markerfacecolor=mfc,
                marker=marker,
                markersize=ms,
                linewidth=lw,
            )

    plt.xlabel(r"$t$", fontsize=xylabel_font_size)
    plt.ylabel("Control Policy", fontsize=xylabel_font_size)

    if len(series) > 0:
        tmax = max(np.max(s["t"]) for s in series)
        xt = np.arange(0, np.floor(tmax / n_tick) * n_tick + n_tick, n_tick)
        plt.xticks(xt, fontsize=axis_font_size)
    else:
        plt.xticks(fontsize=axis_font_size)

    plt.yticks(fontsize=axis_font_size)
    plt.legend(fontsize=legend_font_size)

    plt.savefig(os.path.join(directory, filename), dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def check_eq_inside_ineq_and_plot(
    *,
    x1,
    x2,
    eq_sym,          # sympy expression for equality: eq(x1,x2)=0
    ineq_sym,        # sympy expression for inequality: ineq(x1,x2)<0
    xmin=-2,
    xmax=2,
    ymin=-2,
    ymax=2,
    res=1300,
    tol=1e-6,
    ineq_thresh=1e-5,
    figsize=(4, 4),
    dpi=140,
    show=True,
):
    """
    Replicates your exact workflow:
      - build grid
      - evaluate eq and ineq
      - find points where |eq| < tol
      - check ineq on those points
      - print all_inside + fraction_inside
      - plot: blue contour eq=0 and grey mask ineq<0
    """

    # --- 1) Domain / grid ---
    x_check = np.linspace(xmin, xmax, res)
    y_check = np.linspace(ymin, ymax, res)
    X_check, Y_check = np.meshgrid(x_check, y_check)

    # --- 2) Lambdify ---
    eq = lambdify((x1, x2), eq_sym, "numpy")
    ineq_expr = lambdify((x1, x2), ineq_sym, "numpy")

    # --- 3) Find approximate points where eq ~ 0 ---
    eq_vals = eq(X_check, Y_check)
    curve_points = np.abs(eq_vals) < tol

    # --- 4) Check inequality at those points ---
    ineq_vals = ineq_expr(X_check, Y_check)
    ineq_on_curve = ineq_vals[curve_points]

    all_inside = np.all(ineq_on_curve < ineq_thresh)
    fraction_inside = np.mean(ineq_on_curve < ineq_thresh)

    print("Is eq completely inside ineq? ->", all_inside)
    print("Fraction of eq points outside ineq:", fraction_inside)

    # --- 5) Plot ---
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.contour(X_check, Y_check, eq_vals, levels=[0], linewidths=2, colors="blue")
    mask = ineq_vals < 0
    ax.imshow(
        mask.astype(float),
        extent=[xmin, xmax, ymin, ymax],
        origin="lower",
        alpha=0.25,
        cmap="Greys",
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_title("Blue = eq(x,y)=0, Grey = ineq(x,y)<0")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return all_inside, fraction_inside


def Plot3D(X, Y, V, r):
    # Plot Lyapunov functions  
    fig = plt.figure(figsize=(1, 1))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X,Y,V, rstride=5, cstride=5, alpha=0.5, cmap=cm.coolwarm)
    ax.contour(X,Y,V,10, zdir='z', offset=0, cmap=cm.coolwarm)
    
    # Plot Valid region computed by dReal
    # theta = np.linspace(0,2*np.pi,50)
    # xc = r*cos(theta)
    # yc = r*sin(theta)
    theta = np.linspace(0, 2*np.pi, 50)
    xc = r * np.cos(theta)
    yc = r * np.sin(theta)
    ax.plot(xc[:],yc[:],'r',linestyle='--', linewidth=2 ,label='Valid region')
    plt.legend(loc='upper right')
    return ax

def Plotflow(Xd, Yd, t):
    # Plot phase plane 
    DX, DY = f([Xd, Yd],t)
    plt.streamplot(Xd,Yd,DX,DY, color=('gray'), linewidth=0.5,
                  density=0.8, arrowstyle='-|>', arrowsize=1)


def Plot3D_V_less_than_zero(X, Y, V, r):
    # Mask out regions where V >= 0
    V_masked = np.where(V < 0, V, np.nan)
    
    # Plot Lyapunov functions
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, V_masked, rstride=5, cstride=5, alpha=0.5, cmap=cm.coolwarm)
    ax.contour(X, Y, V_masked, 10, zdir='z', offset=0, cmap=cm.coolwarm)
    
    # Plot Valid region computed by dReal
    theta = np.linspace(0, 2 * np.pi, 50)
    xc = r * np.cos(theta)
    yc = r * np.sin(theta)
    ax.plot(xc[:], yc[:], 'r', linestyle='--', linewidth=2, label='Valid region')
    plt.legend(loc='upper right')
    return ax


def levelset(x1,x2,V,r):
    N_level = 1000
    cs = plt.contour(x1,x2,V,N_level)
    for i in range(len(cs.levels)-2):
        elements = cs.allsegs[i+1][0]
        elements_sq = np.sum(elements**2,axis = 1)
        if r**2 - np.max(elements_sq) < 1e-6:
            c = cs.levels[i]
            break
    return c
    