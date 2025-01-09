import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patheffects import Stroke, Normal

from SALib.analyze.sobol import analyze
from SALib.sample.sobol import sample

plt.rcParams["axes.prop_cycle"] = plt.cycler(
    "color", ["#ffffcc", "#c2e699", "#78c679", "#31a354", "#006837"][::-1]
)

line_shade = [
    Stroke(linewidth=4, foreground="grey"),
    Normal(),
]

t = np.arange(0, 20.1, 0.1)


def simple_model():
    st.write(
        r"""
        A global sensitivity analysis explores the variance of the output
        based on changes in the parameter space.

        Consider a simple model:

        $$
        \begin{equation}
            f(t, a, b) = a t^2 + b
        \end{equation}
        $$

        Where $a \in [a_\textsf{min}, a_\textsf{max}]$ and
        $b \in [b_\textsf{min}, b_\textsf{max}]$.
        """
    )

    def fn(t: np.ndarray, a: float, b: float):
        return a * (t**2) + b

    def wrapped(X):
        a, b = X
        return fn(t, a, b)

    @st.cache_resource
    def generate_sample(problem):
        return sample(problem, 1024, seed=100)

    @st.cache_resource
    def chunk_model(param_values):
        Y = [wrapped(p) for p in param_values]
        return np.array(Y)

    @st.cache_resource
    def chunck_analysis(problem, ndY):
        return [analyze(problem, Y) for Y in ndY.T]

    st.subheader("Sobol indices")

    with st.sidebar:
        a_bounds = st.slider(
            r"$a_{\textsf{min}},\,a_{\textsf{max}}$",
            min_value=-2.0,
            max_value=2.0,
            value=(-1.0, 1.0),
            step=0.1,
        )
        b_bounds = st.slider(
            r"$b_{\textsf{min}},\,b_{\textsf{max}}$",
            min_value=-20,
            max_value=20,
            value=(-10, 10),
        )

    sobol_plot_cnt = st.container()

    # Define the model inputs
    names = ["$a$", "$b$"]

    problem = dict(
        num_vars=len(names),
        names=names,
        bounds=[
            a_bounds,
            b_bounds,
        ],
    )

    # Generate samples
    param_values = generate_sample(problem)

    # Run model
    ndY = chunk_model(param_values)

    # Analyze results
    Si = chunck_analysis(problem, ndY)

    # Rearange for plotting
    fo_names = problem["names"]
    first_order = np.array([s["S1"] for s in Si]).T
    fo_confidence = np.array([s["S1_conf"] for s in Si]).T

    so_names = [f"{fo_names[0]}, {fo_names[1]}"]
    second_order = np.array([[s["S2"][0, 1] for s in Si]])
    so_confidence = np.array([[s["S2_conf"][0, 1] for s in Si]])

    tot_names = fo_names
    tot_order = np.array([s["ST"] for s in Si]).T
    tot_confidence = np.array([s["ST_conf"] for s in Si]).T

    t_select = st.select_slider("Select a time", np.round(t, 2))
    tidx = np.argmin(np.abs(t - t_select))

    fig, axs = plt.subplots(
        3,
        1,
        sharex="col",
        sharey="row",
        figsize=(5, 6),
    )

    ax = axs[0]
    for name, si, conf in zip(fo_names, first_order, fo_confidence):
        ax.plot(t, si, label=name)
        ax.fill_between(t, si - conf, si + conf, alpha=0.2)
        ax.set_ylabel("First-order\nindices $S_i$")

    ax = axs[1]
    for name, si, conf in zip(so_names, second_order, so_confidence):
        ax.plot(t, si, label=name)
        ax.fill_between(t, si - conf, si + conf, alpha=0.2)
        ax.set_ylabel("Second-order\nindices $S_{i,j}$")

    ax = axs[2]
    for name, si, conf in zip(tot_names, tot_order, tot_confidence):
        ax.plot(t, si, label=name)
        ax.fill_between(t, si - conf, si + conf, alpha=0.2)
        ax.set_ylabel("Total-order\n indices $S_{i,j}$")

    ax.set_xlabel("$t$")

    for ax in axs.flatten():
        ax.axhline(1, color="gray", ls="dotted", lw=0.5, c="k")
        ax.axhline(0, color="gray", ls="dotted", lw=0.5, c="k")
        ax.axvline(x=t_select, c="k", lw=1, ls="dashed")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        ax.spines.top.set_visible(False)
        ax.spines.right.set_visible(False)

    sobol_plot_cnt.pyplot(fig)

    fig, axs = plt.subplots(
        1, 3, figsize=(10, 3), sharey=True, gridspec_kw={"wspace": 0.05}
    )
    colors = ["#31a354", "#006837"][::-1]
    ax = axs[0]
    dummy_x = np.arange(len(tot_names))
    ax.bar(dummy_x, tot_order.T[tidx], facecolor=colors)
    ax.errorbar(
        dummy_x,
        tot_order.T[tidx],
        yerr=tot_confidence.T[tidx],
        fmt="x",
        color="k",
    )
    ax.set_xticks(dummy_x)
    ax.set_xticklabels(tot_names)
    ax.set_title("Total-order")

    ax = axs[1]
    dummy_x = np.arange(len(fo_names))
    ax.bar(dummy_x, first_order.T[tidx])
    ax.errorbar(
        dummy_x,
        first_order.T[tidx],
        yerr=fo_confidence.T[tidx],
        fmt="x",
        color="k",
    )
    ax.set_xticks(dummy_x)
    ax.set_xticklabels(fo_names)
    ax.set_title("First-order")

    ax = axs[2]
    dummy_x = np.arange(len(so_names))
    ax.bar(dummy_x, second_order.T[tidx])
    ax.errorbar(
        dummy_x,
        second_order.T[tidx],
        yerr=so_confidence.T[tidx],
        fmt="x",
        color="k",
    )
    ax.set_xticks(dummy_x)
    ax.set_xticklabels(so_names)
    ax.set_title("Second-order")

    for ax in axs:
        ax.spines.top.set_visible(False)
        ax.spines.right.set_visible(False)
        ax.set_ylim(top=1.1)

    st.pyplot(fig)


def non_linear_model():
    st.write(
        r"""
        Consider a model with a non-linear combination of parameters:

        $$
        \begin{equation}
            f(t, a, b) = a\left( t^2 - b \right)
        \end{equation}
        $$

        Where $a \in [a_\textsf{min}, a_\textsf{max}]$ and
        $b \in [b_\textsf{min}, b_\textsf{max}]$.
        """
    )

    def fn(t: np.ndarray, a: float, b: float):
        return a * (t**2 + b)

    def wrapped(X):
        a, b = X
        return fn(t, a, b)

    @st.cache_resource
    def generate_sample(problem):
        return sample(problem, 1024, seed=100)

    @st.cache_resource
    def chunk_model(param_values):
        Y = [wrapped(p) for p in param_values]
        return np.array(Y)

    @st.cache_resource
    def chunck_analysis(problem, ndY):
        return [analyze(problem, Y) for Y in ndY.T]

    st.subheader("Sobol indices")

    with st.sidebar:
        a_bounds = st.slider(
            r"$a_{\textsf{min}},\,a_{\textsf{max}}$",
            min_value=-2.0,
            max_value=2.0,
            value=(-1.0, 1.0),
            step=0.1,
        )
        b_bounds = st.slider(
            r"$b_{\textsf{min}},\,b_{\textsf{max}}$",
            min_value=-20,
            max_value=20,
            value=(-10, 10),
        )

    sobol_plot_cnt = st.container()

    # Define the model inputs
    names = ["$a$", "$b$"]

    problem = dict(
        num_vars=len(names),
        names=names,
        bounds=[
            a_bounds,
            b_bounds,
        ],
    )

    # Generate samples
    param_values = generate_sample(problem)

    # Run model
    ndY = chunk_model(param_values)

    # Analyze results
    Si = chunck_analysis(problem, ndY)

    # Rearange for plotting
    fo_names = problem["names"]
    first_order = np.array([s["S1"] for s in Si]).T
    fo_confidence = np.array([s["S1_conf"] for s in Si]).T

    so_names = [f"{fo_names[0]}, {fo_names[1]}"]
    second_order = np.array([[s["S2"][0, 1] for s in Si]])
    so_confidence = np.array([[s["S2_conf"][0, 1] for s in Si]])

    tot_names = fo_names
    tot_order = np.array([s["ST"] for s in Si]).T
    tot_confidence = np.array([s["ST_conf"] for s in Si]).T

    t_select = st.select_slider("Select a time", np.round(t, 2))
    tidx = np.argmin(np.abs(t - t_select))

    fig, axs = plt.subplots(
        3,
        1,
        sharex="col",
        sharey="row",
        figsize=(5, 6),
    )

    ax = axs[0]
    for name, si, conf in zip(fo_names, first_order, fo_confidence):
        ax.plot(t, si, label=name)
        ax.fill_between(t, si - conf, si + conf, alpha=0.2)
        ax.set_ylabel("First-order\nindices $S_i$")

    ax = axs[1]
    for name, si, conf in zip(so_names, second_order, so_confidence):
        ax.plot(t, si, label=name)
        ax.fill_between(t, si - conf, si + conf, alpha=0.2)
        ax.set_ylabel("Second-order\nindices $S_{i,j}$")

    ax = axs[2]
    for name, si, conf in zip(tot_names, tot_order, tot_confidence):
        ax.plot(t, si, label=name)
        ax.fill_between(t, si - conf, si + conf, alpha=0.2)
        ax.set_ylabel("Total-order\n indices $S_{i,j}$")

    ax.set_xlabel("$t$")

    for ax in axs.flatten():
        ax.axhline(1, color="gray", ls="dotted", lw=0.5, c="k")
        ax.axhline(0, color="gray", ls="dotted", lw=0.5, c="k")
        ax.axvline(x=t_select, c="k", lw=1, ls="dashed")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        ax.spines.top.set_visible(False)
        ax.spines.right.set_visible(False)

    sobol_plot_cnt.pyplot(fig)

    fig, axs = plt.subplots(
        1, 3, figsize=(10, 3), sharey=True, gridspec_kw={"wspace": 0.05}
    )
    colors = ["#31a354", "#006837"][::-1]
    ax = axs[0]
    dummy_x = np.arange(len(tot_names))
    ax.bar(dummy_x, tot_order.T[tidx], facecolor=colors)
    ax.errorbar(
        dummy_x,
        tot_order.T[tidx],
        yerr=tot_confidence.T[tidx],
        fmt="x",
        color="k",
    )
    ax.set_xticks(dummy_x)
    ax.set_xticklabels(tot_names)
    ax.set_title("Total-order")

    ax = axs[1]
    dummy_x = np.arange(len(fo_names))
    ax.bar(dummy_x, first_order.T[tidx])
    ax.errorbar(
        dummy_x,
        first_order.T[tidx],
        yerr=fo_confidence.T[tidx],
        fmt="x",
        color="k",
    )
    ax.set_xticks(dummy_x)
    ax.set_xticklabels(fo_names)
    ax.set_title("First-order")

    ax = axs[2]
    dummy_x = np.arange(len(so_names))
    ax.bar(dummy_x, second_order.T[tidx])
    ax.errorbar(
        dummy_x,
        second_order.T[tidx],
        yerr=so_confidence.T[tidx],
        fmt="x",
        color="k",
    )
    ax.set_xticks(dummy_x)
    ax.set_xticklabels(so_names)
    ax.set_title("Second-order")

    for ax in axs:
        ax.spines.top.set_visible(False)
        ax.spines.right.set_visible(False)
        ax.set_ylim(top=1.1)

    st.pyplot(fig)


def a_benchmark():
    st.write(
        r"""
        Consider the following model:

        $$
        \begin{equation}
            f(x) = \sin{x_0} + 7\sin^2{x_1} + \tfrac{1}{10}x_2^4\sin{x_0}
        \end{equation}
        $$

        Where $x = \{x_0, x_1, x_2\} \in [-\pi,\pi]$. This is known as the
        Ishigami function and it's commonly used to benchmark sensitivity
        analysis tools. 
        """
    )

    def fn(x: np.ndarray):
        a, b = 7, 0.1

        return (
            np.sin(x[0])
            + a * np.power(np.sin(x[1]), 2)
            + b * np.power(x[2], 4) * np.sin(x[0])
        )

    @st.cache_resource
    def chunk_model(param_values):
        return np.array([fn(p) for p in param_values])

    # Define the model inputs
    names = ["$x_0$", "$x_1$", "$x_2$"]

    problem = dict(
        num_vars=len(names),
        names=names,
        bounds=[[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]],
    )

    # Generate samples
    param_values = sample(problem, 1024, seed=100)

    # Run model
    ndY = chunk_model(param_values)

    # Analyze results
    Si = analyze(problem, ndY)

    fig, axs = plt.subplots(
        1, 3, sharey=True, figsize=(6, 3), gridspec_kw={"wspace": 0.03}
    )
    for ax in axs:
        ax.spines.top.set_visible(False)
        ax.spines.right.set_visible(False)
    Si.plot(ax=axs)
    st.pyplot(fig)

    st.write(Si.to_df())


if __name__ == "__main__":
    with open("assets/style.css") as css:
        st.html(f"""<style>{css}</style>""")

    st.title("Sensitivity indices")

    pages = [
        st.Page(simple_model, title="A linear example"),
        st.Page(non_linear_model, title="A non-linear example"),
        st.Page(a_benchmark, title="Ishigami benchmark"),
    ]

    nav = st.navigation(pages)
    nav.run()
