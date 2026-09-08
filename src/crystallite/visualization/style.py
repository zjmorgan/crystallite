"""Matplotlib styles used by crystallite verification plots."""


PGF_WITH_PDFLATEX = {
    "font.family": "sans-serif",
    "font.size": 12,
    "text.usetex": True,
    "pgf.rcfonts": False,
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": " ".join(
        [
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{libertine}",
            r"\usepackage[libertine]{newtxmath}",
            r"\usepackage[scientific-notation=true]{siunitx}",
            r"\usepackage{tensor}",
            r"\usepackage{physics}",
            r"\usepackage{mhchem}",
        ]
    ),
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.bottom": True,
    "axes.spines.left": True,
    "axes.xmargin": 0.05,
    "axes.ymargin": 0.05,
}


def configure_pgf():
    """Configure Matplotlib for PGF output with the project plot style.

    Returns
    -------
    module
        The configured :mod:`matplotlib.pyplot` module.

    Raises
    ------
    ImportError
        If Matplotlib is not installed.

    Notes
    -----
    This function changes Matplotlib's process-wide backend and rcParams.
    Call it before importing :mod:`matplotlib.pyplot` in a plotting script.
    """
    try:
        import matplotlib as mpl
    except ImportError as error:
        raise ImportError(
            "plotting requires the optional matplotlib dependency"
        ) from error

    mpl.use("pgf")
    mpl.rcParams.update(PGF_WITH_PDFLATEX)

    import matplotlib.pyplot as plt
    from cycler import cycler

    plt.rcParams["axes.prop_cycle"] = cycler(
        "color", plt.get_cmap("Set2").colors
    )
    return plt