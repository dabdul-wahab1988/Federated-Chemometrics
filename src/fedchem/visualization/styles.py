import matplotlib
import matplotlib.pyplot as plt


def set_plot_style(style_name: str = 'publication', dpi: int = 300, palette: str = 'colorblind'):
    # minimal style, consistent with publication look
    plt.style.use('seaborn-whitegrid')
    matplotlib.rcParams['figure.dpi'] = dpi
    matplotlib.rcParams['font.size'] = 10
    matplotlib.rcParams['axes.titlesize'] = 11
    matplotlib.rcParams['axes.labelsize'] = 10
    matplotlib.rcParams['legend.fontsize'] = 9
    matplotlib.rcParams['figure.figsize'] = (8, 6)
