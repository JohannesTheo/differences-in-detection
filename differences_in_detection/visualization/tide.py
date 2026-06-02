import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tidecv.plotting import Plotter


def alpha_blend(fg_color, bg_color=(1.0, 1.0, 1.0), alpha=0.5):
    return tuple(alpha * fg + (1 - alpha) * bg for fg, bg in zip(fg_color, bg_color))


def tide_bar_special(errors, m1, m2, ax, plotter):

    df1 = errors[m1]['special']
    df2 = errors[m2]['special']
    df = pd.concat([df1, df2], keys=[m1, m2], names=["model"])
    df = df.reset_index(level="model").reset_index(drop=True)

    sns.barplot(data=df, x='Error Type', y='Delta mAP', hue='model', width=0.75, errorbar=None, ax=ax)
    ax.get_legend().remove()

    c1 = [sns.desaturate(c, 0.75) for c in list(plotter.colors_special.values())]
    c2 = [sns.desaturate(c, 0.75) for c in list(plotter.colors_special.values())]
    #c2 = [(*c, 0.5) for c in c2]
    c2 = [alpha_blend(c, alpha=0.5) for c in c2]

    colors = [*c1, *c2]

    bars = [p for p in ax.patches if (p.get_height() !=0 and p.get_width()!=0)]
    for i, bar in enumerate(bars):
        bar.set_facecolor(colors[i])
        bar.set_edgecolor("black")
        bar.set_linewidth(0.5)
        if i >= len(bars) / 2:
            bar.set_hatch('////')
            bar._hatch_color = colors[int(i - len(bars) / 2)]

    ax.set_ylim(0, plotter.MAX_SPECIAL_DELTA_AP)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['FP', 'FN'])
    ax.grid(False)
    sns.despine(left=True, bottom=True, right=True)


def tide_bar_main(errors, m1, m2, ax, plotter, hbar_names=True):

    df1 = errors[m1]['main']
    df2 = errors[m2]['main']
    df = pd.concat([df1, df2], keys=[m1, m2], names=["model"])
    df = df.reset_index(level="model").reset_index(drop=True)

    sns.barplot(data=df, x='Delta mAP', y='Error Type', hue='model', width=0.75,  errorbar=None, ax=ax)
    ax.get_legend().remove()

    c1 = [sns.desaturate(c, 1.0) for c in list(plotter.colors_main.values())]
    c2 = [sns.desaturate(c, 1.0) for c in list(plotter.colors_main.values())]
    c2 = [alpha_blend(c, alpha=0.5) for c in c2]
    colors = [*c1, *c2]

    # TODO: this does not work correctly for larger t, e.g. when some things are actually 0...
    bars = [p for p in ax.patches if (p.get_height() !=0 and p.get_width()!=0)]
    for i, bar in enumerate(bars):
        bar.set_facecolor(colors[i])
        bar.set_edgecolor("black")
        bar.set_linewidth(0.5)
        if i >= len(bars) / 2:
            bar.set_hatch('////')
            bar._hatch_color = colors[int(i - len(bars) / 2)]


    ax.set_xlim(0, plotter.MAX_MAIN_DELTA_AP)
    ax.set_xlabel('')
    ax.set_ylabel('')
    if not hbar_names:
        ax.set_yticklabels([''] * 6)

    ax.grid(False)
    sns.despine(left=True, bottom=True, right=True)


def tide_pie(errors, model_name, ax, plotter, secondary=False):

    df = errors[model_name]['main']
    error_types = df['Error Type'].to_list()
    error_sum = df['Delta mAP'].sum()
    error_sizes = [e / error_sum for e in df['Delta mAP'].to_list()]

    # pie plot for error type breakdown
    patches, outer_text, inner_text = ax.pie(
        error_sizes,
        colors=plotter.colors_main.values(),
        labels=error_types,
        autopct='%1.1f%%', startangle=90)

    if secondary:
        c1 = [sns.desaturate(c, 1.0) for c in list(plotter.colors_main.values())]
        #c2 = [(1.0, 1.0, 1.0) for c in list(plotter.colors_main.values())]
        #c2 = [(*c, 0.35) for c in c2]
        c2 = [sns.desaturate(c, 1.0) for c in list(plotter.colors_main.values())]
        c2 = [alpha_blend((1.0, 1.0, 1.0), c, alpha=0.35) for c in c2]
        for i, wedge in enumerate(ax.patches):
            wedge.set_facecolor(c1[i])
            #wedge.set_linewidth(0.5)
            wedge.set_hatch('////')
            wedge._hatch_color = c2[i]

    for text in outer_text + inner_text:
        text.set_text('')

    for i in range(len(plotter.colors_main)):
        if error_sizes[i] > 0.05:
            inner_text[i].set_text(list(plotter.colors_main.keys())[i])
        inner_text[i].set_fontsize(10)
        inner_text[i].set_fontweight('bold')

    ax.axis('equal')

    ax.set_title(model_name, fontdict={'fontsize': 14})#, 'fontweight': 'bold'})


def update_max_limits(errors, plotter):

    max_main = max(errors['main'].values())
    max_spec = max(errors['special'].values())

    if max_main > plotter.MAX_MAIN_DELTA_AP:
        plotter.MAX_MAIN_DELTA_AP = (int(max_main) % 5) * 5 + 5

    if max_spec > plotter.MAX_SPECIAL_DELTA_AP:
        plotter.MAX_SPECIAL_DELTA_AP = (int(max_spec) % 5) * 5 + 5


def get_error_dfs(errors):

    error_dfs = {
        errtype: pd.DataFrame(data={
            'Error Type': list(errors[errtype].keys()),
            'Delta mAP':  list(errors[errtype].values()),
            }) for errtype in ['main', 'special']}

    return error_dfs


def plot_tide_results(m1_results, m2_results, m1_name, m2_name, title_iou):

    fig = plt.figure(figsize=(10, 2))
    fig.suptitle(f"TIDE Error Analysis @ IoU={title_iou}", fontsize=20, y=1.25)

    gs = gridspec.GridSpec(1, 4, height_ratios=[1], width_ratios=[2,2,2,1], wspace=0.5)

    plotter = Plotter()
    update_max_limits(m1_results, plotter)
    update_max_limits(m2_results, plotter)

    errors = {
        m1_name: get_error_dfs(m1_results),
        m2_name: get_error_dfs(m2_results)
    }

    tide_bar_main(   errors, m1_name, m2_name, fig.add_subplot(gs[0, 2]), plotter=plotter)
    tide_bar_special(errors, m1_name, m2_name, fig.add_subplot(gs[0, 3]), plotter=plotter)
    tide_pie(errors, m1_name, fig.add_subplot(gs[0, 0]), plotter=plotter)
    tide_pie(errors, m2_name, fig.add_subplot(gs[0, 1]), plotter=plotter, secondary=True)

    plt.show()
