import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec


def plot_cocoapi_results(m1_results, m2_results, m1_name, m2_name, x_label='Model'):

    fig = plt.figure(figsize=(10, 5))
    fig.suptitle("Mean Average Precision and Recall", fontsize=20)

    c1 =  '#f8d04e'
    c2 =  '#ef847e'
    gs = gridspec.GridSpec(2, 2, height_ratios=[1,1], width_ratios=[1,3], hspace=0.1, wspace=0.2)

    # mAP & mAR - summary
    x1,x2 = 1, 3
    y_offset = 10
    t_size = 16
    ax1_1 = fig.add_subplot(gs[0, 0])
    y1 = m1_results['AP']['mean'] * 100
    y2 = m2_results['AP']['mean'] * 100

    ax1_1.bar(x1, y1, width=1, color=c1, linewidth=0.5, edgecolor='black', label=m1_name)
    ax1_1.bar(x2, y2, width=1, color=c2, linewidth=0.5, edgecolor='black', label=m2_name)
    ax1_1.text(x1, y1 + y_offset,  f"{y1:.1f}", ha='center', va='center', fontsize=t_size, color='black')
    ax1_1.text(x2, y2 + y_offset,  f"{y2:.1f}", ha='center', va='center', fontsize=t_size, color='black')

    ax2_1 = fig.add_subplot(gs[1, 0])
    y1 = m1_results['AR100']['mean'] * 100
    y2 = m2_results['AR100']['mean'] * 100
    ax2_1.bar(x1, y1, width=1, color=c1, linewidth=0.5, edgecolor='black', label=m1_name)
    ax2_1.bar(x2, y2, width=1, color=c2, linewidth=0.5, edgecolor='black', label=m1_name)
    ax2_1.text(x1, y1 + y_offset,  f"{y1:.1f}", ha='center', va='center', fontsize=t_size, color='black')
    ax2_1.text(x2, y2 + y_offset,  f"{y2:.1f}", ha='center', va='center', fontsize=t_size, color='black')

    # mAP & mAR - per category
    x = list(range(80))
    ax1_2 = fig.add_subplot(gs[0, 1:])
    lw1= 2.0
    lw2= 2.75
    y1 = m1_results['AP']['per_class'] * 100
    y2 = m2_results['AP']['per_class'] * 100
    lines, = ax1_2.plot(x, y1, color=c1, linewidth=lw1)
    lines.set_path_effects([pe.Stroke(linewidth=lw2, foreground='black'), pe.Normal()])
    lines, = ax1_2.plot(x, y2, color=c2, linewidth=lw1)
    lines.set_path_effects([pe.Stroke(linewidth=lw2, foreground='black'), pe.Normal()])

    ax2_2 = fig.add_subplot(gs[1, 1:])
    y1 = m1_results['AR100']['per_class'] * 100
    y2 = m2_results['AR100']['per_class'] * 100
    lines, = ax2_2.plot(x, y1, color=c1, linewidth=lw1)
    lines.set_path_effects([pe.Stroke(linewidth=lw2, foreground='black'), pe.Normal()])
    lines, = ax2_2.plot(x, y2, color=c2, linewidth=lw1)
    lines.set_path_effects([pe.Stroke(linewidth=lw2, foreground='black'), pe.Normal()])

    # Style axis
    y_lim = (0.0, 100.0)
    ax1_1.set_ylim(y_lim)
    ax1_2.set_ylim(y_lim)
    ax2_1.set_ylim(y_lim)
    ax2_2.set_ylim(y_lim)
    ax1_1.set_yticks(ax1_1.get_yticks()[1:]) 
    ax1_2.set_yticks(ax1_2.get_yticks()[1:]) 
    ax2_1.set_yticks(ax2_1.get_yticks()[1:]) 
    ax2_2.set_yticks(ax2_2.get_yticks()[1:]) 

    xlim = (0,4)
    ax1_1.set_xlim(xlim)
    ax2_1.set_xlim(xlim)

    plt.setp(ax1_1.get_xticklabels(), visible=False)
    plt.setp(ax1_2.get_xticklabels(), visible=False)
    ax1_1.set_xticks([0,1,2,3,4])
    ax2_1.set_xticks([0,1,2,3,4])
    ax2_1.set_xticklabels(['', m1_name, '', m2_name,''])
    ax1_1.set_ylabel('$mAP$', fontsize=14)
    ax2_1.set_ylabel('$mAR$', fontsize=14)
    ax2_1.set_xlabel(x_label, fontsize=14)
    ax2_2.set_xlabel('Category', fontsize=14)

    plt.show()
