from general_utilities import mean_std
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns


color_list = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
# trovata al link https://gist.github.com/thriveth/8560036

# color_list = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']
# Tableau colorblind10 palette trovata al link https://viscid-hub.github.io/Viscid-docs/docs/dev/styles/tableau-colorblind10.html

col_mrc = color_list[0]
col_nata = color_list[1]
col_clab = color_list[2]
col_cleansed = color_list[3]

linestyle_list = ['dotted', 'dashed', 'dashdot', 'solid']

style_clean = linestyle_list[0]
style_nocorr = linestyle_list[1]
style_corr = linestyle_list[3]
style_cleansed = linestyle_list[2]

style_err = linestyle_list[0]
style_est = linestyle_list[3]


def my_plot(x, y, std, color, style, label, shade=True):

    plt.plot(x, y, color=color, label=label, linestyle=style)
    if shade:
        plt.fill_between(x, y - std, y + std, color=color, alpha=0.08)


def plot_ntrain_all_methods(nvector, data_mrc, data_lr, data_nata, data_cl,
                            plot_dir='', title='', figure_name='', shade=True, format=''):

    e1, std_e1 = mean_std(data_mrc.mrc_clean['errors'])
    e2, std_e2 = mean_std(data_mrc.mrc_nocorr['errors'])
    e3, std_e3 = mean_std(data_mrc.mrc_back['errors'])

    my_plot(nvector, e1, std_e1, color=col_mrc, style=style_clean, label='Oracle MRC', shade=shade)
    my_plot(nvector, e2, std_e2, color=col_mrc, style=style_nocorr, label='Naive MRC', shade=shade)
    my_plot(nvector, e3, std_e3, color=col_mrc, style=style_corr, label='Noisy MRC', shade=shade)

    plot_dir_errors = plot_dir + '/ntrain_mrc'

    if data_lr is not None:
        elr1, std_elr1 = mean_std(data_lr.lr_clean['errors'])
        elr2, std_elr2 = mean_std(data_lr.lr_nocorr['errors'])
        my_plot(nvector, elr1, std_elr1, color=col_nata, style=style_clean, label='Oracle LR', shade=shade)
        my_plot(nvector, elr2, std_elr2, color=col_nata, style=style_nocorr, label='Naive LR', shade=shade)

        plot_dir_errors = plot_dir_errors + '_lr'

    if data_nata is not None:
        enata, std_enata = mean_std(data_nata.natarajan['errors'])
        my_plot(nvector, enata, std_enata, color=col_nata, style=style_corr, label='Noisy LR', shade=shade)

        plot_dir_errors = plot_dir_errors + '_nata'

    if data_cl is not None:
        ecl, std_ecl = mean_std(data_cl.cl['errors'])
        my_plot(nvector, ecl, std_ecl, color=col_clab, style=style_corr, label='CleanLearning', shade=shade)

        plot_dir_errors = plot_dir_errors + '_cl'

    plt.legend(loc='upper right')
    plt.xlabel('Training size')
    plt.ylabel('Classification error')
    plt.title(title)

    plot_dir_errors = plot_dir_errors + '/'
    os.makedirs(plot_dir_errors, exist_ok=True)
    plt.savefig(plot_dir_errors + figure_name, format=format, bbox_inches='tight')
    plt.close()


def plot_ntrain_corrected(nvector, data_mrc, data_nata, data_cl,
                          plot_dir='', title='', figure_name='', shade=True, format=''):

    plot_dir_errors = plot_dir + '/ntrain_corrected'

    emrc, std_emrc = mean_std(data_mrc.mrc_back['errors'])
    enata, std_enata = mean_std(data_nata.natarajan['errors'])
    ecl, std_ecl = mean_std(data_cl.cl['errors'])

    my_plot(nvector, emrc, std_emrc, color=col_mrc, style=style_corr, label='Noisy MRC', shade=shade)
    my_plot(nvector, enata, std_enata, color=col_nata, style=style_corr, label='Noisy LR', shade=shade)
    my_plot(nvector, ecl, std_ecl, color=col_clab, style=style_corr, label='CleanLearning', shade=shade)

    plt.legend()
    plt.xlabel('Training size')
    plt.ylabel('Classification error')
    plt.title(title)

    plot_dir_errors = plot_dir_errors + '/'
    os.makedirs(plot_dir_errors, exist_ok=True)
    plt.savefig(plot_dir_errors + figure_name, format=format, bbox_inches='tight')
    plt.close()


def plot_ntrain_all_mrcs(nvector, data_mrc, data_mrc_est,
                         plot_dir='', title='', figure_name='', shade=True, format=''):

    e1, std_e1 = mean_std(data_mrc.mrc_clean['errors'])
    e2, std_e2 = mean_std(data_mrc.mrc_nocorr['errors'])
    e3, std_e3 = mean_std(data_mrc.mrc_back['errors'])
    e4, std_e4 = mean_std(data_mrc_est.mrc_back_est['errors'])

    my_plot(nvector, e1, std_e1, color=col_mrc, style=style_corr, label='Oracle MRC', shade=shade)
    my_plot(nvector, e2, std_e2, color=col_nata, style=style_corr, label='Naive MRC', shade=shade)
    my_plot(nvector, e3, std_e3, color=col_clab, style=style_corr, label='Noisy MRC', shade=shade)
    my_plot(nvector, e4, std_e4, color=col_cleansed, style=style_corr, label='Noisy MRC (T est)', shade=shade)

    plot_dir_errors = plot_dir + '/ntrain_mrc_est'

    plt.legend()
    plt.xlabel('Training size')
    plt.ylabel('Classification error')
    plt.title(title)

    plot_dir_errors = plot_dir_errors + '/'
    os.makedirs(plot_dir_errors, exist_ok=True)
    plt.savefig(plot_dir_errors + figure_name, format=format, bbox_inches='tight')
    plt.close()


def plot_ntrain_cleansed(nvector, data_mrc, data_mrc_cleansed, data_nata, data_lr_cleansed,
                         plot_dir='', title='', figure_name='', shade=True, format=''):

    e1, std_e1 = mean_std(data_mrc.mrc_back['errors'])
    e2, std_e2 = mean_std(data_mrc_cleansed.mrc_cleansed['errors'])
    enata, std_enata = mean_std(data_nata.natarajan['errors'])
    elr, std_elr = mean_std(data_lr_cleansed.lr_cleansed['errors'])

    my_plot(nvector, e1, std_e1, color=col_mrc, style=style_corr, label='Noisy MRC', shade=shade)
    my_plot(nvector, e2, std_e2, color=col_mrc, style=style_cleansed, label='Cleansed MRC', shade=shade)
    my_plot(nvector, enata, std_enata, color=col_nata, style=style_corr, label='Noisy LR', shade=shade)
    my_plot(nvector, elr, std_elr, color=col_nata, style=style_cleansed, label='Cleansed LR', shade=shade)

    plot_dir_errors = plot_dir + '/ntrain_cleansed'

    plt.legend()
    plt.xlabel('Training size')
    plt.ylabel('Classification error')
    plt.title(title)

    plot_dir_errors = plot_dir_errors + '/'
    os.makedirs(plot_dir_errors, exist_ok=True)
    plt.savefig(plot_dir_errors + figure_name, format=format, bbox_inches='tight')
    plt.close()


def plot_perfeval(nvector, data_mrc, data_nata,
                  plot_dir='', title='', figure_name='', shade=True, format=''):

    plot_dir_perfeval = plot_dir + '/perfeval_mrc'

    emrc, std_emrc = mean_std(data_mrc.mrc_back['errors'])
    ebound, std_ebound = mean_std(data_mrc.mrc_back['bounds'])

    my_plot(nvector, emrc, std_emrc, color=col_mrc, label='MRC corrected' + '(Error)', style=style_err, shade=shade)
    my_plot(nvector, ebound, std_ebound, color=col_mrc, label='MRC corrected' + '(BOUND)', style=style_est, shade=shade)

    if data_nata is not None:
        enata, std_enata = mean_std(data_nata.natarajan['errors'])
        eule, std_eule = mean_std(data_nata.natarajan['unbiased_loss'])

        my_plot(nvector, enata, std_enata, color=col_nata, label='LR corrected' + '(Error)', style=style_err,
                shade=shade)
        my_plot(nvector, eule, std_eule, color=col_nata, label='LR corrected' + '(ULE)', style=style_est, shade=shade)

        plot_dir_perfeval = plot_dir_perfeval + '_nata'

    plt.legend(loc='upper right')
    plt.xlabel('Training size')
    plt.ylabel('Error Measures')
    plt.title(title)
    plot_dir_perfeval = plot_dir_perfeval + '/'
    os.makedirs(plot_dir_perfeval, exist_ok=True)
    plt.savefig(plot_dir_perfeval + figure_name, format=format, bbox_inches='tight')
    plt.close()


def plot_perfeval_cleansed(nvector, data_mrc, data_nata, data_cleansed, data_clab,
                  plot_dir='', title='', figure_name='', shade=True, format=''):

    plot_dir_perfeval = plot_dir + '/perfeval_mrc'

    emrc, std_emrc = mean_std(data_mrc.mrc_back['errors'])
    ebound, std_ebound = mean_std(data_mrc.mrc_back['bounds'])

    my_plot(nvector, emrc, std_emrc, color=col_mrc, label='Noisy MRC' + ' (Error)', style=style_err, shade=shade)
    my_plot(nvector, ebound, std_ebound, color=col_mrc, label='Noisy MRC' + ' (MINIMAX)', style=style_est, shade=shade)

    if data_nata is not None:
        enata, std_enata = mean_std(data_nata.natarajan['errors'])
        eule, std_eule = mean_std(data_nata.natarajan['unbiased_loss'])

        my_plot(nvector, enata, std_enata, color=col_nata, label='Noisy LR' + ' (Error)', style=style_err,
                shade=shade)
        my_plot(nvector, eule, std_eule, color=col_nata, label='Noisy LR' + ' (ULE)', style=style_est, shade=shade)

        plot_dir_perfeval = plot_dir_perfeval + '_nata'

    if data_cleansed is not None:

        ecled, std_ecled = mean_std(data_cleansed.lr_cleansed['errors'])
        eble, std_eble = mean_std(data_cleansed.lr_cleansed['biased_loss'])
        my_plot(nvector, ecled, std_ecled, color=col_cleansed, label='Cleansed LR' + ' (Error)', style=style_err,
                shade=shade)
        my_plot(nvector, eble, std_eble, color=col_cleansed, label='Cleansed LR' + ' (BLE)', style=style_est,
                shade=shade)

        plot_dir_perfeval = plot_dir_perfeval + '_cleansed'

    if data_clab is not None:

        eclab, std_eclab = mean_std(data_clab.cl['errors'])
        eble, std_eble = mean_std(data_clab.cl['biased_loss'])
        my_plot(nvector, eclab, std_eclab, color=col_clab, label='CleanLearning' + ' (Error)', style=style_err,
                shade=shade)
        my_plot(nvector, eble, std_eble, color=col_clab, label='CleanLearning' + ' (BLE)', style=style_est,
                shade=shade)

        plot_dir_perfeval = plot_dir_perfeval + '_clab'

    plt.legend()
    plt.xlabel('Training size')
    plt.ylabel('Error Measures')
    plt.title(title)
    plot_dir_perfeval = plot_dir_perfeval + '/'
    os.makedirs(plot_dir_perfeval, exist_ok=True)
    plt.savefig(plot_dir_perfeval + figure_name, format=format, bbox_inches='tight')
    plt.close()


def plot_boxplot_cleansed(data, str_varyingrho='', plot_dir='', title='', figure_name='', format=''):

    """
    BOXPLOT of BLE + lr_cleansed , MINIMAX + mrc , ULE + natarajan

    Do boxplot with different rho values in the x-axis (rho1 or rho2 depending on the parameter str_varyingrho )
    and "grouping" the 3 methods ('lr', 'mrc_back', 'natarajan') for each str_varyingrho

    Parameters
    ----------
    data
    str_varyingrho
    plot_dir
    title
    figure_name
    format

    Returns
    -------

    """
    colors = [col_clab, col_mrc, col_nata]
    bp = sns.boxplot(data=data,
                     y='value',
                     x=str_varyingrho,
                     hue='classifier',
                     hue_order=['lr_cleansed', 'mrc_back', 'natarajan'],
                     width=0.5,
                     linewidth=0.7,
                     boxprops=dict(alpha=.5),
                     palette={"lr_cleansed": col_clab, "mrc_back": col_mrc, "natarajan": col_nata})

    x_values = plt.gca().get_xticks()
    mean_values = data.groupby(['classifier', str_varyingrho])['errors'].median()   # median or mean

    # Change colors and add the line of the errors' median of each method
    # for the different values of str_varyingrho

    g = []
    h = []
    m = []
    for (group, hue), mean in mean_values.items():
        g.append(group)
        h.append(hue)
        m.append(mean)

    g = np.reshape(g, (3, 3))
    h = np.reshape(h, (3, 3))
    m = np.reshape(m, (3, 3))
    shift = np.array([-0.15, 0, 0.15])

    for i in range(g.shape[0]):
        plt.scatter(x_values + shift[i], m[i], s=40, color=colors[i], marker='X')

    box = bp.get_position()

    # Legend down the plot:
    # bp.set_position([box.x0, box.y0 + box.height * 0.1,
    #                  box.width, box.height * 0.9])
    # bp.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #           fancybox=True, shadow=True, ncol=3)

    # Legend above the plot
    # bp.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
    #           ncol=2, fancybox=True, shadow=True)

    # Legend on the side of the plot
    #bp.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    #bp.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    bp.legend(loc='upper right')
    bp.legend_.texts[0].set_text('LE - Cleansed LR')
    bp.legend_.texts[1].set_text('MINIMAX - Noisy MRC')
    bp.legend_.texts[2].set_text('ULE - Noisy LR')
    plt.title(title)

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })

    plt.ylabel('Error probability')
    if str_varyingrho == 'rho2':
        plt.xlabel(r'$\rho_2$')
    else:
        plt.xlabel(r'$\rho_1$')
    #plt.xlim(right=4.5)
    plt.ylim(top=0.48)
    plt.savefig(plot_dir + figure_name, format=format, bbox_inches='tight')
    plt.close()
