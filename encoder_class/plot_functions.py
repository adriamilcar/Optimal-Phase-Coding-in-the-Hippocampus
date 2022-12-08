from brian2 import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D

theta_range = [4, 8]


def index_from_tick(value, vector):
    return np.abs(vector - value).argmin()


def plt_fig_1b(input_params, I_min_lock, I_max_lock, x_min, x_max):

    plt.hlines(I_min_lock/1e-12, x_min, x_max, colors='orange', linestyles='dashed', label=r"$I_{min}$")
    plt.hlines(I_max_lock/1e-12, x_min, x_max, colors='orange', linestyles='dashed', label=r"$I_{min}$")
    plt.hlines(90, x_min, x_max, colors='black', linestyles='dashed', linewidth = 0.4, label=r"$I_{min}$")
    plt.hlines(106, x_min, x_max, colors='black', linestyles='dashed', linewidth = 0.4, label=r"$I_{min}$")
    plt.hlines(118, x_min, x_max, colors='black', linestyles='dashed', linewidth = 0.4, label=r"$I_{min}$")
    plt.hlines(130, x_min, x_max, colors='black', linestyles='dashed', linewidth = 0.4, label=r"$I_{min}$")

    plt.xlabel('Time (s)', fontsize=26)
    xticks = np.linspace(0, 1, 5)
    xlabels = [round(xtick, 2) for xtick in xticks]
    plt.xticks(xticks, xlabels, fontsize=24)
    plt.ylabel(r"$I_s$ (pA)", fontsize=26)
    #plt.ylim([input_params["I_min"]/1e-12, (input_params["I_max"] + 2*pA)/1e-12])
    plt.xlim([x_min, x_max])

    yticks = np.arange(60, 162, 20)
    ylabels = [round(ytick) for ytick in yticks]
    plt.yticks(yticks, ylabels, fontsize=24)

    sns.despine()
    plt.tight_layout()
    
    

def plt_fig_1c(input_params):
    plt.vlines(input_params["I_min"], 0, 5*pi/4, colors='orange', linestyles='dashed', zorder=1)
    plt.vlines(input_params["I_max"], 0, 5*pi/4, colors='orange', linestyles='dashed', zorder=1)

    plt.xlabel(r"$I_s$ (pA)", fontsize=26)
    plt.ylabel(r"$\phi$ (rad)", fontsize=26)
    #plt.xlim([0.5e-10, 1.6e-10])

    yticks = [0, pi/4, pi/2, 3*pi/4, pi] 
    yticks_dic = {0 : r"$0$", pi/4 : r"$\pi/4$", pi/2 : r"$\pi/2$", 3*pi/4 :r"$3\pi/4$", pi : r"$\pi$"}
    ylabels = [yticks_dic[ytick] for ytick in yticks]
    plt.yticks(yticks, ylabels, fontsize=24)

    xticks = np.arange(0.6e-10, 1.8e-10, 0.2e-10)
    xlabels = [round(xtick/1e-12) for xtick in xticks]
    plt.xticks(xticks, xlabels, fontsize=24)
    plt.xlim([60e-12, 160e-12])

    plt.vlines(90*pA,  0, 5*pi/4,  colors='black', linestyles='dashed', linewidth = 0.4)
    plt.vlines(106*pA,  0, 5*pi/4,  colors='black', linestyles='dashed', linewidth = 0.4)
    plt.vlines(118*pA,  0, 5*pi/4,  colors='black', linestyles='dashed', linewidth = 0.4)
    plt.vlines(130*pA,  0, 5*pi/4,  colors='black', linestyles='dashed', linewidth = 0.4)

    lgnd = plt.legend(frameon=False, bbox_to_anchor=(1.05, 0.95), fontsize=20, markerscale=3)
    sns.despine()
    
    plt.tight_layout()
    
    
    
def plt_fig_2b(start_time, runtime, T, num_oscillations, ax1, ax2):
    
    ax1.axhline(15, color='black', linestyle='dashed')
    ax1.text(-0.05, 17, r"$V_{th}$", fontsize=18)  
    ax1.set_ylabel(r"$V$ (mV)", fontsize=20)
    ax1.set_xlabel('Time (s)', fontsize=20)
    ax1.set_xlim([None, (runtime - start_time)/second])
    ax1.set_ylim([None, 20])
    xticks = np.arange(0, (num_oscillations + 1)*T, T)
    ax1.set_xticks(xticks)
    T_labels = [str(i) + r"$T$" for i in range(num_oscillations + 1)]
    T_labels[0] = r"$0$"
    T_labels[1] = r"$T$"
    ax1.set_xticklabels(T_labels)
    ax1.tick_params(labelsize=18)
    
    xticks = np.arange(0, (num_oscillations + 1)*T, T/4) - start_time/second
    ax2.set_xticks(xticks)
    pi_labels = [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/4$"]*(num_oscillations + 1)
    ax2.set_xticklabels(pi_labels)
    ax2.tick_params(labelsize=18)
    ax2.set_xlabel(r"$\phi$ (rad)", fontsize=20)
    ax2.axvline(0, ymax=1.6, color='black',  alpha=0.3, linestyle="dashed", clip_on=False) 
    ax2.set_xlim([None, (runtime - start_time)/second])
    ax2.text(0.01, -20, r"$\phi_0$", fontsize=20)
    ax2.set_ylabel(r"$I_\theta$ (pA)", fontsize=20)
    sns.despine()
    
    
def plt_fig_3a(c, M, fig, axs):
    xticks = [0, pi/2, pi, 3*pi/2] 
    xticks_dic = {0 : r"$0$", pi/2 : r"$\pi/2$", pi : r"$\pi$", 3*pi/2 :r"$3\pi/2$"}
    xlabels = [xticks_dic[xtick] for xtick in xticks]
    yticks = [0, 1, 2, 3, 4, 5]

    for index, ax in enumerate(fig.get_axes()):
        if index not in (12, 13, 14 ,15):
            setp(ax.get_xticklabels(), visible=False)

        else:
            ax.set_xticks(ticks=xticks)
            ax.set_xticklabels(labels=xlabels, fontsize=20)

    for index, ax in enumerate(fig.get_axes()):
        if index not in (0, 4, 8, 12):
            setp(ax.get_yticklabels(), visible=False)

        else:
            ax.set_yticks(ticks=yticks)
            ax.set_yticklabels(yticks, fontsize=20)
    s_labels = []        
    for s in range(1, M + 1):
        s_labels.append(r"$s = $" + str(s))
    lines = [Line2D([0], [0], color=c[0], lw=8),
               Line2D([0], [0], color=c[1], lw=8),
               Line2D([0], [0], color=c[2], lw=8),
               Line2D([0], [0], color=c[3], lw=8),
               Line2D([0], [0], color=c[4], lw=8)]
    legend = axs[0, 0].legend(lines, s_labels, frameon=False, fontsize=16)

    fig.supxlabel(r"$\phi$ (rad)", fontsize=30)
    fig.supylabel("p.d.f.", fontsize=30)
    sns.despine()
    fig.tight_layout()
    plt.show()
    
    
def plt_fig_3b():
    plt.legend(frameon=False, fontsize=20)
    plt.xticks([0, 10, 20, 30, 40, 50], fontsize=20)
    plt.yticks([0, 1, 2, 3, 4], fontsize=20)
    plt.xlabel(r"$f$ (Hz)", fontsize=20)
    plt.ylabel(r"$|\phi'(I_s)|$ (pA$^{-1}$)", fontsize=20)
    sns.despine()
    plt.tight_layout()
    
    
def plt_fig_3c():
    plt.legend(frameon=False)
    plt.xlabel(r"$f$ (Hz)", fontsize=20)
    plt.xticks([0, 10, 20, 30, 40, 50], fontsize=20)
    plt.yticks([0, 1, 2, 3], fontsize=20)
    plt.ylabel(r"$\langle\sigma_{\phi}\rangle_s$", fontsize=20)
    plt.legend(fontsize=12, frameon=False, loc='upper left')
    sns.despine()
    plt.tight_layout()
    
    
def plt_fig_3d(stoch_sigmas):
    plt.yticks([0, 0.5, 1, 1.5], fontsize=20)
    plt.legend(fontsize=12, frameon=False)
    xticklist = np.linspace(stoch_sigmas[0], stoch_sigmas[-1], 4).round(2)
    xticklist[0] = stoch_sigmas[0].round(3)
    plt.xticks(np.linspace(stoch_sigmas[0], stoch_sigmas[-1], 4), xticklist, fontsize=20)
    plt.xlabel(r"$\sigma_W$ (V·s$^{-1/2}$)", fontsize=20)
    plt.ylabel(r"$\langle\sigma_{\phi}\rangle_s$", fontsize=20)
    sns.despine()
    plt.tight_layout()
    

def plt_fig_3e(fs, fig, ax1):
    ax1_color = 'black'
    f_ticks = [1, 10, 20, 30, 40, 50]
    ax1.set_xticks(f_ticks)
    ax1.set_yticks(np.linspace(0, 2, 4).round(0))
    ax1.set_xlabel(r"$f$ (Hz)", fontsize=20)
    ax1.set_ylabel(r"$I_m$ (bit/cycle)", color=ax1_color, fontsize=20)
    ax1.tick_params(axis='y', labelcolor=ax1_color, labelsize=20)
    ax1.tick_params(axis='x', labelsize=20)

    lgnd = fig.legend(frameon=False, loc="upper right", fontsize=12)
    plt.setp(lgnd.get_title(),fontsize='large')
    fig.tight_layout()
    sns.despine()
    
    
def plt_fig_3f(stoch_sigmas):
    
    plt.xlabel(r"$\sigma_W$ (V·s$^{-1/2}$)", fontsize=22)
    xticklist = np.linspace(stoch_sigmas[0], stoch_sigmas[-1], 4).round(2)
    xticklist[0] = stoch_sigmas[0].round(3)
    plt.xticks(np.linspace(stoch_sigmas[0], stoch_sigmas[-1], 4), xticklist, fontsize=20)

    plt.yticks(np.linspace(0, 4, 5), fontsize=20)
    plt.ylabel(r"$I_m$ (bit/cycle)", fontsize=22)
    plt.axhline(-np.log2(1/10), linestyle='--')
    plt.text(x=0.001, y=3.6, s=r"$-\log{({\frac{1}{M})}}$", fontsize=18, color='black')
    lgnd = plt.legend(frameon=False, fontsize=12)
    plt.setp(lgnd.get_title(),fontsize='large')

    plt.tight_layout()
    sns.despine()
    
    

def plt_fig_3g(fs, stoch_sigmas):
    ticks = np.linspace(0, 2, 5, endpoint=True).round(1)
    cbar = plt.colorbar(shrink=.67, ticks=ticks)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(r'$I_m$ (bit)', fontsize=22)
    cbar.ax.set_yticklabels(['0', '0.5', '>1'])

    plt.xlabel('$f$ (Hz)', fontsize=22)
    plt.ylabel(r"$\sigma_W$ (V·s$^{-1/2}$)", fontsize=22)
    
    f_ticks = [1, 10, 20, 30, 40, 50]
    f_index = list(map(lambda value: index_from_tick(value, fs/Hz), f_ticks))
    plt.xticks(f_index, f_ticks, fontsize=20)
    
    s_ticks = [0., 0.01, 0.02, 0.03, 0.04]
    s_index = list(map(lambda value: index_from_tick(value, stoch_sigmas), s_ticks))
    plt.yticks(s_index, s_ticks, fontsize=20)
    
    plt.tight_layout()
    
    
def plt_fig_4b():
    
    f_ticks = [1, 10, 20, 30, 40, 50]
    plt.xticks(f_ticks, fontsize=20)
    plt.xlabel(r"$f$ (Hz)", fontsize=20)

    plt.yticks([0, 5, 10, 15, 20, 25], fontsize=20)
    plt.ylabel(r"$R$ (bit/s)", fontsize=20)
    plt.legend(frameon=False)

    plt.tight_layout()
    sns.despine()
    

def plt_fig_4c(fs, stoch_sigmas):
    ticks = np.linspace(0, 6, 4, endpoint=True).round(1)
    cbar = plt.colorbar(shrink=.67, ticks=ticks)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(r'$R$ (bit/s)', fontsize=22)
    cbar.ax.set_yticklabels(['0', '2', '4', '>6'])

    plt.xlabel('$f$ (Hz)', fontsize=22)
    plt.ylabel(r"$\sigma_W$ (V·s$^{-1/2}$)", fontsize=22)
    
    f_ticks = [1, 10, 20, 30, 40, 50]
    f_index = list(map(lambda value: index_from_tick(value, fs/Hz), f_ticks))
    plt.xticks(f_index, f_ticks, fontsize=20)
    
    s_ticks = [0., 0.01, 0.02, 0.03, 0.04]
    s_index = list(map(lambda value: index_from_tick(value, stoch_sigmas), s_ticks))
    plt.yticks(s_index, s_ticks, fontsize=20)
    
    theta_index = list(map(lambda value: index_from_tick(value, fs/Hz), theta_range))
    plt.axvline(x=theta_index[0], linestyle='--', linewidth=3, color='white')
    plt.axvline(x=theta_index[1], linestyle='--', linewidth=3, color='white')
    th_char_index = (theta_range[1] + theta_range[0])//2 + 2
    plt.text(x=th_char_index, y=90, s='θ', fontsize=18, color='white')

    plt.tight_layout()
    
    return


def plt_fig_4d(fs, stoch_sigmas):
    ticks = np.linspace(0, 1, 2, endpoint=True).round(1)
    cbar = plt.colorbar(shrink=.683, ticks=ticks)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(r'Norm. $R$', fontsize=22)

    plt.xlabel('$f$ (Hz)', fontsize=22)
    plt.ylabel(r"$\sigma_W$ (V·s$^{-1/2}$)", fontsize=22)
    
    f_ticks = [1, 10, 20, 30, 40, 50]
    f_index = list(map(lambda value: index_from_tick(value, fs/Hz), f_ticks))
    plt.xticks(f_index, f_ticks, fontsize=20)
    
    s_ticks = [0., 0.01, 0.02, 0.03, 0.04]
    s_index = list(map(lambda value: index_from_tick(value, stoch_sigmas), s_ticks))
    plt.yticks(s_index, s_ticks, fontsize=20)
    
    theta_index = list(map(lambda value: index_from_tick(value, fs/Hz), theta_range))
    plt.axvline(x=theta_index[0], linestyle='--', linewidth=3, color='black')
    plt.axvline(x=theta_index[1], linestyle='--', linewidth=3, color='black')
    
    th_char_index = (theta_range[1] + theta_range[0])//2 + 2
    plt.text(x=th_char_index, y=90, s='θ', fontsize=18, color='black')

    plt.tight_layout()
    
    
def plt_fig_5b(fs, DV_gradient):
    ticks = np.linspace(0, 1, 3, endpoint=True).round(1)
    cbar = plt.colorbar(ticks= ticks, shrink=.71)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(r'$R$ (bit/s)', fontsize=22)
    plt.xlabel(r'$f$ (Hz)', fontsize=22)
    plt.ylabel('D-V gradient', fontsize=22)

    f_ticks = [1, 10, 20, 30, 40, 50]
    f_index = list(map(lambda value: index_from_tick(value, fs/Hz), f_ticks))
    plt.xticks(f_index, f_ticks, fontsize=20)
    
    dv_ticks = [0, 20, 40, 60, 80, 100]
    dv_index = list(map(lambda value: index_from_tick(value, DV_gradient), dv_ticks))
    plt.yticks(dv_index, dv_ticks, fontsize=20)
    
    theta_index = list(map(lambda value: index_from_tick(value, fs/Hz), theta_range))
    plt.axvline(x=theta_index[0], linestyle='--', linewidth=3, color='black')
    plt.axvline(x=theta_index[1], linestyle='--', linewidth=3, color='black')
    th_char_index = (theta_range[1] + theta_range[0])//2 + 2
    plt.text(x=th_char_index, y=90, s='θ', fontsize=18, color='black')
    
    plt.tight_layout()
def plt_fig_5c(fs, DV_gradient):
    
    ticks = np.linspace(0, 1, 2, endpoint=True).round(1)
    cbar = plt.colorbar(ticks= ticks, shrink=.71)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(r'Norm. $R$', fontsize=22)
    plt.xlabel(r'$f$ (Hz)', fontsize=22)
    plt.ylabel('D-V gradient', fontsize=22)

    f_ticks = [1, 10, 20, 30, 40, 50]
    f_index = list(map(lambda value: index_from_tick(value, fs/Hz), f_ticks))
    plt.xticks(f_index, f_ticks, fontsize=20)
    
    dv_ticks = [0, 20, 40, 60, 80, 100]
    dv_index = list(map(lambda value: index_from_tick(value, DV_gradient), dv_ticks))
    plt.yticks(dv_index, dv_ticks, fontsize=20)
    
    theta_index = list(map(lambda value: index_from_tick(value, fs/Hz), theta_range))
    plt.axvline(x=theta_index[0], linestyle='--', linewidth=3, color='black')
    plt.axvline(x=theta_index[1], linestyle='--', linewidth=3, color='black')
    th_char_index = (theta_range[1] + theta_range[0])//2 + 2
    plt.text(x=th_char_index, y=90, s='θ', fontsize=18, color='black')
    plt.tight_layout()
    
def plt_fig_6b():
    f_ticks = [1, 15, 35, 50]
    plt.xticks(f_ticks, fontsize=20)
    plt.xlabel(r"$f$ (Hz)", fontsize=20)

    plt.yticks(np.linspace(0,10, 3), fontsize=20)
    plt.ylabel(r"$R$ (bit/s)", fontsize=20)
    plt.legend(frameon=False)

    plt.tight_layout()
    sns.despine()

def plt_fig_6c(fs, Ioscs):
    ticks = np.linspace(0, 10, 3, endpoint=True).round(1)
    cbar = plt.colorbar(shrink=.67, ticks=ticks)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(r'$R$ (bit/s)', fontsize=22)

    plt.xlabel('$f$ (Hz)', fontsize=22)
    plt.ylabel(r"$I_{osc}$ (pA)", fontsize=22)

    f_ticks = [1, 10, 20, 30, 40, 50]
    f_index = list(map(lambda value: index_from_tick(value, fs/Hz), f_ticks))
    plt.xticks(f_index, f_ticks, fontsize=20)
    
    Iosc_ticks = [30, 40, 50, 60, 70, 80]
    Iosc_index = list(map(lambda value: index_from_tick(value, Ioscs/pA), Iosc_ticks))
    plt.yticks(Iosc_index, Iosc_ticks, fontsize=20)

    theta_index = list(map(lambda value: index_from_tick(value, fs/Hz), theta_range))
    plt.axvline(x=theta_index[0], linestyle='--', linewidth=3, color='black')
    plt.axvline(x=theta_index[1], linestyle='--', linewidth=3, color='black')
    th_char_index = (theta_range[1] + theta_range[0])//2 + 2
    plt.text(x=th_char_index, y=90, s='θ', fontsize=18, color='black')

    plt.tight_layout()
    return
    
def plt_fig_6d(fs, Ioscs):
    ticks = np.linspace(0, 1, 2, endpoint=True).round(1)
    cbar = plt.colorbar(shrink=.683, ticks=ticks)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(r'Norm. $R$', fontsize=22)

    plt.xlabel('$f$ (Hz)', fontsize=22)
    plt.ylabel(r"$I_{osc}$ (pA)", fontsize=22)
    f_ticks = [1, 10, 20, 30, 40, 50]
    f_index = list(map(lambda value: index_from_tick(value, fs/Hz), f_ticks))
    plt.xticks(f_index, f_ticks, fontsize=20)
    plt.xticks(f_index, f_ticks, fontsize=20)
    
    Iosc_ticks = [30, 40, 50, 60, 70, 80]
    Iosc_index = list(map(lambda value: index_from_tick(value, Ioscs/pA), Iosc_ticks))
    plt.yticks(Iosc_index, Iosc_ticks, fontsize=20)
    
    theta_index = list(map(lambda value: index_from_tick(value, fs/Hz), theta_range))
    plt.axvline(x=theta_index[0], linestyle='--', linewidth=3, color='black')
    plt.axvline(x=theta_index[1], linestyle='--', linewidth=3, color='black')
    th_char_index = (theta_range[1] + theta_range[0])//2 + 2 
    plt.text(x=th_char_index, y=90, s='θ', fontsize=18, color='black')
    
    plt.tight_layout()
    
def plt_fig_7a(c, M, fig, axs):
    s_labels = []
    for s in range(1, M + 1):
        s_labels.append(r"$s = $" + str(s))
    lines = [Line2D([0], [0], color=c[0], lw=8),
               Line2D([0], [0], color=c[1], lw=8),
               Line2D([0], [0], color=c[2], lw=8),
               Line2D([0], [0], color=c[3], lw=8),
               Line2D([0], [0], color=c[4], lw=8)]
    legend = axs[0, 1].legend(lines, s_labels, loc=2, frameon=False, fontsize=16)
    axs[-1, -1].axis('off')
    fig.supylabel("p.d.f.", fontsize=30)
    sns.despine()
    fig.tight_layout()
    plt.show()
    
    
def plt_fig_7b():
    #noise_label = r"$\sigma_W = $" + str(round(stoch_sigmas[noise_level], 3))[:5] + r" V·s$^{-1/2}$"
    #plt.title(noise_label, fontsize=20)
    s_labels = [r"$I_s$", r"$\langle I_{eff}\rangle$"]
    lines = [Line2D([0], [0], color='black', lw=4),
             Line2D([0], [0], color='black', lw=1)]
    legend = plt.legend(lines, s_labels, loc=2, frameon=False, fontsize=20)
    plt.xlabel(r"$f$ (Hz)", fontsize=20)
    plt.ylabel(r"$I$ (pA)", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks([80, 120, 160, 200], fontsize=20)
    sns.despine()
    plt.tight_layout()
    

def plt_fig_8a():
    plt.xticks([0, 10, 20, 30, 40, 50], fontsize=20)
    plt.yticks([0, 5, 10, 15, 20], fontsize=20)
    plt.legend(fontsize=12, frameon=False)
    plt.xlabel(r"$f$ (Hz)", fontsize=20)
    plt.ylabel(r"$\sigma_I$", fontsize=20)
    sns.despine()
    plt.tight_layout()
    
    
def plt_fig_8b(stoch_sigmas):
    plt.yticks([0, 5, 10, 15, 20], fontsize=20)
    plt.legend(fontsize=12, frameon=False)
    xticklist = np.linspace(stoch_sigmas[0], stoch_sigmas[-1], 4).round(2)
    xticklist[0] = stoch_sigmas[0].round(3)
    plt.xticks(np.linspace(stoch_sigmas[0], stoch_sigmas[-1], 4), xticklist, fontsize=20)
    plt.xlabel(r"$\sigma_W$ (V·s$^{-1/2}$)", fontsize=20)
    plt.ylabel(r"$\sigma_I$", fontsize=20)
    sns.despine()
    plt.tight_layout()
    

def plt_fig_8c():
    plt.xticks([0, 10, 20, 30 , 40, 50], fontsize=20)
    plt.yticks([0, 2, 4, 6 , 8, 10], fontsize=20)
    plt.ylabel(r"$\Delta$ spikes/cycle", fontsize=20)
    plt.xlabel("$f$ (Hz)", fontsize=20)
    plt.legend(frameon=False, fontsize=16)
    sns.despine()
    plt.tight_layout()
    
    

    
    
