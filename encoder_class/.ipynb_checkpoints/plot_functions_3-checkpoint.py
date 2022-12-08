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

    plt.xlabel('Time (s)', fontsize=22)
    xticks = np.linspace(0, 1, 5)
    xlabels = [round(xtick, 2) for xtick in xticks]
    plt.xticks(xticks, xlabels, fontsize=20)
    plt.ylabel(r"$I_s$ (pA)", fontsize=22)
    #plt.ylim([input_params["I_min"]/1e-12, (input_params["I_max"] + 2*pA)/1e-12])
    plt.xlim([x_min, x_max])

    yticks = np.arange(60, 162, 20)
    ylabels = [round(ytick) for ytick in yticks]
    plt.yticks(yticks, ylabels, fontsize=20)

    sns.despine()
    plt.tight_layout()
    
    

def plt_fig_1c(input_params):
    plt.vlines(input_params["I_min"], 0, 5*pi/4, colors='orange', linestyles='dashed', zorder=1)
    plt.vlines(input_params["I_max"], 0, 5*pi/4, colors='orange', linestyles='dashed', zorder=1)

    plt.xlabel(r"$I_s$ (pA)", fontsize=22)
    plt.ylabel(r"$\phi$ (rad)", fontsize=22)
    #plt.xlim([0.5e-10, 1.6e-10])

    yticks = [0, pi/4, pi/2, 3*pi/4, pi] 
    yticks_dic = {0 : r"$0$", pi/4 : r"$\pi/4$", pi/2 : r"$\pi/2$", 3*pi/4 :r"$3\pi/4$", pi : r"$\pi$"}
    ylabels = [yticks_dic[ytick] for ytick in yticks]
    plt.yticks(yticks, ylabels, fontsize=20)

    xticks = np.arange(0.6e-10, 1.8e-10, 0.2e-10)
    xlabels = [round(xtick/1e-12) for xtick in xticks]
    plt.xticks(xticks, xlabels, fontsize=20)

    plt.vlines(90*pA,  0, 5*pi/4,  colors='black', linestyles='dashed', linewidth = 0.4)
    plt.vlines(106*pA,  0, 5*pi/4,  colors='black', linestyles='dashed', linewidth = 0.4)
    plt.vlines(118*pA,  0, 5*pi/4,  colors='black', linestyles='dashed', linewidth = 0.4)
    plt.vlines(130*pA,  0, 5*pi/4,  colors='black', linestyles='dashed', linewidth = 0.4)

    lgnd = plt.legend(frameon=False, bbox_to_anchor=(1.25, 0.95), fontsize=16, markerscale=3)
    sns.despine()
    
    plt.tight_layout()
    

def plt_fig_2b():
    plt.xlabel(r"$I_s$ (pA)", fontsize=22)
    plt.xlim([0.5e-10, 1.6e-10])
    xticks = np.arange(0.6e-10, 1.8e-10, 0.2e-10)
    xlabels = [round(xtick/1e-12) for xtick in xticks]
    plt.xticks(xticks, xlabels, fontsize=20)

    yticks = [0, pi/2, pi, 3*pi/2] 
    yticks_dic = {0 : r"$0$", pi/2 : r"$\pi/2$", pi : r"$\pi$", 3*pi/2 :r"$3\pi/2$"}
    ylabels = [yticks_dic[ytick] for ytick in yticks]
    plt.yticks(yticks, ylabels, fontsize=20)
    plt.ylabel(r"$\phi$ (rad)", fontsize=22)   
    
    #plt.legend(frameon=False, bbox_to_anchor=(1.2, 0.95), fontsize=16)
    lgnd = plt.legend(title=r"$f$ (Hz)", frameon=False, fontsize=12)
    plt.setp(lgnd.get_title(),fontsize='large')

    sns.despine()
    plt.tight_layout()
    
    
def plt_fig_2c(fs, fig, ax1):
    
    ax1_color = 'black'
    #f_ticks = np.linspace(1, 50, 3).round(0)
    #f_ticks[0] = 1
    f_ticks = [1, 10, 20, 30, 40, 50]
    ax1.set_xticks(f_ticks)
    ax1.set_xlabel(r"$f$ (Hz)", fontsize=20)
    ax1.set_ylabel(r"$I_m$ (bit/cycle)", color=ax1_color, fontsize=20)
    ax1.tick_params(axis='y', labelcolor=ax1_color, labelsize=20)
    ax1.tick_params(axis='x', labelsize=20)

    lgnd = fig.legend(frameon=False, loc="upper right", fontsize=12)
    plt.setp(lgnd.get_title(),fontsize='large')
    fig.tight_layout()
    sns.despine()
    
def plt_fig_2d(stoch_sigmas):
    
    plt.xlabel(r"$\sigma_W$ (V·s$^{-1/2}$)", fontsize=22)
    xticklist = np.linspace(stoch_sigmas[0], stoch_sigmas[-1], 4).round(2)
    xticklist[0] = stoch_sigmas[0].round(3)
    plt.xticks(np.linspace(stoch_sigmas[0], stoch_sigmas[-1], 4), xticklist, fontsize=20)

    plt.yticks(np.linspace(0, 2, 3), fontsize=20)
    plt.ylabel(r"$I_m$ (bit/cycle)", fontsize=22)
    lgnd = plt.legend(title=r"$f$ (Hz)", frameon=False, fontsize=12)
    plt.setp(lgnd.get_title(),fontsize='large')

    plt.tight_layout()
    sns.despine()
    
    
def plt_fig_2e(stoch_sigmas):
    
    plt.xlabel(r"$\sigma_W$ (V·s$^{-1/2}$)", fontsize=22)
    xticklist = np.linspace(stoch_sigmas[0], stoch_sigmas[-1], 4).round(2)
    xticklist[0] = stoch_sigmas[0].round(3)
    plt.xticks(np.linspace(stoch_sigmas[0], stoch_sigmas[-1], 4), xticklist, fontsize=20)

    plt.yticks(np.linspace(0, 2, 3), fontsize=20)
    plt.ylabel(r"$I_m$ (bit/cycle)", fontsize=22)
    lgnd = plt.legend(frameon=False, fontsize=12)
    plt.setp(lgnd.get_title(),fontsize='large')

    plt.tight_layout()
    sns.despine()
    
def plt_fig_2f(fs, stoch_sigmas):
    ticks = np.linspace(0, 2, 5, endpoint=True).round(1)
    cbar = plt.colorbar(shrink=.67, ticks=ticks)
    #cbar = plt.colorbar(shrink=.67)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(r'$I_m$ (bit)', fontsize=22)

    plt.xlabel('$f$ (Hz)', fontsize=22)
    plt.ylabel(r"$\sigma_W$ (V·s$^{-1/2}$)", fontsize=22)
    
    f_ticks = [1, 10, 20, 30, 40, 50]
    f_index = list(map(lambda value: index_from_tick(value, fs/Hz), f_ticks))
    plt.xticks(f_index, f_ticks, fontsize=20)
    
    s_ticks = [0.005, 0.01, 0.02, 0.03, 0.04]
    s_index = list(map(lambda value: index_from_tick(value, stoch_sigmas), s_ticks))
    plt.yticks(s_index, s_ticks, fontsize=20)
    

    theta_index = list(map(lambda value: index_from_tick(value, fs/Hz), theta_range))
    plt.axvline(x=theta_index[0], linestyle='--', linewidth=3, color='white')
    plt.axvline(x=theta_index[1], linestyle='--', linewidth=3, color='white')
    th_char_index = (theta_range[1] + theta_range[0])//2 + 2
    plt.text(x=th_char_index, y=90, s='θ', fontsize=28, color='white')

    plt.tight_layout()


    
def plt_fig_3b():
    
    f_ticks = [1, 10, 20, 30, 40, 50]
    plt.xticks(f_ticks, fontsize=20)
    plt.xlabel(r"$f$ (Hz)", fontsize=20)

    plt.yticks([0, 5, 10, 15], fontsize=20)
    plt.ylabel(r"$R$ (bit/s)", fontsize=20)
    plt.legend(frameon=False)

    plt.tight_layout()
    sns.despine()
    
def plt_fig_3c(fs, stoch_sigmas):
    ticks = np.linspace(0, 100, 5, endpoint=True).round(1)
    cbar = plt.colorbar(shrink=.67, ticks=ticks)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(r'$R$ (bit/s)', fontsize=22)

    plt.xlabel('$f$ (Hz)', fontsize=22)
    plt.ylabel(r"$\sigma_W$ (V·s$^{-1/2}$)", fontsize=22)
    
    f_ticks = [1, 10, 20, 30, 40, 50]
    f_index = list(map(lambda value: index_from_tick(value, fs/Hz), f_ticks))
    plt.xticks(f_index, f_ticks, fontsize=20)
    
    s_ticks = [0.005, 0.01, 0.02, 0.03, 0.04]
    s_index = list(map(lambda value: index_from_tick(value, stoch_sigmas), s_ticks))
    plt.yticks(s_index, s_ticks, fontsize=20)
    
    theta_index = list(map(lambda value: index_from_tick(value, fs/Hz), theta_range))
    plt.axvline(x=theta_index[0], linestyle='--', linewidth=3, color='white')
    plt.axvline(x=theta_index[1], linestyle='--', linewidth=3, color='white')
    th_char_index = (theta_range[1] + theta_range[0])//2 + 2
    plt.text(x=th_char_index, y=90, s='θ', fontsize=28, color='white')

    plt.tight_layout()
    
    return

def plt_fig_3d(fs, stoch_sigmas):
    ticks = np.linspace(0, 1, 5, endpoint=True).round(1)
    cbar = plt.colorbar(shrink=.683, ticks=ticks)
    cbar.ax.tick_params(labelsize=20)

    '''
    plt.figure(figsize=(8,8))
    plt.imshow(normalized_bsm, cmap='hot', interpolation='bilinear', vmin=0, origin='lower')
    cbar = plt.colorbar(shrink=.683)
    #ticks = np.linspace(0, 1, 4, endpoint=True).round(1)
    #cbar = plt.colorbar(shrink=.683, ticks=ticks)
    #cbar.ax.tick_params(labelsize=20)
    '''

    plt.title('Normalized', fontsize=28, pad=20)

    plt.xlabel('$f$ (Hz)', fontsize=22)
    plt.ylabel(r"$\sigma_W$ (V·s$^{-1/2}$)", fontsize=22)
    
    f_ticks = [1, 10, 20, 30, 40, 50]
    f_index = list(map(lambda value: index_from_tick(value, fs/Hz), f_ticks))
    plt.xticks(f_index, f_ticks, fontsize=20)
    
    s_ticks = [0.005, 0.01, 0.02, 0.03, 0.04]
    s_index = list(map(lambda value: index_from_tick(value, stoch_sigmas), s_ticks))
    plt.yticks(s_index, s_ticks, fontsize=20)
    

    theta_index = list(map(lambda value: index_from_tick(value, fs/Hz), theta_range))

    plt.axvline(x=theta_index[0], linestyle='--', linewidth=3, color='black')
    plt.axvline(x=theta_index[1], linestyle='--', linewidth=3, color='black')
    
    th_char_index = (theta_range[1] + theta_range[0])//2 + 2
    plt.text(x=th_char_index, y=90, s='θ', fontsize=28, color='black')

    cbar.ax.tick_params(labelsize=20)

    plt.tight_layout()
    
    
def plt_fig_4b(fs, DV_gradient):
    ticks = np.linspace(0, 0.6, 4, endpoint=True).round(1)
    cbar = plt.colorbar(ticks= ticks, shrink=.71)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(r'$R$ (bit/s)', fontsize=22)
    plt.xlabel('Frequency (Hz)', fontsize=22)
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
    plt.text(x=th_char_index, y=90, s='θ', fontsize=28, color='black')

def plt_fig_4c(fs, DV_gradient):
    
    ticks = np.linspace(0, 1, 2, endpoint=True).round(1)
    cbar = plt.colorbar(ticks= ticks, shrink=.71)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(r'Norm. $R$', fontsize=22)
    plt.xlabel('Frequency (Hz)', fontsize=22)
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
    plt.text(x=th_char_index, y=90, s='θ', fontsize=28, color='black')
    
    return 
    
    
    
def plt_fig_5b():
    f_ticks = [1, 15, 35, 50]
    plt.xticks(f_ticks, fontsize=20)
    plt.xlabel(r"$f$ (Hz)", fontsize=20)

    plt.yticks(np.linspace(0,6, 3), fontsize=20)
    plt.ylabel(r"$R$ (bit/s)", fontsize=20)
    plt.legend(frameon=False)

    plt.tight_layout()
    sns.despine()

def plt_fig_5c(fs, Ioscs):
    ticks = np.linspace(0, 20, 5, endpoint=True).round(1)
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
    plt.text(x=th_char_index, y=90, s='θ', fontsize=28, color='black')

    plt.tight_layout()
    return

def plt_fig_5d(fs, Ioscs):
    ticks = np.linspace(0, 1, 5, endpoint=True).round(1)
    cbar = plt.colorbar(shrink=.683, ticks=ticks)
    cbar.ax.tick_params(labelsize=20)

    '''
    plt.figure(figsize=(8,8))
    plt.imshow(normalized_bsm, cmap='hot', interpolation='bilinear', vmin=0, origin='lower')
    cbar = plt.colorbar(shrink=.683)
    #ticks = np.linspace(0, 1, 4, endpoint=True).round(1)
    #cbar = plt.colorbar(shrink=.683, ticks=ticks)
    #cbar.ax.tick_params(labelsize=20)
    '''

    plt.title('Normalized', fontsize=28, pad=20)

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
    plt.text(x=th_char_index, y=90, s='θ', fontsize=28, color='black')
    
    plt.tight_layout()

    