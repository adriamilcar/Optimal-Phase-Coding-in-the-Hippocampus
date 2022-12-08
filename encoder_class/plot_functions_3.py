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
    
    
    




    