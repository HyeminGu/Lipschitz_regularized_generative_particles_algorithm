#!/usr/bin/env python3
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sys import platform, argv

def show_or_close_figure(show=False):
    if "linux" in platform or show == False:
        plt.clf()
        plt.close()
    else:
        plt.show() 

def load_pickle(filename):
    with open(filename, "rb") as fr:
        param, result = pickle.load(fr)
    return param, result
    
def read_pickle_data(filename):
    param, result = load_pickle(filename)
    
    X_ = param['X_']
    Y_ = param['Y_']
    
    trajectories = [Y_] + result['trajectories']
    if param['lr_Ps'][-1] == param['lr_Ps'][0]:
        lr_P = param['lr_P']
    else:
        lr_P = param['lr_Ps']
    dataset = param['dataset']
    if 'gaussian' in dataset:
         r_param = param['sigma_Q']
    elif 'student_t' in dataset:
        r_param = param['nu']
    elif dataset == 'Extension_of_gaussian':
        r_param = param['a']
    
    N_samples_P = param['N_samples_P']
    N_samples_Q = param['N_samples_Q']
    epochs = param['epochs']
    save_iter = param['save_iter']
    try:
        X_label, Y_label = param['X_label'], param['Y_label']
    else:
        X_label, Y_label = param['data_label'], param['data_label']
    
    N_dim = param['N_dim']
    
    return param, result, X_, Y_, trajectories, lr_P, dataset, r_param, N_samples_P, N_samples_Q, epochs, save_iter, X_label, Y_label, N_dim
    

    
# -----------------------------------------
# General drawing functions
# -----------------------------------------
def handle_dt(lr_P=None, lr_Ps=None, physical_time=True):
# lr_P (old) : lr_P scalar value for the entire epochs/the first epoch
# lr_P (new) : [lr_P]*N_moving_particles for the first epoch
# lr_Ps (old) : [lr_P]*epochs for all different epochs
# lr_Ps (new) : [[lr_P]*N_moving_particles]*epochs for all different epochs
    if physical_time == True:
        try:
            dt = lr_P[0]
        except:
            dt = lr_P
        if lr_Ps != None:
            try:
                dt = lr_Ps
                dt = [x[0] for x in dt]
            except:
                dt = lr_Ps
    else:
        dt = 1
    return dt
    
def calculate_time_steps(dt, iter_nos, physical_time=True):
# calculate real_time / iteration count
    if physical_time == True:
        iter_nos_new = []
        if type(dt) == list: # decaying or varying dt
            for iter_no in iter_nos:
                if iter_no == 0:
                    iter_nos_new.append(0)
                else:
                    iter_nos_new.append(sum(dt[:iter_no]))
        else: # constant dt
            for iter_no in iter_nos:
                iter_nos_new.append(dt*iter_no)
        iter_nos = iter_nos_new
    return iter_nos
                
def proj_and_sample(snapshot, proj_axes=None, pick_samples=None):
# return chosen samples and chosen axes
    if type(snapshot) != np.ndarray:
        return None
    snapshot = np.array(snapshot)
    if proj_axes != None:
        snapshot = snapshot[:,proj_axes]
    if pick_samples != None:
        snapshot = snapshot[pick_samples,:]
    return snapshot 
    
def set_axis_lim(samples, mask=1, lb=None, ub=None):
# set axis limit as [lb-1, ub+1]
# samples: list of arrays
    s_max, s_min = 0, 0
    for sample in samples:
        s_max, s_min = max(s_max, max(sample)), min(s_min, min(sample))
    s_max, s_min = s_max + mask, s_min - mask
    
    if lb != None:
        s_min = max(s_min, lb)
    if ub != None:
        s_max = min(s_max, ub)
    
    return (s_min, s_max)
    
    
    
# -----------------------------------------
# Additional features for Trajectories plot
# -----------------------------------------   
def add_quantile_contour(ax, dataset, r_param):
# gaussian) 50% quantile: solid line, 90% quantile: dashed line
# heavy-tailed) 25% quantile: solid line, 50% quantile: dashed line
# only applicable to dataset = gaussian/student_t/Stretched_exponential
    if 'gaussian' in dataset:
        r1 = 0.6745*r_param   # 50%
        r2 = 1.644854*r_param # 90%
    elif 'student_t' in dataset:
        if r_param == 0.5:
            r1 = 0.51856      # 25%
            r2 = 1.55377      # 50%
        elif r_param == 5:
            r1 = 0.33672      # 25%
            r2 = 0.72669      # 50%
    elif dataset == 'Stretched_exponential':
        if r_param == 0.7:
            r1 = 0.55         # 25%
            r2 = 1.2          # 50%
        elif r_param == 0.4:
            r1 = 2.8          # 25%
            r2 = 7            # 50% 
    else:
        return -1# irrelevant to plot quantiles
    
    if 'Mixture' not in dataset:
        centers=[(0, 0)]
    elif 'Mixture_of_gaussians' in dataset:
        centers=[(0, 0), (4,0), (0,4), (4,4)]
        if 'Mixture_of_gaussians2' in dataset:
            centers=[(0, 0), (8,0), (0,8), (8,8)]
    elif dataset == 'Mixture_of_student_t':
        centers=[(-10, -10), (10,-10), (-10,10), (10,10)]
    elif dataset == 'Mixture_of_student_t_submnfld':
        centers=[(-10, -10), (10,-10), (-10,10), (10,10)]
        
    circles1, circles2 = [], []
    for c in centers:
        circles1.append(plt.Circle(tuple(c), r1, color='k', linestyle='-', fill=False))
        circles2.append(plt.Circle(tuple(c), r2, color='k', linestyle='--', fill=False))
        ax.add_patch(circles1[-1])
        ax.add_patch(circles2[-1])
    return 0
        
# -----------------------------------------
# Individual figures
# -----------------------------------------    

# Scatter plot for given data
def plot_initial_data(X_, Y_, proj_axes = [0,1], x_lim = [None,None], y_lim = [None,None], show = False, marker_size = None, color_X = None, color_Y = None, base_font_size=16, save_filename=None):
# plot X_ : target data, Y_ : initial data in 2D projected plane
    X_ = proj_and_sample(X_, proj_axes)
    Y_ = proj_and_sample(Y_, proj_axes)
    
    plt.scatter(X_[:, 0], X_[:, 1], label="Target X", s=marker_size, c=color_X)
    plt.scatter(Y_[:, 0], Y_[:, 1], label="Initial Y", s=marker_size, c=color_Y)
    
    xlims = set_axis_lim([X_[:,0], Y_[:,0]], mask=1, lb=x_lim[0], ub=x_lim[1])
    ylims = set_axis_lim([X_[:,1], Y_[:,1]], mask=1, lb=y_lim[0], ub=y_lim[1])
    plt.xlim(xlims)
    plt.ylim(ylims)
    
    plt.legend(fontsize = base_font_size)
    plt.tight_layout()
    
    if save_filename != None:
        f = save_filename.split('.pickle')
        plt.savefig(f[0]+"-initial_data.png",bbox_inches='tight')
    
    show_or_close_figure(show)
    
# Scatter plot for target vs output
def plot_output_target(X_, trajectories, epochs, proj_axes = [0,1], x_lim = [None,None],y_lim = [None,None], show = False, marker_size = 5, color_X = None, color_Y = None, base_font_size=16, save_filename=None):
# plot X_ : target data, Y_ : initial data in 2D projected plane
    Y_ = trajectories[epochs]
            
    X_ = proj_and_sample(X_, proj_axes)
    Y_ = proj_and_sample(Y_, proj_axes)
    
    plt.scatter(X_[:, 0], X_[:, 1], label="Target", s=marker_size, c=color_X, alpha=0.7)
    plt.scatter(Y_[:, 0], Y_[:, 1], label="Output", s=marker_size, c=color_Y, alpha=0.7)
    
    xlims = set_axis_lim([X_[:,0], Y_[:,0]], mask=1, lb=x_lim[0], ub=x_lim[1])
    ylims = set_axis_lim([X_[:,1], Y_[:,1]], mask=1, lb=y_lim[0], ub=y_lim[1])
    plt.xlim(xlims)
    plt.ylim(ylims)
    
    plt.legend(fontsize = base_font_size)
    plt.tight_layout()
    
    if save_filename != None:
        f = save_filename.split('.pickle')
        plt.savefig(f[0]+"-output_target.png",bbox_inches='tight')
    
    show_or_close_figure(show)
 
# Scatter plots for time trajectories
def plot_trajectories(trajectories, lr_P, X_, dataset, epochs, save_iter, proj_axes = [0,1], r_param=None, iter_nos = None, physical_time=True, quantile=True, exp_alias_ = None, x_lim = [None,None],y_lim = [None,None], show = False, marker_size = None, color_X = None, color_Y = None, base_font_size=16, save_filename=None):
# plot trajectories of one file
# exp_alias_: one of the keys in param dictionary as a string '...' (to specify title)
    print('plot_trajectories')
    # make frames
    if iter_nos == None:
        n_figs = 4
    else:
        n_figs = len(iter_nos)
    f, axs = plt.subplots(nrows=1, ncols=n_figs, figsize=(15, 3.5))  
    
    if type(lr_P) == type([]):
        dt = handle_dt(None, lr_Ps=lr_P, physical_time=physical_time)
    else:
        dt = handle_dt(lr_P, lr_Ps=None, physical_time=physical_time)
            
    
    # pre-processing
    X_ = proj_and_sample(X_, proj_axes, pick_samples)
    trajectories = [proj_and_sample(x, proj_axes, pick_samples) for x in trajectories] 
    
    if iter_nos == None:
        iter_nos = [int(epochs*x/(n_figs-1)) for x in range(n_figs)]
    if type(trajectories[iter_nos[0]]) != np.ndarray: # no initial samples
        iter_nos[0] = save_iter
      
    trajectories = [trajectories[int(i/save_iter)] for i in iter_nos]
    time_steps = calculate_time_steps(dt, iter_nos, physical_time=physical_time)  
    
    # draw plots
    for i, ax in enumerate(axs):
        if physical_time == True:
            ax.set_title('T=%.3f'% time_steps[i])
        else:
            ax.set_title('T=%d' % iter_nos[i])
        # plot quantile contour or target data
        if quantile == True:
            flag = add_quantile_contour(ax, dataset, r_param)
            if flag == -1:
                ax.scatter(X_[:,0], X_[:,1], s=marker_size, c=color_X)
        else:
            ax.scatter(X_[:,0], X_[:,1], s=marker_size, c=color_X)
        
        # trajectories by speed of particles
        ax.scatter(trajectories[i][ :, 0], trajectories[i][ :, 1], s=marker_size, c=color_Y)
        
        xlims = set_axis_lim([X_[:,0], trajectories[i][:,0]], mask=1, lb=x_lim[0], ub=x_lim[1])
        ylims = set_axis_lim([X_[:,1], trajectories[i][:,1]], mask=1, lb=y_lim[0], ub=y_lim[1])
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)   
        
    plt.tight_layout()
    
    if save_filename != None:
        f = save_filename.split('.pickle')
        plt.savefig(f[0]+"-trajectories.png",bbox_inches='tight')
    
    show_or_close_figure(show)
            
def trajectories_to_animation(x_lim, y_lim, trajectories, N_samples_P, lr_P, epochs, save_iter, r_param = None, physical_time=True, quantile = True, show = False, marker_size = None, color_X = None, color_Y = None, base_font_size=16, save_filename=None):
    import matplotlib.animation as animation
    
    if type(lr_P) == type([]):
        dt = handle_dt(None, lr_Ps=lr_P, physical_time=physical_time)
    else:
        dt = handle_dt(lr_P, lr_Ps=None, physical_time=physical_time)
            
    trajectories = trajectories[:int(epochs/save_iter)]
    
    iter_nos = list(range(save_iter, save_iter+epochs, save_iter))
    time_steps = calculate_time_steps(dt, iter_nos, physical_time=physical_time)
        
    # make a frame
    fig, ax = plt.subplots()
    
    ims = []
    
    x1 = [x[:,0] for x in trajectories]
    x2 = [x[:,1] for x in trajectories]

    xlims = set_axis_lim([X_[:,0]] + x1, mask=1, lb=x_lim[0], ub=x_lim[1])
    ylims = set_axis_lim([X_[:,1]] + x2, mask=1, lb=y_lim[0], ub=y_lim[1])
    for i, x in enumerate(trajectories):
        if physical_time == True:
            ttl = ax.text(0.5,1.05, "t = %.3f" % time_steps[i], bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},transform=ax.transAxes, ha="center", size=base_font_size)
        else:
            ttl = ax.text(0.5,1.05, "t = %d" % iter_nos[i], bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},transform=ax.transAxes, ha="center", size=base_font_size)
        
        #print(x.shape)
        im = ax.scatter(x[ :, 0], x[ :, 1], s=marker_size, c=color_Y, zorder=10, alpha=0.7)
        #im.set_xlim(xlims)
        #im.set_ylim(ylims)
        
        # plot quantile contour or target data
        if quantile == True:
            flag = add_quantile_contour(ax, dataset, r_param)
            if flag == -1:
                ax.scatter(X_[:,0], X_[:,1], s=marker_size, c=color_X, zorder=1, alpha=0.7)
        else:
            ax.scatter(X_[:,0], X_[:,1], s=marker_size, c=color_X, zorder=1, alpha=0.7)
            
        
        ims.append([im, ttl])
        
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=False, repeat_delay=200)
   
    if save_filename != None:
        writergif = animation.PillowWriter(fps=3)
        f = save_filename.split('.pickle')
        ani.save(f[0]+"-movie.gif", writer=writergif)
    
    show_or_close_figure(show)
    
    
def trajectories_to_animation3D(x_lim, y_lim, z_lim, disp_angle=None, trajectories, N_samples_P, lr_P, epochs, save_iter, physical_time=True, show = False, marker_size = None, color_X = None, color_Y = None, base_font_size=16, save_filename=None):
    import matplotlib.animation as animation
    
    if type(lr_P) == type([]):
        dt = handle_dt(None, lr_Ps=lr_P, physical_time=physical_time)
    else:
        dt = handle_dt(lr_P, lr_Ps=None, physical_time=physical_time)
            
    trajectories = trajectories[:int(epochs/save_iter)]
    
    iter_nos = list(range(save_iter, save_iter+epochs, save_iter))
    time_steps = calculate_time_steps(dt, iter_nos, physical_time=physical_time)
    
    # make a frame
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    #ax.set_xlim([-7,6])
    #ax.set_ylim([-7, 12])
    if type(disp_angle) != type(None):
        ax.view_init(elev=disp_angle[0], azim=disp_angle[1], roll=disp_angle[2])
    
    ims = []
    
    x1 = [x[:,0] for x in trajectories]
    x2 = [x[:,1] for x in trajectories]
    x3 = [x[:,2] for x in trajectories]

    xlims = set_axis_lim([X_[:,0]] + x1, mask=1, lb=x_lim[0], ub=x_lim[1])
    ylims = set_axis_lim([X_[:,1]] + x2, mask=1, lb=y_lim[0], ub=y_lim[1])
    zlims = set_axis_lim([X_[:,2]] + x3, mask=1, lb=z_lim[0], ub=z_lim[1])
    
    for i, x in enumerate(trajectories):
        if physical_time == True:
            ttl = ax.text2D(0.5,1.05, "t = %.3f" % time_steps[i], bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},transform=ax.transAxes, ha="center", size=base_font_size)
        else:
            ttl = ax.text2D(0.5,1.05, "t = %d" % iter_nos[i], bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},transform=ax.transAxes, ha="center", size=base_font_size)
        
        #print(x.shape)
        im = ax.scatter(x[ :, 0], x[ :, 1], x[:,2], s=marker_size, c=color_Y, zorder=10, alpha=0.7)
    
        # plot target data
        ax.scatter(X_[:,0], X_[:,1], X_[:,2], s=marker_size, c=color_X, zorder=1, alpha=0.7)
        plt.tight_layout()
            
        ims.append([im, ttl])
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=False, repeat_delay=200)

    if save_filename != None:
        writergif = animation.PillowWriter(fps=3)
        f = save_filename.split('.pickle')
        ani.save(f[0]+"-movie.gif", writer=writergif)
    
    show_or_close_figure(show)







# Scatter plots for time trajectories with multiple experimental parameters(exp_alias_)
def plot_multiple_trajectories(filepath, exp_alias_, proj_axes = [0,1], pick_samples =None, epochs = 0, iter_nos = None, physical_time=True, save_iter = 1, track_velocity=False, arrow_scale = 1, iscolor=False, quantile=True, x_lim = [None,None],y_lim = [None,None], show = False):
# plot trajectories of one file
# exp_alias_: one of the keys in param dictionary as a string '...' (to specify title)
    # load filenames
    import os
    import re

    if not re.search("/$", filepath):
        filepath = filepath+"/"
    r = re.compile(".*pickle")
    filepath2 = filepath+'!/'
    filenames = list(filter(r.match, os.listdir(filepath2)))
    filenames = [filepath2+x for x in filenames]
    filenames.sort()
    
    print('plot_multiple_trajectories')
    
    # make frames
    if iter_nos == None:
        n_figs = 4
    else:
        n_figs = len(iter_nos)
    f, axs = plt.subplots(nrows=len(filenames), ncols=n_figs, figsize=(15, 7))
    
    
    
    for n, axs_row, filename in zip(range(len(axs)), axs, filenames):
        print(filename)
    
        # load pickled data
        param, result = load_pickle(filename)
        if epochs == 0:
            epochs = param['epochs']
        if epochs > 100:
            save_iter = param['save_iter']                
        X_ = param['X_']
        try:
            Y_ = param['Y_']
        except:
            pass
        trajectories = [Y_] + result['trajectories'] 
        try:
            vectorfields = result['vectorfields']
        except:
            pass
            
        if 'lr_Ps' in param.keys():
            dt = handle_dt(param['lr_P'], lr_Ps=param['lr_Ps'], physical_time=physical_time)
        else:
            dt = handle_dt(param['lr_P'], lr_Ps=None, physical_time=physical_time)

        if quantile == True:
            dataset = param['dataset']
            if 'gaussian' in dataset:
                 r_param = param['sigma_Q']
            elif 'student_t' in dataset:
                r_param = param['nu']
            elif dataset == 'Extension_of_gaussian':
                r_param = param['a']   
                       
        exp_alias = '%s=\n%s' % (exp_alias_[n], param[exp_alias_[n]])
    
        # pre-processing
        X_ = proj_and_sample(X_, proj_axes, pick_samples)
        trajectories = [proj_and_sample(x, proj_axes, pick_samples) for x in trajectories] 
        
        if iter_nos == None:
            iter_nos = [int(epochs*x/(n_figs-1)) for x in range(n_figs)] 
        if type(trajectories[0]) != np.ndarray: # no initial samples
            iter_nos[0] = save_iter          
          
        trajectories = [trajectories[int(i/save_iter)] for i in iter_nos]   
        time_steps = calculate_time_steps(dt, iter_nos, physical_time=physical_time)  
        
        if track_velocity or iscolor:
            iter_nos_ = iter_nos
            if iter_nos_[-1] == epochs:
                iter_nos_[-1] = epochs-1
            if vectorfields != []:
                vectorfields = [vectorfields[int(i/save_iter)] for i in iter_nos_]
        
        if iscolor == True:
            cv_max, cv_min = 0, 1e+16
            for vf in vectorfields:
                cv_max = max(cv_max, max(np.linalg.norm(vf, axis=1)))
                cv_min = min(cv_min, min(np.linalg.norm(vf, axis=1)))  
                c_map = [np.linalg.norm(vectorfields[i] , axis=1) for i in range(len(vectorfields))]  
        if track_velocity == True:
            dP = [proj_and_sample(x, proj_axes, pick_samples) for x in vectorfields]
        
        
        axs_row[0].text(-0.1, 0.5, exp_alias, size=16, transform=axs_row[0].transAxes, horizontalalignment='right')
        
        # draw plots in the same row
        for i, ax in enumerate(axs_row):
            if physical_time == True:
                ax.set_title('T=%.3f'% time_steps[i], fontsize=14)
            else:
                ax.set_title('T=%d' % iter_nos[i], fontsize=14)
            # plot quantile contour or target data
            if quantile == True:
                flag = add_quantile_contour(ax, param['dataset'], r_param)
                if flag == -1:
                    ax.scatter(X_[:,0], X_[:,1], s=5)
            else:
                ax.scatter(X_[:,0], X_[:,1], s=5)
            
            # color/uncolor trajectories by speed of particles        
            if iscolor == True:
                ax.scatter(trajectories[i][ :, 0], trajectories[i][ :, 1], c=abs(c_map[i]), cmap='seismic', s=5, vmin=cv_min, vmax=cv_max)
            else:
                ax.scatter(trajectories[i][ :, 0], trajectories[i][ :, 1], s=5) 
            
            xlims = set_axis_lim([X_[:,0], trajectories[i][:,0]], mask=1, lb=x_lim[0], ub=x_lim[1])
            ylims = set_axis_lim([X_[:,1], trajectories[i][:,1]], mask=1, lb=y_lim[0], ub=y_lim[1])
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)   
            
            # draw arrow for particle speed
            if track_velocity == True:
                n_Samples_P = dP[i].shape[0]
                for s in range(min(3, n_Samples_P)):
                    plot_arrow(ax, trajectories[i][s], dP[i][s], arrow_scale)
    plt.tight_layout()
    
    plt.savefig(filepath+"trajectories.png",bbox_inches='tight')
    
    show_or_close_figure(show)
     
    

# -----------------------------------------------
# one-label image trajectories
def plot_trajectories_img(X_, trajectories, lr_P, epochs, save_iter, pick_samples=None, iter_nos = None, physical_time=True, show=False, marker_size = None, color_X = None, color_Y = None, base_font_size=16, save_filename=None):
# plot trajectories of 2D image data of one file
    # exp_alias_: one of the keys in param dictionary as a string '...'    
    
    # load data
    if 'epochs' not in filename:
        param, result = load_pickle(filename)
        X_ = param['X_']
        try: 
            Y_ = param['Y_']
        except: 
            pass        
        trajectories = result['trajectories']
        N_samples_P = param['N_samples_P']
        if epochs == 0:
            epochs = param['epochs']
        if epochs > 100:
            save_iter = param['save_iter'] 
        
        if 'lr_Ps' in param.keys():
            dt = handle_dt(param['lr_P'], lr_Ps=param['lr_Ps'], physical_time=physical_time)
        else:
            dt = handle_dt(param['lr_P'], lr_Ps=None, physical_time=physical_time)
    else:
        dt = handle_dt(lr_Ps=dt, physical_time=physical_time)
        N_samples_P = trajectories[0].shape[0]
    
    # make frames
    if iter_nos == None:
        n_figs = 4
        iter_nos = [int(epochs*x/(n_figs-1)) for x in range(n_figs)] 
        if type(Y_) !=np.ndarray and iter_nos[0] == 0:
            iter_nos[0] = save_iter
    else:
        n_figs = len(iter_nos)
    f, axs = plt.subplots(nrows=1, ncols=n_figs)#+1)
    
    # pick an image
    if pick_samples == None:
        pick_samples = np.random.randint(N_samples_P)
                
    # determine time steps and plot certain time step trajectories iteratively
    trajectories = [Y_] + trajectories
    trajectories = [trajectories[int(i/save_iter)] for i in iter_nos]
    #trajectories = [X_] + trajectories
    
    time_steps =calculate_time_steps(dt, iter_nos, physical_time=physical_time)
    #time_steps = ['Target']+calculate_time_steps(dt, iter_nos, physical_time=physical_time)
     
    for i, ax in enumerate(axs):
        ax.set_title('T=%.3f'% time_steps[i])
        '''
        if physical_time == True and i>0:  
            ax.set_title('T=%.3f'% time_steps[i])
        else:
            ax.set_title(f'T={time_steps[i]}')
        '''
        ax.imshow(trajectories[i][pick_samples],interpolation='nearest', vmin=-0.0, vmax=1.0)
        ax.axis('off')
    plt.tight_layout()
    
    f = filename.split('.pickle')
    plt.savefig(f[0]+"-trajectories.png",bbox_inches='tight')
    
    show_or_close_figure(show)
    
def plot_multiple_trajectories_img(filepath, exp_alias_, proj_axes = [0,1], pick_samples =None, epochs = 0, iter_nos = None, physical_time=True, save_iter = 1, show=False):
# plot trajectories of one file
# exp_alias_: one of the keys in param dictionary as a string '...' (to specify title)
    # load filenames
    import os
    import re

    if not re.search("/$", filepath):
        filepath = filepath+"/"
    r = re.compile(".*pickle")
    filepath2 = filepath+'!/'
    filenames = list(filter(r.match, os.listdir(filepath2)))
    filenames = [filepath2+x for x in filenames]
    filenames.sort()
    
    print('plot_multiple_trajectories_img')
    
    # make frames
    if iter_nos == None:
        n_figs = 4
    else:
        n_figs = len(iter_nos)
    f, axs = plt.subplots(nrows=len(filenames), ncols=n_figs+1, figsize=(15, 7))     
   
    
    for n, axs_row, filename in zip(range(len(axs)), axs, filenames):
        print(filename)
    
        # load pickled data
        param, result = load_pickle(filename)
        X_ = param['X_']
        try:
            Y_ = param['Y_']
        except:
            Y_ = None
        trajectories = [Y_] + result['trajectories'] 
            
        if epochs == 0:
            epochs = param['epochs']
        if epochs > 100:
            save_iter = param['save_iter']         
       
        if 'lr_Ps' in param.keys():
            dt = handle_dt(param['lr_P'], lr_Ps=param['lr_Ps'], physical_time=physical_time)
        else:
            dt = handle_dt(param['lr_P'], lr_Ps=None, physical_time=physical_time)
            
        N_samples_P = param['N_samples_P']       
        if pick_samples == None:
            pick_sample = np.random.randint(N_samples_P)
                       
        exp_alias = '%s=\n%s' % (exp_alias_[n], param[exp_alias_[n]])
    
        # determine time steps and plot certain time step trajectories iteratively        
        if iter_nos == None:
            iter_nos = [int(epochs*x/(n_figs-1)) for x in range(n_figs)] 
        if type(trajectories[0]) != np.ndarray: # no initial samples
            iter_nos[0] = save_iter      

        trajectories = [Y_] + trajectories
        trajectories = [trajectories[int(i/save_iter)] for i in iter_nos] 
        trajectories = [X_] + trajectories
          
        time_steps = ['Target']+calculate_time_steps(dt, iter_nos, physical_time=physical_time)  
        
        axs_row[0].text(-0.1, 0.5, exp_alias, size=15, transform=axs_row[0].transAxes, horizontalalignment='right')
        
        # plot each row
        for i, ax in enumerate(axs_row):
            if physical_time == True and i>0:  
                ax.set_title('T=%.3f'% time_steps[i])
            else:
                ax.set_title(f'T={time_steps[i]}')
            ax.imshow(trajectories[i][pick_sample],interpolation='nearest', vmin=-0.0, vmax=1.0)
            ax.axis('off')
 
    plt.tight_layout()
    
    plt.savefig(filepath+"trajectories.png",bbox_inches='tight')
    
    show_or_close_figure(show)
       
    
def plot_trained_img(X_ = None, trajectories = None, pick_samples=None, epochs=0, filename=None, show=False):
# plot target and final trajectories of 2D image data 
    # make frames
    if pick_samples == None:
        n_figs = 4
    else:
        n_figs = len(pick_samples)
    f, axs = plt.subplots(nrows=2, ncols=n_figs)
    
    # load data
    if 'epochs' not in filename:
        param, result = load_pickle(filename)
        X_ = param['X_']
        trajectories = result['trajectories']
        N_samples_P = param['N_samples_P']
        N_samples_Q = param['N_samples_Q']
    else:
        N_samples_P = trajectories[0].shape[0]
        N_samples_Q = X_.shape[0]

    if pick_samples == None:
        pick_samples = list(np.random.randint(N_samples_P, size=n_figs))
    
    axs[0,0].text(-0.1, 0.3, 'Target', size=15, transform=axs[0,0].transAxes, horizontalalignment='right') 
    axs[1,0].text(-0.1, 0.3, 'Learned', size=15, transform=axs[1,0].transAxes, horizontalalignment='right') 
    
    idx = np.random.randint(0, N_samples_P, n_figs)
    for i in range(n_figs):
        if N_samples_P == N_samples_Q:
            axs[0,i].imshow(X_[idx[i]], interpolation='nearest', vmin=-0.0, vmax=1.0)
        else:
            idx_Q = np.random.randint(0, N_samples_Q, n_figs)
            axs[0,i].imshow(X_[idx_Q[i]], interpolation='nearest', vmin=-0.0, vmax=1.0)
        axs[0,i].axis('off')
        axs[1,i].imshow(trajectories[-1][idx[i]], interpolation='nearest', vmin=-0.0, vmax=1.0)
        axs[1,i].axis('off')
    
    plt.tight_layout()
    f = filename.split('.pickle')
    plt.savefig(f[0]+"-learned.png",bbox_inches='tight')
    
    show_or_close_figure(show)
    

# multi-label image tiles for conditional gpa 
def plot_tiled_images(print_multiplier, samples=None, sample_label=None, epochs = 0, filename=None, show=False):
# plot several 2D images from designated epoch from one conditional gpa/gan experiment and show a tiled plot
# epochs = 0: last trajectory, -1: target images, ##: ##'th trajectory
# sample_label: N_samples x num_classes one-hot encoded
    print('plot_tiled_images')
    
    # load pickled data
    if 'epochs' not in filename:
        param, result = load_pickle(filename)
        if epochs == 0:
            epochs = param['epochs']
        save_iter = param['save_iter']
            
        if epochs == -1:
            try:
                samples, sample_label = param['X_'], param['X_label'] # one-hot encoding label
            except:
                samples, sample_label = param['X_'],param['data_label']
        else:
            try:
                samples, sample_label = result['trajectories'][int(epochs/save_iter)-1], param['Y_label']
            except:
                samples, sample_label = result['trajectories'][int(epochs/save_iter)-1], param['data_label']
        '''
        try:
            print(param['mobility'], param['activation_ftn'], param['ode_solver'])
        except:
            print(param['activation_ftn'], param['ode_solver'])
        '''
            

    zero_arr = np.zeros_like(samples[0])[np.newaxis,:]
    num_classes = np.shape(sample_label)[1]
    for i in range(num_classes):
        i_idx = np.squeeze(np.where(sample_label[:,i] ==1))
        np.random.shuffle(i_idx)
        if len(i_idx) < print_multiplier:
            zeros_arr = np.repeat(zero_arr,print_multiplier-len(i_idx), axis=0)
            i_data = np.concatenate( (samples[i_idx], zeros_arr), axis=0)
        else:
            i_data = samples[i_idx[:print_multiplier]]
        
        try:
            samples_ = i_data.transpose(1,0,2,3)
            newrows = np.reshape(samples_, (samples_.shape[0], samples_.shape[1]*samples_.shape[2], samples_.shape[3]))
        except:
            samples_ = i_data.transpose(1,0,2)
            newrows = np.reshape(samples_, (samples_.shape[0], samples_.shape[1]*samples_.shape[2]))
        if i == 0:
            rows = newrows
        else:
            rows = np.concatenate((rows, newrows), axis=0)    
    plt.imshow(rows, interpolation='nearest', vmin=-0.0, vmax=1.0)    
    plt.axis('off')
    plt.tight_layout()
    
    f = filename.split('.pickle')
    if epochs != -1:
        plt.savefig(f[0]+"-tiled_image.png",bbox_inches='tight')
    else:
        plt.savefig(f[0]+"-tiled_target.png",bbox_inches='tight')
    
    show_or_close_figure(show)
    
    x = np.reshape(rows, -1)
    print('[',min(x), max(x),']')
    plt.hist(x, range=(-2, 3), bins=100)
    plt.savefig(f[0]+"-pixel_values.png")
    show_or_close_figure(show)
    
def images_to_animation(trajectories=None, dt=None, physical_time=True, pick_samples = None, epochs=0, save_gif=True, filename = None, show=False):
    import matplotlib.animation as animation
    
    # load data
    if 'epochs' not in filename:
        param, result = load_pickle(filename)
        trajectories = result['trajectories']
        N_samples_P = param['N_samples_P']
        
        if epochs==0:
            epochs = param['epochs']
        else:
            param['epochs'] = epochs
        save_iter = param['save_iter']
        
        if 'lr_Ps' in param.keys():
            dt = handle_dt(param['lr_P'], lr_Ps=param['lr_Ps'], physical_time=physical_time)
        else:
            dt = handle_dt(param['lr_P'], lr_Ps=None, physical_time=physical_time)
            
    trajectories = result['trajectories'][:int(epochs/save_iter)]
    N_samples_P = result['trajectories'][0].shape[0]
    if pick_samples == None:
        pick_samples = np.random.randint(N_samples_P)
        
    iter_nos = list(range(save_iter, save_iter+epochs, save_iter))
    time_steps = calculate_time_steps(dt, iter_nos, physical_time=physical_time)
        
    # make a frame
    fig, ax = plt.subplots()
    
    ims = []
    for i, x in enumerate(trajectories):
        im = ax.imshow(x[pick_samples], interpolation='nearest', vmin=-0.0, vmax=1.0)
        ttl = ax.text(0.5,1.05, "t = {}".format(time_steps[i]), bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},transform=ax.transAxes, ha="center", size=base_font_size)
        ims.append([im, ttl])
        
    ani = animation.ArtistAnimation(fig, ims, interval=300, blit=False, repeat_delay=500)
   
    if save_gif:
        writergif = animation.PillowWriter(fps=3) 
        f = filename.split('.pickle')
        ani.save(f[0]+"-movie.gif", writer=writergif)
    
    show_or_close_figure(show)
    
# -------------------------------------
# 1D example
def plot_density_1D(X_ = None, Y_ = None, trajectories = None, save_iter = 1,dt = None, proj_axes=[0,1], epochs=0, iter_nos=None, exclude_initial_time = False, exclude_target = False, physical_time = True, filename=None, show=False):
# plot density of values in the orthogonal axes: concentration to 0 is better
    print(f'plot_density_1D {filename}')
    from scipy.stats import gaussian_kde
    
    # load data
    if 'epochs' not in filename:
        param, result = load_pickle(filename)
        trajectories = result['trajectories']
        N_samples_P = param['N_samples_P']
        X_ = param['X_']
        try:
            Y_ = param['Y_']
        except:
            pass
        
        if epochs==0:
            epochs = param['epochs']
        else:
            param['epochs'] = epochs
        save_iter = param['save_iter']
        
        if 'lr_Ps' in param.keys():
            dt = handle_dt(param['lr_P'], lr_Ps=param['lr_Ps'], physical_time=physical_time)
        else:
            dt = handle_dt(param['lr_P'], lr_Ps=None, physical_time=physical_time)
    else:
        dt = handle_dt(lr_Ps = dt, physical_time = physical_time)
            
    N_dim = trajectories[0].shape[1]
    try:
        n_figs = len(iter_nos)
    except:
        n_figs = 4
        if type(Y_) !=np.ndarray or exclude_initial_time == True: # do not count initial distribution
            iter_nos = [int(epochs*x/n_figs)+int(epochs/n_figs) for x in range(n_figs)]
        else:
            iter_nos = [int(epochs*x/(n_figs-1)) for x in range(n_figs)]
    
    trajectories = [Y_] + trajectories[:int(epochs/save_iter)]
    trajectories = [trajectories[int(i/save_iter)] for i in iter_nos]
    
    time_steps = calculate_time_steps(dt, iter_nos, physical_time=physical_time)

    if exclude_target == False:
        iter_nos.append('target')
        n_figs += 1
    
    for i in range(n_figs):
        if exclude_target == False and i == n_figs-1:
            trajectory = X_[:,0]
        else:
            trajectory = trajectories[i][:,0]
        x= np.linspace(trajectory.min()-0.01*np.abs(trajectory.min()), trajectory.max()+0.01*np.abs(trajectory.max()), 1000)
        z = gaussian_kde(trajectory)(x)
        
        # ---
        print(max(z))
        if exclude_target == True or i < n_figs-1:
            plt.plot(x, z, linestyle='--', label='T = {}'.format(time_steps[i]) )
        else:
            plt.plot(x, z, linestyle='-.', label='Target')
    plt.tight_layout()
    plt.legend()
    f = filename.split('.pickle')
    plt.savefig(f[0]+"-density.png",bbox_inches='tight')
    # ---
    try:
        print(param['lr_NN'], param['lr_P'], param['epochs_nn'],param['epochs'])
    except:
        print(param['lr_phi'], param['lr_P'], param['epochs_phi'],param['epochs'])
    
    show_or_close_figure(show)
    
def plot_multiple_orth_axes_saturation(filepath, proj_axes=[5,6], epochs=0, iter_nos=None, physical_time = True, show=False):
# plot density of values in the orthogonal axes: concentration to 0 is better
    from scipy.stats import gaussian_kde
    
    # load data
    import os
    import re
    
    r = re.compile(".*pickle")
    filenames = list(filter(r.match, os.listdir(filepath)))
    filenames = [filepath+x for x in filenames]
    filenames.sort()
    print('plot multiple orth axes saturations')
    
    for i, filename in enumerate(filenames):
        print(filename)
        exp = filename.split('_')
        
        param, result = load_pickle(filename)
        trajectories = result['trajectories']
        N_samples_P = param['N_samples_P']
        try:
            Y_ = param['Y_']
        except:
            pass
        
        if epochs==0:
            epochs = param['epochs']
        else:
            param['epochs'] = epochs
        save_iter = param['save_iter']
            
        N_dim = param['N_dim']
        trajectories = trajectories[int(epochs/save_iter)]
    
        orth_axes = list(set(range(N_dim))-set(proj_axes))
    
        complement_data = proj_and_sample(trajectories, orth_axes, pick_samples=None)
        complement_data = complement_data.flatten()
    
        x= np.linspace(complement_data.min(), complement_data.max(), 1000)
        z = gaussian_kde(complement_data)(x)
        plt.plot(x, z, linestyle='-', label='{}'.format(exp[0]) )
    plt.tight_layout()
    plt.legend()
    plt.savefig(f[0]+"-orthogonal.png",bbox_inches='tight')
    
    show_or_close_figure(show)
    
# -------------------------------------
# 12D embedded examples
def plot_orth_axes_saturation(Y_ = None, trajectories = None, save_iter = 1,dt = None, proj_axes=[5,6], epochs=0, iter_nos=None, exclude_initial_time = True, physical_time = True, filename=None, show=False):
# plot density of values in the orthogonal axes: concentration to 0 is better
    print(f'plot_orth_axes_saturation {filename}')
    from scipy.stats import gaussian_kde
    
    # load data
    if 'epochs' not in filename:
        param, result = load_pickle(filename)
        trajectories = result['trajectories']
        N_samples_P = param['N_samples_P']
        try:
            Y_ = param['Y_']
        except:
            pass
        
        if epochs==0:
            epochs = param['epochs']
        else:
            param['epochs'] = epochs
        save_iter = param['save_iter']
        
        if 'lr_Ps' in param.keys():
            dt = handle_dt(param['lr_P'], lr_Ps=param['lr_Ps'], physical_time=physical_time)
        else:
            dt = handle_dt(param['lr_P'], lr_Ps=None, physical_time=physical_time)
    else:
        dt = handle_dt(lr_Ps = dt, physical_time = physical_time)
            
    N_dim = trajectories[0].shape[1]
    try:
        n_figs = len(iter_nos)
    except:
        n_figs = 4
        if type(Y_) !=np.ndarray or exclude_initial_time == True: # do not count initial distribution
            iter_nos = [int(epochs*x/n_figs)+int(epochs/n_figs) for x in range(n_figs)]
        else:
            iter_nos = [int(epochs*x/(n_figs-1)) for x in range(n_figs)]
    
    trajectories = [Y_] + trajectories[:int(epochs/save_iter)]
    trajectories = [trajectories[int(i/save_iter)] for i in iter_nos]
    
    orth_axes = list(set(range(N_dim))-set(proj_axes))
    time_steps = calculate_time_steps(dt, iter_nos, physical_time=physical_time)

    for i in range(n_figs):
        complement_data = proj_and_sample(trajectories[i], orth_axes, pick_samples=None)
        complement_data = complement_data.flatten()
    
        x= np.linspace(complement_data.min(), complement_data.max(), 1000)
        z = gaussian_kde(complement_data)(x)
        
        # ---
        print(max(z))
        plt.plot(x, z, linestyle='-', label='T = {}'.format(time_steps[i]) ) 
    plt.tight_layout()
    plt.legend()
    f = filename.split('.pickle')
    plt.savefig(f[0]+"-orthogonal.png",bbox_inches='tight')
    # ---
    try:
        print(param['lr_NN'], param['lr_P'], param['epochs_nn'],param['epochs'])
    except:
        print(param['lr_phi'], param['lr_P'], param['epochs_phi'],param['epochs'])
    
    show_or_close_figure(show)
    
    
def plot_multiple_orth_axes_saturation(filepath, proj_axes=[5,6], epochs=0, iter_nos=None, physical_time = True, show=False):
# plot density of values in the orthogonal axes: concentration to 0 is better
    from scipy.stats import gaussian_kde
    
    # load data
    import os
    import re
    
    r = re.compile(".*pickle")
    filenames = list(filter(r.match, os.listdir(filepath)))
    filenames = [filepath+x for x in filenames]
    filenames.sort()
    print('plot multiple orth axes saturations')
    
    for i, filename in enumerate(filenames):
        print(filename)
        exp = filename.split('_')
        
        param, result = load_pickle(filename)
        trajectories = result['trajectories']
        N_samples_P = param['N_samples_P']
        try:
            Y_ = param['Y_']
        except:
            pass
        
        if epochs==0:
            epochs = param['epochs']
        else:
            param['epochs'] = epochs
        save_iter = param['save_iter']  
            
        N_dim = param['N_dim']
        trajectories = trajectories[int(epochs/save_iter)]  
    
        orth_axes = list(set(range(N_dim))-set(proj_axes))
    
        complement_data = proj_and_sample(trajectories, orth_axes, pick_samples=None)
        complement_data = complement_data.flatten()
    
        x= np.linspace(complement_data.min(), complement_data.max(), 1000)
        z = gaussian_kde(complement_data)(x)
        plt.plot(x, z, linestyle='-', label='{}'.format(exp[0]) ) 
    plt.tight_layout()
    plt.legend()
    plt.savefig(f[0]+"-orthogonal.png",bbox_inches='tight')
    
    show_or_close_figure(show)
    

# -----------------------------------------
# Additional features for Loss plot
# -----------------------------------------      
def fit_line(epochs, loss_states, plot_scale, save_iter, dt):
# calculate fitting line of loss
    import numpy as np
    start_idx = 2
    end_idx = min([20, int(epochs/save_iter)])
    if type(dt) == list : # decaying dt
        x_range = np.cumsum(dt[start_idx*save_iter+1:end_idx*save_iter+1])
    else: # constant dt
        x_range = np.arange(start_idx*save_iter+1, end_idx*save_iter+1, save_iter)*dt
    logB = np.log10(np.abs(loss_states[start_idx:end_idx]))
    #print(len(x_range), len(logB))
    
    if plot_scale == "semilogy":
        line_coefs = np.polyfit(x_range, logB, 1)
    elif plot_scale == "loglog":
        log_x = np.log10(x_range)
        line_coefs = np.polyfit(log_x, logB, 1)
    return x_range, line_coefs
    
def plot_fitting_line(x_range, line_coefs, plot_scale):
# plot fitting line
    m, y0 = line_coefs
    if plot_scale == "semilogy":
        plt.plot(x_range, 10**(x_range*m+y0), linestyle = 'dotted', label = 'exp[(%.4f)x+(%.2f)]' %(m,y0) )
    if plot_scale == "loglog":
        plt.plot(x_range, (10**y0)*x_range**m, linestyle = 'dotted', label = 'x^(%.4f)*10^(%.2f)' %(m,y0) )
    
def plot_loss(loss_states, epochs, physical_time, plot_scale, dt, exp_alias=None, save_iter=1, lty="solid", color = None, linewidth=1):
# plot setting for the certain type of loss (loss_states) with(out) fitting lines
# plot_scale = 'semilogy' or 'loglog'
    from numpy import arange
    if physical_time == True:
        xlabel_name = 'Time'
    else:
        xlabel_name = 'Iteration'
    
    iter_nos = range(save_iter, epochs+save_iter, save_iter)
    x_val = calculate_time_steps(dt, iter_nos, physical_time)
    '''
    if type(dt) == list : # decaying dt
        from numpy import cumsum
        try:
            dt = [x[0] for x in dt]
        except:
            pass
        x_val = cumsum(dt[0:epochs+save_iter:save_iter])
    else: # constant dt
        x_val = arange(save_iter,epochs+save_iter, save_iter)*dt
    '''
    if exp_alias == None:
        if plot_scale == "semilogy":
            plt.semilogy(x_val, [abs(x) for x in loss_states], color=color, linestyle=lty, linewidth=linewidth)
        if plot_scale == "loglog":
            plt.loglog(x_val, [abs(x) for x in loss_states], color=color, linestyle=lty, linewidth=linewidth)
            xlabel_name = xlabel_name+ ' (log scale)'
    else:
        if plot_scale == "semilogy":
            plt.semilogy(x_val, [abs(x) for x in loss_states], color=color, linestyle=lty, linewidth=linewidth, label=exp_alias)
        if plot_scale == "loglog":
            plt.loglog(x_val, [abs(x) for x in loss_states], color=color, linestyle=lty, linewidth=linewidth, label=exp_alias)
            xlabel_name = xlabel_name+ ' (log scale)'
    plt.xlabel(xlabel_name, fontsize="16")
    

    
# -------------------------------------
# Loss plot
# -------------------------------------     
# for 12D submnfld data
def plot_speeds(vectorfields = [], save_iter=1, dt=None, plot_scale='semilogy', proj_axes = [5,6], physical_time=True, epochs=0, filename = None, show=False):
    print('plot speed trajectories decomposed to the projected plane and the orthogonal complement')
    
    if 'epochs' not in filename:
        param, result = load_pickle(filename)        
        if epochs==0:
            epochs = param['epochs']
        else:
            param['epochs'] = epochs
        save_iter = param['save_iter']
        
        if 'lr_Ps' in param.keys():
            dt = handle_dt(param['lr_P'], lr_Ps=param['lr_Ps'], physical_time=physical_time)
        else:
            dt = handle_dt(param['lr_P'], lr_Ps=None, physical_time=physical_time)
        vectorfields = result['vectorfields']
    else:
        dt = handle_dt(lr_Ps = dt, physical_time=physical_time)
    N_dim = vectorfields[0].shape[1]
        
    # preprocess data
    pick_samples = None
    vectorfields = vectorfields[:int(epochs/save_iter)]
    proj_vf = [proj_and_sample(x, proj_axes, pick_samples) for x in vectorfields]
    orth_axes = list(set(range(N_dim))-set(proj_axes))
    orth_vf = [proj_and_sample(x, orth_axes, pick_samples) for x in vectorfields]
    
    proj_speed = [np.linalg.norm(proj_vf[i])/len(proj_axes) for i in range(int(epochs/save_iter))]
    orth_speed = [np.linalg.norm(orth_vf[i])/len(orth_axes) for i in range(int(epochs/save_iter))]
    
    # plot loss
    plot_loss(proj_speed, epochs, physical_time, plot_scale, dt, save_iter=save_iter, exp_alias='projected')
    plot_loss(orth_speed, epochs, physical_time, plot_scale, dt, save_iter=save_iter, exp_alias='orthogonal')
    plt.ylabel('average speed (log scale)')
    
    if type(dt) == list : # decaying dt
        plt.xlim([dt[0]*save_iter, sum(dt)])
    else: # constant dt
        plt.xlim([dt*save_iter, epochs*dt]) 
    plt.legend()    
    plt.tight_layout()
    f = filename.split('.pickle')
    plt.savefig(f[0]+"-separated_speeds.png",bbox_inches='tight')
    
    show_or_close_figure(show)
    

def plot_f_lip_div_and_wass1(loss_states=None, wass1s = None, plot_scale='semilogy', save_iter = 1, dt = None, epochs=0, ylims=None, physical_time=True, filename=None, show=False):
# from one file, plot a designated type of f-Lipschitz divergence and Wasserstein 1 metric
# iter_nos = [t_1, t_2, t_3,...] marks dots on the loss value of chosen epochs
    print(f'plot f-Lipschitz divergence and Wasserstein 1 metric')
    
    # load data
    if 'epochs' not in filename:
        param, result = load_pickle(filename)
        if epochs==0:
            epochs = param['epochs']
        if epochs > 100:
            save_iter = param['save_iter']
        else:
            save_iter = 1
        
        if 'lr_Ps' in param.keys():
            dt = handle_dt(param['lr_P'], lr_Ps=param['lr_Ps'], physical_time=physical_time)
        else:
            dt = handle_dt(param['lr_P'], lr_Ps=None, physical_time=physical_time)
            
        loss_states = result['divergences']
        wass1s = result['wasserstein1s']
        if np.isnan(result['divergences']).any():
            print(f'{filename} loss diverged')
            result['divergences'] = len(result['divergences'])*[0]
        loss_states = np.array(result['divergences'])
        
    else:
        dt = handle_dt(lr_Ps=dt, physical_time=physical_time)
        
    for x in loss_states:
        if np.isnan(x):
            x = 0
        
    save_iter = 1
       
    loss_states = loss_states[:int(epochs/save_iter)]
    wass1s = wass1s[:int(epochs/save_iter)]
    
    # plot loss
    plot_loss(loss_states, epochs, physical_time, plot_scale, dt=dt, save_iter=save_iter, exp_alias='f-Lip divergence')
    plot_loss(wass1s, epochs, physical_time, plot_scale, dt=dt, save_iter=save_iter, exp_alias='Wass1 metric', lty="--")

    plt.ylabel('losses')
    
    if type(dt) == list : # decaying dt
        plt.xlim([dt[0]*save_iter, sum(dt)])
    else: # constant dt
        plt.xlim([dt*save_iter, epochs*dt])
    if ylims != None:
        plt.ylim(ylims)
    plt.legend()
    
    plt.tight_layout()
    f = filename.split('.pickle')
    plt.savefig(f[0]+"-divergence_and_wasserstein1.png",bbox_inches='tight')
    

def plot_multiple_f_lip_div_and_wass1(exp_alias_, filepath, plot_scale='semilogy', save_iter = 1, epochs=0, ylims=None, physical_time=True, show=False):
# from one file, plot a designated type of loss (loss_type)
# iter_nos = [t_1, t_2, t_3,...] marks dots on the loss value of chosen epochs
    print(f'plot multiple f-Lipschitz divergence and Wasserstein 1 metric')
    
    colors = ['r','g','b','c','m','y']
    lws = [2, 3, 1, 4, 1.5, 3.5]
    
    # load data
    import os
    import re
    
    if not re.search("/$", filepath):
        filepath = filepath+"/"
    r = re.compile(".*pickle")
    directories = [x for x in os.listdir(filepath) if not (re.search("(png$|gif$|^\.)", x))]
    directories.sort()
    for j, directory in enumerate(directories[1:]):
        filenames = list(filter(r.match, os.listdir(filepath+directory)))
        filenames = ["/".join((filepath, directory, x)) for x in filenames]
        
        for i, filename in enumerate(filenames):
            #print(filename)
            param, result = load_pickle(filename)
            if epochs==0:
                epochs = param['epochs']
            else:
                param['epochs'] = epochs
            if epochs > 100:
                save_iter = param['save_iter']
            else:
                save_iter = 1
            L = param['L']
            
            if 'lr_Ps' in param.keys():
                dt = handle_dt(param['lr_P'], lr_Ps=param['lr_Ps'], physical_time=physical_time)
            else:
                dt = handle_dt(param['lr_P'], lr_Ps=None, physical_time=physical_time)
                
            for x in result['divergences']:
                if np.isnan(x):
                    x = 0
            #if np.isnan(result[loss_type]).any():
            #    print(f'{filename} loss diverged')
            #    result[loss_type] = len(result[loss_type])*[0]
            if i == 0:
                loss_states = np.array(result['divergences'])
            else:
                loss_states = loss_states + np.array(result['divergences'])
                
            for x in result['wasserstein1s']:
                if np.isnan(x):
                    x = 0
            #if np.isnan(result[loss_type]).any():
            #    print(f'{filename} loss diverged')
            #    result[loss_type] = len(result[loss_type])*[0]
            if i == 0:
                wass1s = np.array(result['wasserstein1s'])
            else:
                wass1s = wass1s + np.array(result['wasserstein1s'])
        
            if exp_alias_ != None:
                exp_alias = '%s' % (param[exp_alias_[j]])
        
        save_iter = 1
        loss_states = loss_states[:int(epochs/save_iter)]
        wass1s = wass1s[:int(epochs/save_iter)]
        
        # plot loss
        aa = "L=%3.1f, " % L
        plot_loss(loss_states, epochs, physical_time, plot_scale, dt=dt, save_iter=save_iter, exp_alias=aa+"f-Lip", linewidth=lws[j], color=colors[j])
        plot_loss(L*wass1s, epochs, physical_time, plot_scale, dt=dt, save_iter=save_iter, exp_alias=aa+"L * Wass1", lty="--", linewidth=lws[j], color = 'k')
    
        plt.ylabel('losses')
        
    if type(dt) == list : # decaying dt
        plt.xlim([dt[0]*save_iter, sum(dt)])
    else: # constant dt
        plt.xlim([dt*save_iter, epochs*dt])
    if ylims != None:
        plt.ylim(ylims)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filepath+"divergence_and_wasserstein1.png",bbox_inches='tight')
        
    show_or_close_figure(show)

# -----------------------------------
# plot generic loss
# -----------------------------------
def plot_losses(loss_type, loss_states=None, plot_scale='semilogy', fitting_line=False, save_iter = 1, dt = None, iter_nos = None, exp_alias_=None, epochs=0, ylims=None, physical_time=True, filename=None, show=False):
# from one file, plot a designated type of loss (loss_type) 
# iter_nos = [t_1, t_2, t_3,...] marks dots on the loss value of chosen epochs
    print(f'plot {loss_type}')
    
    # load data
    if 'epochs' not in filename:
        param, result = load_pickle(filename)        
        if epochs==0:
            epochs = param['epochs']
        if epochs > 100:
            save_iter = param['save_iter']
        else:
            save_iter = 1
        
        if 'lr_Ps' in param.keys():
            dt = handle_dt(param['lr_P'], lr_Ps=param['lr_Ps'], physical_time=physical_time)
        else:
            dt = handle_dt(param['lr_P'], lr_Ps=None, physical_time=physical_time)
            
        loss_states = result[loss_type]
        if np.isnan(result[loss_type]).any():
            print(f'{filename} loss diverged')   
            result[loss_type] = len(result[loss_type])*[0] 
        loss_states = np.array(result[loss_type])
        
        if exp_alias_ != None:
            exp_alias = '%s=%s' % (exp_alias_, param[exp_alias_])
    else:
        dt = handle_dt(lr_Ps=dt, physical_time=physical_time)
        
    for x in loss_states:
        if np.isnan(x):
            x = 0
        
    if loss_type != 'FIDs':
        save_iter = 1
    if fitting_line == True:
        x_range, line_coefs = fit_line(epochs, loss_states, plot_scale, save_iter, dt)
        
    loss_states = loss_states[:int(epochs/save_iter)]
    print("Last: %s = %f" % (loss_type, loss_states[-1]))
    
    # plot loss
    plot_loss(loss_states, epochs, physical_time, plot_scale, dt=dt, save_iter=save_iter)    
    
    # mark specific points
    if iter_nos != None:
        if type(dt) == list : # decaying dt
            x_val = [sum(dt[:iter_no]) for iter_no in iter_nos]
        else: # constant dt
            x_val = [iter_no*dt for iter_no in iter_nos]
        
        plt.plot(x_val, [abs(loss_states[x-1]) for x in iter_nos], '.', color='red')
        
    # fitting line
    if fitting_line == True:
        plot_fitting_line(x_range, line_coefs, plot_scale)
    if loss_type == 'divergences':
        plt.ylabel('Divergences (log scale)')
    elif loss_type == 'KE_Ps':    
        plt.ylabel(r'KE = $\frac{1}{2}\|\|dP\|\|_2^2 (log scale)$')
    elif loss_type == 'FIDs':    
        plt.ylabel('FID (log scale)')    
    
    '''
    if type(dt) == list : # decaying dt
        plt.xlim([dt[0]*save_iter, sum(dt)])
    else: # constant dt
        plt.xlim([dt*save_iter, epochs*dt])
    '''
    plt.xlim([dt[0]*save_iter, epochs*dt[0]])
    if ylims != None:
        plt.ylim(ylims)    
    if exp_alias_ != None:
        plt.legend()
    
    plt.tight_layout()
    f = filename.split('.pickle')
    plt.savefig(f[0]+"-"+loss_type+".png",bbox_inches='tight')
    
        
    show_or_close_figure(show)
    
def plot_multiple_losses(loss_type, exp_alias_, filepath, colors=None, plot_scale='semilogy', fitting_line=False, save_iter = 1, iter_nos = None, epochs=0, ylims=None, physical_time=True, show=False):
# from one file, plot a designated type of loss (loss_type) 
# iter_nos = [t_1, t_2, t_3,...] marks dots on the loss value of chosen epochs
    print(f'plot multiple {loss_type}')
    
    # load data
    import os
    import re
    
    if not re.search("/$", filepath):
        filepath = filepath+"/"
    r = re.compile(".*pickle")
    directories = [x for x in os.listdir(filepath) if not (re.search("(png$|gif$|^\.)", x))]
    directories.sort()
    
    if type(colors)==type(None):
        colors = [None]*len(directories[1:])
        
    for j, directory in enumerate(directories[1:]):
        print(directory)
        filenames = list(filter(r.match, os.listdir(filepath+directory)))
        filenames = ["/".join((filepath, directory, x)) for x in filenames]
        
        
        for i, filename in enumerate(filenames):
            
            param, result = load_pickle(filename)        
            if epochs==0:
                epochs = param['epochs']
            else:
                param['epochs'] = epochs
            if epochs > 100:
                save_iter = param['save_iter']
            else:
                save_iter = 1
            
            if 'lr_Ps' in param.keys():
                dt = handle_dt(param['lr_P'], lr_Ps=param['lr_Ps'], physical_time=physical_time)
            else:
                dt = handle_dt(param['lr_P'], lr_Ps=None, physical_time=physical_time)
                
            for x in result[loss_type]:
                if np.isnan(x):
                    x = 0
            #if np.isnan(result[loss_type]).any():
            #    print(f'{filename} loss diverged')
            #    result[loss_type] = len(result[loss_type])*[0]
            if i == 0:
                loss_states = np.array(result[loss_type])
            else:
                loss_states = loss_states + np.array(result[loss_type])
        
            if exp_alias_ != None:
                print(param[exp_alias_[i]])
                exp_alias = '%s' % (param[exp_alias_[i]])
        
        if fitting_line == True:
            x_range, line_coefs = fit_line(epochs, loss_states, plot_scale, save_iter, dt)
            
        if loss_type != 'FIDs':
            save_iter = 1
        loss_states = loss_states[:int(epochs/save_iter)]
        print("Last: %s = %f" % (loss_type, loss_states[-1]))
        
        # plot loss
        plot_loss(loss_states, epochs, physical_time, plot_scale, dt=dt, save_iter=save_iter, exp_alias=exp_alias, linewidth=2, color=colors[j])
    
        # mark specific points
        if iter_nos != None:
            if type(dt) == list : # decaying dt
                x_val = [sum(dt[:iter_no]) for iter_no in iter_nos]
            else: # constant dt
                x_val = [iter_no*dt for iter_no in iter_nos]
            
            plt.plot(x_val, [abs(loss_states[x-1]) for x in iter_nos], '.', color='red')
            
        # fitting line
        if fitting_line == True:
            plot_fitting_line(x_range, line_coefs, plot_scale)
        if loss_type == 'divergences':
            plt.ylabel('Divergences (log scale)', fontsize="16")
        elif loss_type == 'KE_Ps':    
            plt.ylabel(r'Kinetic energy (log scale)', fontsize="16")
        elif loss_type == 'FIDs':    
            plt.ylabel('FID (log scale)', fontsize="16")
    
    if type(dt) == list : # decaying dt
        plt.xlim([dt[0]*save_iter, sum(dt)])
    else: # constant dt
        plt.xlim([dt*save_iter, epochs*dt]) 
    if ylims != None:
        plt.ylim(ylims)    
    if exp_alias_ != None:
        plt.legend(fontsize="16")
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=14)
    
    plt.tight_layout()
    plt.savefig(filepath+loss_type+".png",bbox_inches='tight')
        
    show_or_close_figure(show)
    
# ------------------------------
# Show performance of generator
# ------------------------------
def plot_generated_sample(generated_samples, Y_, X_, Y_current, proj_axes, it, filename, show=False):
    color_code = generated_samples[:,proj_axes[0]]**2 + generated_samples[:,proj_axes[1]]**2
    
    plt.scatter(Y_[:,proj_axes[0]], Y_[:,proj_axes[1]], alpha=0.3, c=color_code, label='initial particles')
    plt.scatter(X_[:,proj_axes[0]], X_[:,proj_axes[1]], alpha=0.3, label='target particles')
    plt.scatter(Y_current[:,proj_axes[0]], Y_current[:,proj_axes[1]], alpha=0.5, label='Current GPA particles')
    plt.scatter(generated_samples[:,proj_axes[0]], generated_samples[:,proj_axes[1]], alpha=0.5, c=color_code,  label='Current generated particles')
    plt.title(f'Iteration: {it}')
    plt.legend()
    plt.tight_layout()
    
    f = filename.split('.pickle')
    plt.savefig(f[0]+"-generated_samples_it%d.png" % it,bbox_inches='tight')
    
    show_or_close_figure(show)
    
def plot_more_generated_sample(generated_samples, X_, proj_axes, it, filename, Y_=None, show=False):
    color_code = generated_samples[:,proj_axes[0]]**2 + generated_samples[:,proj_axes[1]]**2
    
    if type(Y_) != type(None):
        plt.scatter(Y_[:,proj_axes[0]], Y_[:,proj_axes[1]], alpha=0.3, c = color_code, label='initial particles')
    plt.scatter(X_[:,proj_axes[0]], X_[:,proj_axes[1]], alpha=0.3, label='target particles')
    plt.scatter(generated_samples[:,proj_axes[0]], generated_samples[:,proj_axes[1]], alpha=0.5, c = color_code, label='Current generated particles')
    plt.title(f'Iteration: {it}')
    plt.legend()
    plt.tight_layout()
    
    f = filename.split('.pickle')
    plt.savefig(f[0]+"-more_generated_samples_it%d.png" % it,bbox_inches='tight')
    
    show_or_close_figure(show)

# -------------------------------------------
def plot_result(filename, data = []):
    param, result, X_, Y_, trajectories, lr_P, dataset, r_param, N_samples_P, N_samples_Q, epochs, save_iter, X_label, Y_label, N_dim = read_pickle_data(filename)
    
    

def plot_result(filename, intermediate=False, epochs = 0, iter_nos = None, data = [], show=False):
    if intermediate == True:
        trajectories = data['trajectories']
        divergences = data['divergences']
        wass1s = data['wasserstein1s']
        KE_Ps = data['KE_Ps']
        FIDs = data['FIDs']
        X_ = data['X_']
        Y_ = data['Y_']
        X_label = data['X_label']
        Y_label = data['Y_label']
        dt = data['dt']
        save_iter = data['save_iter']
        # dataset = data['dataset']
        # r_param = data['r_param']
        # vectorfields = data['vectorfields']
        f = filename.split('.pickle')
        filename_ = f[0] + "_%depochs" % epochs + f[1]
        
    else:
        trajectories, divergences, KE_Ps, FIDs, X_, Y_, X_label, Y_label, dt, save_iter = None, None, None, None, None, None,  None, None, None, 1
        wass1s = None
        filename_ = filename
        epochs = 0
        
    ## 2D image data
    if ('MNIST' in filename  or 'CIFAR10' in filename) and ".pickle" in filename:
        iter_nos = [0, 1000, 2000, 10000, 20000]
        
        if intermediate == False:
            #images_to_animation(trajectories=trajectories,dt=dt, physical_time=True, pick_samples = None, epochs=epochs, save_gif=True, filename = filename_, show = show)
            if 'all' in filename or 'cond' in filename: # (un)conditional gpa
                plot_tiled_images(print_multiplier=26, samples=None, sample_label=Y_label, epochs = epochs, filename=filename_, show = show)
                plot_tiled_images(print_multiplier=26, samples=None, sample_label=Y_label, epochs = -1, filename=filename_, show = show)
        else:
            if 'all' in filename or 'cond' in filename: # (un)conditional gpa
                plot_tiled_images(print_multiplier=10, samples=trajectories[-1], sample_label=Y_label, epochs = epochs, filename=filename_, show = show)
                
        plot_trajectories_img(X_=X_,Y_=Y_, trajectories = trajectories, dt = dt, pick_samples=None, epochs=epochs, save_iter = save_iter, iter_nos = iter_nos, physical_time=True, filename=filename_, show = show)
        plot_trained_img(X_ = X_, trajectories = trajectories, pick_samples=None, epochs=epochs, filename=filename_, show = show)
        
        plot_losses(loss_type='divergences', loss_states=divergences, plot_scale='semilogy', fitting_line=False, save_iter = 1, dt = dt, iter_nos = None, exp_alias_=None, epochs=epochs, ylims=None, physical_time=True, filename=filename_, show = show)
        plot_losses(loss_type='KE_Ps', loss_states=KE_Ps, plot_scale='semilogy', fitting_line=False, save_iter = 1, dt = dt, iter_nos = None, exp_alias_=None, epochs=epochs, ylims=None, physical_time=True, filename=filename_, show = show)
        #plot_losses(loss_type='FIDs', loss_states=FIDs, plot_scale='semilogy', fitting_line=False, save_iter = save_iter, dt = dt, iter_nos = None, exp_alias_=None, epochs=epochs, ylims=None, physical_time=True, filename=filename_, show = show)
        
            
    ## 2D embedded in high dimensions examples
    elif 'submnfld' in filename and ".pickle" in filename:
        iter_nos = None
        exp_alias_ = None
        track_velocity = True
        iscolor = True
        quantile = True
        physical_time = True
        
        if intermediate == False:
            plot_initial_data(proj_axes = [5,6], x_lim = [None,None],y_lim = [None,None], filename=filename, show = show)
            dataset, r_param, vectorfields = None, None, []
        else:
            dataset = data['dataset']
            r_param = data['r_param']
            vectorfields = data['vectorfields']
            
        plot_trajectories(trajectories=trajectories, dt=dt, X_=X_, Y_=Y_, dataset = dataset, r_param=r_param, vectorfields = vectorfields, proj_axes = [5,6], pick_samples =None, epochs = 0, iter_nos = None, physical_time=physical_time, save_iter = save_iter, track_velocity=track_velocity, arrow_scale = 1, iscolor=iscolor, quantile=quantile, exp_alias_ = exp_alias_, x_lim = [None,None],y_lim = [None,None],  filename = filename_, show = show)
        plot_orth_axes_saturation(Y_ = Y_, trajectories = trajectories, dt = dt, save_iter = 1,proj_axes=[5,6], epochs=0, iter_nos=None, physical_time = physical_time, exclude_initial_time = True, filename = filename_, show = show)
        plot_speeds(vectorfields = vectorfields, dt=dt, save_iter=1, plot_scale='semilogy', proj_axes = [5,6], physical_time=physical_time, epochs=0, filename = filename_, show = show)
        plot_losses(loss_type='divergences', loss_states=divergences, dt = dt, plot_scale='semilogy', fitting_line=False, save_iter = 1, iter_nos = None, exp_alias_=None, epochs=0, ylims=None, physical_time=physical_time, filename=filename_, show = show)
        plot_losses(loss_type='KE_Ps', loss_states=KE_Ps, dt = dt, plot_scale='semilogy', fitting_line=False, save_iter = 1, iter_nos = None, exp_alias_=None, epochs=0, ylims=None, physical_time=physical_time, filename=filename_, show = show)
      
    
    ## low dimensional example
    elif  ".pickle" in filename:
        iter_nos = None
        exp_alias_ = None
        track_velocity = False
        iscolor = False
        quantile = False
        
        proj_axes = [0,1]
        if 'student_t' in filename:
            x_lim = [-30, 30]
            y_lim = [-30, 30]
        else:
            x_lim = [None, None]
            y_lim = [None, None]
    
        if intermediate == False:
            dataset, r_param, vectorfields = None, None, []
            
            if '3D' in filename:
                z_lim = [None, None]
                disp_angle = [8, -85, 0] # 3D Swiss roll
                #disp_angle = [30, 15,0]
                trajectories_to_animation3D(x_lim=x_lim, y_lim=y_lim, z_lim=z_lim, disp_angle = disp_angle, trajectories=None, N_samples_P=None, dt=None, physical_time=True, epochs=epochs, quantile = quantile, dataset=dataset, r_param = r_param, save_gif=True, filename = filename, show = show)
            else:
                trajectories_to_animation(physical_time=True, epochs=epochs, quantile = quantile, dataset=dataset, r_param=r_param, x_lim=x_lim, y_lim=y_lim, save_gif=True, filename = filename, show = show)
            plot_initial_data(proj_axes = [0,1], x_lim = x_lim, y_lim = y_lim, filename=filename_, show = show)
            plot_output_target(proj_axes = [0,1], x_lim = x_lim,y_lim = y_lim, filename=filename_, show = show)
        else:
            dataset = data['dataset']
            r_param = data['r_param']
            vectorfields = data['vectorfields']
            
            phi = data['phi']
            W = data['W']
            b = data['b']
            NN_par = data['NN_par']
            
            
        ## plots for 1D example
        if "1D" in filename:
            plot_density_1D(X_ = X_, Y_ = Y_, trajectories = trajectories, dt = dt, save_iter = 1,proj_axes=proj_axes, epochs=epochs, iter_nos=None, physical_time = True, exclude_initial_time = False, exclude_target = False, filename = filename_, show = show)
            plot_f_lip_div_and_wass1(loss_states=divergences, wass1s = wass1s, plot_scale='semilogy', save_iter = 1, dt = dt, epochs=epochs, ylims=None, physical_time=True, filename=filename_, show=show)
        if "3D_Swiss_roll" in filename:
            proj_axes = [0,2]
            
        plot_trajectories(trajectories=trajectories, dt=dt, X_=X_, Y_=Y_, dataset = dataset, r_param=r_param, vectorfields = vectorfields, proj_axes = proj_axes, pick_samples =None, epochs = epochs, iter_nos = iter_nos, physical_time=True, save_iter = save_iter, track_velocity=track_velocity, arrow_scale = 1, iscolor=iscolor, quantile=quantile, exp_alias_ = exp_alias_, x_lim = x_lim, y_lim = y_lim, filename = filename_, show = show)
        plot_losses(loss_type='divergences', loss_states=divergences, plot_scale='semilogy',dt = dt,  fitting_line=False, save_iter = 1, iter_nos = None, exp_alias_=None, epochs=epochs, ylims=None, physical_time=True, filename=filename_, show = show)
        plot_losses(loss_type='KE_Ps', loss_states=KE_Ps, plot_scale='semilogy', dt = dt, fitting_line=False, save_iter = 1,  iter_nos = None, exp_alias_=None, epochs=epochs, ylims=None, physical_time=True, filename=filename_, show = show)
       
    
       
    # ------------------------------
    # Multiple experiments in one plot
    elif ('MNIST' in filename  or 'CIFAR10' in filename):
        iter_nos = None
        exp_alias_ = ['N_latent_dim','N_latent_dim', 'N_dim']
        
        plot_multiple_trajectories_img(filename, exp_alias_, proj_axes = [0,1], pick_samples =None, epochs = 0, iter_nos = None, physical_time=True, save_iter = save_iter, show = show)
        plot_multiple_losses(loss_type='divergences', exp_alias_ = exp_alias_, filepath = filename_, plot_scale='semilogy', fitting_line=False, save_iter = 1, iter_nos = None, epochs=0, ylims=None, physical_time=True, show = show)
        plot_multiple_losses(loss_type='KE_Ps', exp_alias_ = exp_alias_, filepath = filename_, plot_scale='semilogy', fitting_line=False, save_iter = 1, iter_nos = None, epochs=0, ylims=None, physical_time=True, show = show)
        plot_multiple_losses(loss_type='FIDs', exp_alias_ = exp_alias_, filepath = filename_, plot_scale='semilogy', fitting_line=False, save_iter = save_iter, iter_nos = None, epochs=0, ylims=None, physical_time=True, show = show)
        
    else:
        iter_nos = None#[0, 20, 100, 200]
        exp_alias_ = ['L',]*4
        colors = None
        track_velocity = True
        iscolor = False
        quantile = True
        epochs=0
        proj_axes = [0,1]
        
        if 'student_t' in filename:
            x_lim = [-30, 30]
            y_lim = [-30, 30]
        else:
            x_lim = [None, None]
            y_lim = [None, None]
            
        #plot_trajectories(trajectories=trajectories, dt=dt, X_=X_, Y_=Y_, dataset = dataset, r_param=r_param, vectorfields = vectorfields, proj_axes = proj_axes, pick_samples =None, epochs = epochs, iter_nos = iter_nos, physical_time=True, save_iter = save_iter, track_velocity=track_velocity, arrow_scale = 1, iscolor=iscolor, quantile=quantile, exp_alias_ = exp_alias_, x_lim = x_lim, y_lim = y_lim, filename = filename_, show = show)
        
        if "1D" in filename_:
            plot_multiple_f_lip_div_and_wass1(exp_alias_, filename_, plot_scale='semilogy', save_iter = 1, epochs=0, ylims=None, physical_time=True, show=False)
        if "3D_Swiss_roll" in filename:
            proj_axes = [0,2]
        if "Labeled_disease" in filename:
            colors = [[0.0, 0.5, 0.6], [0.9, 0.35, 0.45]]
        
        plot_multiple_trajectories(filename_, exp_alias_, proj_axes = proj_axes, pick_samples =None, epochs = 0, iter_nos = iter_nos, physical_time=True, save_iter = save_iter, track_velocity=False, arrow_scale = 1, iscolor=iscolor, quantile=quantile, x_lim = x_lim, y_lim = y_lim, show = show)
        plot_multiple_losses(loss_type='divergences', colors=colors, exp_alias_ = exp_alias_, filepath = filename_, plot_scale='semilogy', fitting_line=False, save_iter = 1, iter_nos = None, epochs=0, ylims=[0.00001, 100], physical_time=True, show = show)
        plot_multiple_losses(loss_type='KE_Ps', colors=colors, exp_alias_ = exp_alias_, filepath = filename_, plot_scale='semilogy', fitting_line=False, save_iter = 1, iter_nos = None, epochs=0, ylims=None, physical_time=True, show = show)
        
    
    

# plot data from loading pickled files
if __name__ == "__main__":
    if len(argv) == 2:
        filename = argv[1]
    else:
        print('Put filename for argv[1]!')
    plot_result(filename, intermediate=False, epochs = 0, iter_nos = None, show = True)
