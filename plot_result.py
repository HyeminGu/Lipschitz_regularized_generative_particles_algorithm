#!/usr/bin/env python3
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sys import platform, argv

def show_or_close_figure():
    if "linux" in platform:
        plt.clf()
        plt.close()
    else:
        plt.show() 

def load_pickle(filename):
    with open(filename, "rb") as fr:
        param, result = pickle.load(fr)
    return param, result
    
# -----------------------------------------
# General drawing functions
# ----------------------------------------- 
def calculate_time_steps(dt, iter_nos, physical_time=True):
# calculate real_time / iteration count
    if physical_time == True:
        iter_nos_new = []
        if type(dt) == list: # decaying or varying dt
            for i, iter_no in enumerate(iter_nos):
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
    elif dataset == 'Mixture_of_student_t':
        centers=[(0, 0), (10,0), (0,10), (10,10)]
    elif dataset == 'Mixture_of_student_t_submnfld':
        centers=[(-10, -10), (10,-10), (-10,10), (10,10)]
        
    circles1, circles2 = [], []
    for c in centers:
        circles1.append(plt.Circle(tuple(c), r1, color='k', linestyle='-', fill=False))
        circles2.append(plt.Circle(tuple(c), r2, color='k', linestyle='--', fill=False))
        ax.add_patch(circles1[-1])
        ax.add_patch(circles2[-1])
    return 0
        
def plot_arrow(ax, x, dx, scale=1):
# plot vectorfield of chosen samples
    if type(dx) != np.ndarray:
        return
    kwargs = {'color':'lawngreen'}
    ax.arrow(x=x[0], y=x[1], dx=-dx[0]*scale, dy=-dx[1]*scale, width=0.1, overhang=.5, **kwargs)
    

# -----------------------------------------
# Individual figures
# -----------------------------------------    

# Scatter plot for given data
def plot_initial_data(X_=None, Y_=None, proj_axes = [0,1], x_lim = [None,None],y_lim = [None,None], filename=None):
# plot X_ : target data, Y_ : initial data in 2D projected plane
    if filename != None:
        param, result = load_pickle(filename)
        X_ = param['X_']
        try: 
            Y_ = param['Y_']
        except:
            pass
            
    X_ = proj_and_sample(X_, proj_axes)
    Y_ = proj_and_sample(Y_, proj_axes)
    
    plt.scatter(X_[:, 0], X_[:, 1], label="target X")
    plt.scatter(Y_[:, 0], Y_[:, 1], label="initial Y")
    
    xlims = set_axis_lim([X_[:,0], Y_[:,0]], mask=1, lb=x_lim[0], ub=x_lim[1])
    ylims = set_axis_lim([X_[:,1], Y_[:,1]], mask=1, lb=y_lim[0], ub=y_lim[1])
    plt.xlim(xlims)
    plt.ylim(ylims)
    
    plt.legend()
    plt.tight_layout()
    
    if filename != None:
        f = filename.split('.pickle')
        plt.savefig(f[0]+"-initial_data.png")
    
    show_or_close_figure()
 
# Scatter plots for time trajectories
def plot_trajectories(trajectories=None, dt=None, X_=None, Y_=None, r_param=None, vectorfields = [], mobilities = [], proj_axes = [0,1], pick_samples =None, epochs = 0, iter_nos = None, physical_time=True, save_iter = 1, track_velocity=False, arrow_scale = 1, iscolor=False, quantile=True, exp_alias_ = None, x_lim = [None,None],y_lim = [None,None], filename = None):
# plot trajectories of one file
# exp_alias_: one of the keys in param dictionary as a string '...' (to specify title)
    print('plot_trajectories')
    # make frames
    if iter_nos == None:
        n_figs = 4
    else:
        n_figs = len(iter_nos)
    f, axs = plt.subplots(nrows=1, ncols=n_figs, figsize=(15, 3.5))  
    
    # load pickled data
    if filename != None:
        param, result = load_pickle(filename)
        if epochs == 0:
            epochs = param['epochs']
        if epochs > 100:
            save_iter = param['save_iter']                
        X_ = param['X_']
        try:
            Y_ = param['Y']
        except:
            pass
        trajectories = [Y_] + result['trajectories'] 
        try:
            vectorfields = result['vectorfields']
        except:
            pass
        try:
            mobilities = param['mobilities']
        except:
            mobilities = [1]*int(epochs/save_iter)
            
        if physical_time == True:
            dt = param['lr_P']
            if 'lr_Ps' in param.keys():
                if abs(param['lr_Ps'][-1] - dt) > 1e-8:
                    dt = param['lr_Ps'] 
        else:
            dt = 1    

        if quantile == True:
            if 'gaussian' in param['dataset']:
                 r_param = param['sigma_Q']
            elif 'student_t' in param['dataset']:
                r_param = param['nu']
            elif param['dataset'] == 'Extension_of_gaussian':
                r_param = param['a']   
                       
        if exp_alias_ != None:
            exp_alias = '%s=\n%s' % (exp_alias_, param[exp_alias_])
    
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
            vectorfields = [mobilities[int(i/save_iter)] * vectorfields[int(i/save_iter)] for i in iter_nos_]
    
    if iscolor == True:
        cv_max, cv_min = 0, 1e+16
        for vf in vectorfields:
            cv_max = max(cv_max, max(np.linalg.norm(vf, axis=1)))
            cv_min = min(cv_min, min(np.linalg.norm(vf, axis=1)))  
            c_map = [np.linalg.norm(vectorfields[i] , axis=1) for i in range(len(vectorfields))]  
    if track_velocity == True:
        dP = [proj_and_sample(x, proj_axes, pick_samples) for x in vectorfields]
    
    # draw plots
    for i, ax in enumerate(axs):
        if physical_time == True:
            ax.set_title('T=%.3f'% time_steps[i])
        else:
            ax.set_title(f'T={iter_nos[i]}')
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
    
    if filename != None:
        f = filename.split('.pickle')
        plt.savefig(f[0]+"-trajectories.png")
    
    show_or_close_figure()
            

# Scatter plots for time trajectories with multiple experimental parameters(exp_alias_)
def plot_multiple_trajectories(filepath, exp_alias_, proj_axes = [0,1], pick_samples =None, epochs = 0, iter_nos = None, physical_time=True, save_iter = 1, track_velocity=False, arrow_scale = 1, iscolor=False, quantile=True, x_lim = [None,None],y_lim = [None,None]):
# plot trajectories of one file
# exp_alias_: one of the keys in param dictionary as a string '...' (to specify title)
    # load filenames
    import os
    import re

    r = re.compile(".*pickle")
    filepath = filepath+'/!/'
    filenames = list(filter(r.match, os.listdir(filepath)))
    filenames = [filepath+x for x in filenames]
    
    print('plot_multiple_trajectories')
    
    # make frames
    if iter_nos == None:
        n_figs = 4
    else:
        n_figs = len(iter_nos)
    f, axs = plt.subplots(nrows=len(filenames), ncols=n_figs, figsize=(15, 7))  
    
    
    
    for axs_row, filename in zip(axs, filenames):
        print(filename)
    
        # load pickled data
        param, result = load_pickle(filename)
        if epochs == 0:
            epochs = param['epochs']
        if epochs > 100:
            save_iter = param['save_iter']                
        X_ = param['X_']
        try:
            Y_ = param['Y']
        except:
            pass
        trajectories = [Y_] + result['trajectories'] 
        try:
            vectorfields = result['vectorfields']
        except:
            pass
        try:
            mobilities = param['mobilities']
        except:
            mobilities = [1]*int(epochs/save_iter)
            
        if physical_time == True:
            dt = param['lr_P']
            if 'lr_Ps' in param.keys():
                if abs(param['lr_Ps'][-1] - dt) > 1e-8:
                    dt = param['lr_Ps'] 
        else:
            dt = 1    

        if quantile == True:
            if 'gaussian' in param['dataset']:
                 r_param = param['sigma_Q']
            elif 'student_t' in param['dataset']:
                r_param = param['nu']
            elif param['dataset'] == 'Extension_of_gaussian':
                r_param = param['a']   
                       
        exp_alias = '%s=\n%s' % (exp_alias_, param[exp_alias_])
    
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
                vectorfields = [mobilities[int(i/save_iter)] * vectorfields[int(i/save_iter)] for i in iter_nos_]
        
        if iscolor == True:
            cv_max, cv_min = 0, 1e+16
            for vf in vectorfields:
                cv_max = max(cv_max, max(np.linalg.norm(vf, axis=1)))
                cv_min = min(cv_min, min(np.linalg.norm(vf, axis=1)))  
                c_map = [np.linalg.norm(vectorfields[i] , axis=1) for i in range(len(vectorfields))]  
        if track_velocity == True:
            dP = [proj_and_sample(x, proj_axes, pick_samples) for x in vectorfields]
        
        
        axs_row[0].text(-0.1, 0.5, exp_alias, size=15, transform=axs_row[0].transAxes, horizontalalignment='right')
        
        # draw plots in the same row
        for i, ax in enumerate(axs_row):
            if physical_time == True:
                ax.set_title('T=%.3f'% time_steps[i])
            else:
                ax.set_title(f'T={iter_nos[i]}')
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
    
    plt.savefig(filepath+"trajectories.png")
    
    show_or_close_figure()
    

# -----------------------------------------------
# one-label image trajectories
def plot_trajectories_img(X_ = None, Y_=None, trajectories = None, dt = None, pick_samples=None, epochs=0, save_iter = 1, iter_nos = None, physical_time=True, filename=None):
# plot trajectories of 2D image data of one file
    # exp_alias_: one of the keys in param dictionary as a string '...'    
    
    # load data
    if filename != None:
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
        
        if physical_time == True:
            dt = param['lr_P']
            if 'lr_Ps' in param.keys():
                if abs(param['lr_Ps'][-1] - dt) > 1e-8:
                    dt = param['lr_Ps'] 
        else:
            dt = 1    
    
    # make frames
    if iter_nos == None:
        n_figs = 4
        iter_nos = [int(epochs*x/(n_figs-1)) for x in range(n_figs)] 
        if type(Y_) !=np.ndarray and iter_nos[0] == 0:
            iter_nos[0] = save_iter
    else:
        n_figs = len(iter_nos)
    f, axs = plt.subplots(nrows=1, ncols=n_figs+1)
    
    # pick an image
    if pick_samples == None:
        pick_samples = np.random.randint(N_samples_P)
        
    # determine time steps and plot certain time step trajectories iteratively
    trajectories = [Y_] + trajectories
    trajectories = [trajectories[int(i/save_iter)] for i in iter_nos] 
    trajectories = [X_] + trajectories
    
    time_steps = ['Target']+calculate_time_steps(dt, iter_nos, physical_time=physical_time)
     
    for i, ax in enumerate(axs):
        if physical_time == True and i>0:  
            ax.set_title('T=%.3f'% time_steps[i])
        else:
            ax.set_title(f'T={time_steps[i]}')
        ax.imshow(trajectories[i][pick_samples],interpolation='nearest', vmin=-0.0, vmax=1.0)
        ax.axis('off')
    plt.tight_layout()
    
    f = filename.split('.pickle')
    plt.savefig(f[0]+"-trajectories.png")
    
    show_or_close_figure()
    
def plot_multiple_trajectories_img(filepath, exp_alias_, proj_axes = [0,1], pick_samples =None, epochs = 0, iter_nos = None, physical_time=True, save_iter = 1):
# plot trajectories of one file
# exp_alias_: one of the keys in param dictionary as a string '...' (to specify title)
    # load filenames
    import os
    import re

    r = re.compile(".*pickle")
    filepath = filepath+'/!/'
    filenames = list(filter(r.match, os.listdir(filepath)))
    filenames = [filepath+x for x in filenames]
    
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
       
        if physical_time == True:
            dt = param['lr_P']
            if 'lr_Ps' in param.keys():
                if abs(param['lr_Ps'][-1] - dt) > 1e-8:
                    dt = param['lr_Ps'] 
        else:
            dt = 1    
            
        N_samples_P = param['N_samples_P']       
        if pick_samples == None:
            pick_sample = np.random.randint(N_samples_P)
                       
        exp_alias = '%s=\n%s' % (exp_alias_[0], param[exp_alias_[0]])
    
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
    
    plt.savefig(filepath+"trajectories.png")
    
    show_or_close_figure()
       
    
def plot_trained_img(X_ = None, trajectories = None, pick_samples=None, epochs=0, filename=None):
# plot target and final trajectories of 2D image data 
    # make frames
    if pick_samples == None:
        n_figs = 4
    else:
        n_figs = len(pick_samples)
    f, axs = plt.subplots(nrows=2, ncols=n_figs)
    
    # load data
    if filename != None:
        param, result = load_pickle(filename)
        X_ = param['X_']
        trajectories = result['trajectories']
        N_samples_P = param['N_samples_P']
    if pick_samples == None:
        pick_samples = list(np.random.randint(N_samples_P, size=n_figs))
    
    axs[0,0].text(-0.1, 0.3, 'Target', size=15, transform=axs[0,0].transAxes, horizontalalignment='right') 
    axs[1,0].text(-0.1, 0.3, 'Learned', size=15, transform=axs[1,0].transAxes, horizontalalignment='right') 
    
    idx = np.random.randint(0, N_samples_P, n_figs)
    for i in range(n_figs):
        axs[0,i].imshow(X_[idx[i]], interpolation='nearest', vmin=-0.0, vmax=1.0)
        axs[0,i].axis('off')
        axs[1,i].imshow(trajectories[-1][idx[i]], interpolation='nearest', vmin=-0.0, vmax=1.0)
        axs[1,i].axis('off')
    
    plt.tight_layout()
    f = filename.split('.pickle')
    plt.savefig(f[0]+"-learned.png")
    
    show_or_close_figure()
    

# multi-label image tiles for conditional gpa 
def plot_tiled_images(print_multiplier, last_traj=None, last_digit=None, epochs = 0, data = None, data_label=None, filename=None):
# plot several 2D images from designated epoch from one conditional gpa/gan experiment and show a tiled plot
# epochs = 0: last trajectory, -1: target images, ##: ##'th trajectory
# data_label: N_samples x num_classes one-hot encoded
    print('plot_tiled_images')
    
    # load pickled data
    if filename != None:
        param, result = load_pickle(filename)
        if epochs == 0:
            epochs = param['epochs']
        save_iter = param['save_iter']
            
        if epochs == -1:  
            data, data_label = param['X_'], param['X_label'] # one-hot encoding label
        else:
            data, data_label = result['trajectories'][int(epochs/save_iter)-1], param['data_label'] 

        
    num_classes = np.shape(data_label)[1]
    for i in range(num_classes):
        i_idx = np.squeeze(np.where(data_label[:,i] ==1))
        data = np.squeeze(data)    
        i_data = data[i_idx[:print_multiplier]]
        samples = i_data.transpose(1,0,2)
        
        
        newrows = np.reshape(samples, (samples.shape[0], samples.shape[1]*samples.shape[2]))
        if i == 0:
            rows = newrows
        else:
            rows = np.concatenate((rows, newrows), axis=0)    
    plt.imshow(rows, interpolation='nearest', vmin=-0.0, vmax=1.0)    
    plt.axis('off')
    plt.tight_layout()
    
    f = filename.split('.pickle')
    plt.savefig(f[0]+"-tiled_image.png")
    
    show_or_close_figure()
    
def images_to_animation(trajectories=None, N_samples_P=None, dt=None, physical_time=True, pick_samples = None, epochs=0, save_gif=True, filename = None):
    import matplotlib.animation as animation
    
    # load data
    if filename != None:
        param, result = load_pickle(filename)
        trajectories = result['trajectories']
        N_samples_P = param['N_samples_P']
        
        if epochs==0:
            epochs = param['epochs']
        else:
            param['epochs'] = epochs
        save_iter = param['save_iter']
        
        if physical_time == True:
            dt = param['lr_P']
            if 'lr_Ps' in param.keys():
                if abs(param['lr_Ps'][-1] - dt) > 1e-8:
                    dt = param['lr_Ps'] 
        else:
            dt = 1    
            
    trajectories = result['trajectories'][:int(epochs/save_iter)]
    if pick_samples == None:
        pick_samples = np.random.randint(N_samples_P)
        
    iter_nos = list(range(save_iter, save_iter+epochs, save_iter))
    time_steps = calculate_time_steps(dt, iter_nos, physical_time=physical_time)
        
    # make a frame
    fig, ax = plt.subplots()
    
    ims = []
    for i, x in enumerate(trajectories):
        im = ax.imshow(x[pick_samples], interpolation='nearest', vmin=-0.0, vmax=1.0)
        ttl = ax.text(0.5,1.05, "t = {}".format(time_steps[i]), bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},transform=ax.transAxes, ha="center")
        ims.append([im, ttl])
        
    ani = animation.ArtistAnimation(fig, ims, interval=300, blit=False, repeat_delay=500)
   
    if save_gif:
        writergif = animation.PillowWriter(fps=3) 
        f = filename.split('.pickle')
        ani.save(f[0]+"movie.gif", writer=writergif)        
    
    show_or_close_figure()
    
# -------------------------------------
# 12D embedded examples
def plot_orth_axes_saturation(N_dim = None, Y_ = None, trajectories = None, save_iter = 1,dt = None, proj_axes=[5,6], epochs=0, iter_nos=None, physical_time = True, filename=None):
# plot density of values in the orthogonal axes: concentration to 0 is better
    print(f'plot_orth_axes_saturation {filename}')
    from scipy.stats import gaussian_kde
    
    # load data
    if filename != None:
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
        
        if physical_time == True:
            dt = param['lr_P']
            if 'lr_Ps' in param.keys():
                if abs(param['lr_Ps'][-1] - dt) > 1e-8:
                    dt = param['lr_Ps'] 
        else:
            dt = 1    
            
        N_dim = param['N_dim']
    try:
        n_figs = len(iter_nos)
    except:
        n_figs = 4
    if iter_nos == None:
        iter_nos = [int(epochs*x/(n_figs-1)) for x in range(n_figs)] 
        if type(Y_) !=np.ndarray and iter_nos[0] == 0:
            iter_nos[0] = save_iter
    trajectories = [Y_] + trajectories[:int(epochs/save_iter)]  
    trajectories = [trajectories[int(i/save_iter)] for i in iter_nos]
    
    orth_axes = list(set(range(N_dim))-set(proj_axes))
    time_steps = calculate_time_steps(dt, iter_nos, physical_time=physical_time)
    for i in range(n_figs):
        complement_data = proj_and_sample(trajectories[i], orth_axes, pick_samples=None)
        complement_data = complement_data.flatten()
    
        x= np.linspace(complement_data.min(), complement_data.max(), 1000)
        z = gaussian_kde(complement_data)(x)
        plt.plot(x, z, linestyle='-', label='T = {}'.format(time_steps[i]) ) 
    plt.tight_layout()
    plt.legend()
    f = filename.split('.pickle')
    plt.savefig(f[0]+"-orthogonal.png")
    
    show_or_close_figure()
    
def plot_multiple_orth_axes_saturation(filepath, proj_axes=[5,6], epochs=0, iter_nos=None, physical_time = True):
# plot density of values in the orthogonal axes: concentration to 0 is better
    from scipy.stats import gaussian_kde
    
    # load data
    import os
    import re
    
    r = re.compile(".*pickle")
    filenames = list(filter(r.match, os.listdir(filepath)))
    filenames = [filepath+x for x in filenames]
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
    plt.savefig(f[0]+"-orthogonal.png")
    
    show_or_close_figure()
    

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
    
def plot_loss(loss_states, epochs, physical_time, plot_scale, dt, exp_alias=None, save_iter=1):
# plot setting for the certain type of loss (loss_states) with(out) fitting lines
# plot_scale = 'semilogy' or 'loglog'
    from numpy import arange
    if physical_time == True:
        xlabel_name = 'Time'
    else:
        xlabel_name = 'Iteration'
    
    if type(dt) == list : # decaying dt
        from numpy import cumsum
        x_val = cumsum(dt[0:epochs+save_iter:save_iter])
    else: # constant dt
        x_val = arange(save_iter,epochs+save_iter, save_iter)*dt
    if exp_alias == None:
        if plot_scale == "semilogy":
            plt.semilogy(x_val, [abs(x) for x in loss_states], '-')
        if plot_scale == "loglog":
            plt.loglog(x_val, [abs(x) for x in loss_states], '-')
            xlabel_name = xlabel_name+ ' (log scale)'
    else:
        if plot_scale == "semilogy":
            plt.semilogy(x_val, [abs(x) for x in loss_states], '-', label=exp_alias)
        if plot_scale == "loglog":
            plt.loglog(x_val, [abs(x) for x in loss_states], '-', label=exp_alias)
            xlabel_name = xlabel_name+ ' (log scale)'
    plt.xlabel(xlabel_name)
    

    
# -------------------------------------
# Loss plot
# -------------------------------------     
# for 12D submnfld data
def plot_speeds(vectorfields =  None, mobilities = None, N_dim = None, save_iter=1, dt=None, plot_scale='semilogy', proj_axes = [5,6], physical_time=True, epochs=0, filename = None):
    print('plot speed trajectories decomposed to the projected plane and the orthogonal complement')
    
    if filename != None:
        param, result = load_pickle(filename)        
        if epochs==0:
            epochs = param['epochs']
        else:
            param['epochs'] = epochs
        save_iter = param['save_iter']
        
        if physical_time == True:
            dt = param['lr_P']
            if 'lr_Ps' in param.keys():
                if abs(param['lr_Ps'][-1] - dt) > 1e-8:
                    dt = param['lr_Ps'] 
        else:
            dt = 1  
        vectorfields = result['vectorfields']
        try:
            mobilities = result['mobilities']
            for i in range(len(vectorfields)):
                vectorfields[i] = mobilities[i]*vectorfields[i]   
        except:
            pass                   
        N_dim = param['N_dim']
        
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
        plt.xlim([dt[0]*save_iter, sum(dt)*save_iter]) 
    else: # constant dt
        plt.xlim([dt*save_iter, epochs*dt]) 
    plt.legend()    
    plt.tight_layout()
    f = filename.split('.pickle')
    plt.savefig(f[0]+"-separated_speeds.png")
    
    show_or_close_figure()
    
# -----------------------------------
# plot generic loss
# -----------------------------------
def plot_losses(loss_type, loss_state=None, plot_scale='semilogy', fitting_line=False, save_iter = 1, dt = None, iter_nos = None, exp_alias_=None, epochs=0, ylims=None, physical_time=True, filename=None):
# from one file, plot a designated type of loss (loss_type) 
# iter_nos = [t_1, t_2, t_3,...] marks dots on the loss value of chosen epochs
    print(f'plot {loss_type}')
    
    # load data
    if filename != None:
        param, result = load_pickle(filename)        
        if epochs==0:
            epochs = param['epochs']
        else:
            param['epochs'] = epochs
        if epochs > 100:
            save_iter = param['save_iter']
        else:
            save_iter = 1
        
        if physical_time == True:
            dt = param['lr_P']
            if 'lr_Ps' in param.keys():
                if abs(param['lr_Ps'][-1] - dt) > 1e-8:
                    dt = param['lr_Ps'] 
        else:
            dt = 1  
        if np.isnan(result[loss_type]).any():
            print(f'{filename} loss diverged')   
            result[loss_type] = len(result[loss_type])*[0] 
        loss_states = np.array(result[loss_type])
        
        if exp_alias_ != None:
            exp_alias = '%s=%s' % (exp_alias_, param[exp_alias_])
        
    if fitting_line == True:
        x_range, line_coefs = fit_line(epochs, loss_states, plot_scale, save_iter, dt)
        
    if loss_type != 'FIDs':
        save_iter = 1
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
    
    if type(dt) == list : # decaying dt
        plt.xlim([dt[0]*save_iter, sum(dt)*save_iter]) 
    else: # constant dt
        plt.xlim([dt*save_iter, epochs*dt]) 
    if ylims != None:
        plt.ylim(ylims)    
    if exp_alias_ != None:
        plt.legend()
    
    plt.tight_layout()
    f = filename.split('.pickle')
    plt.savefig(f[0]+"-"+loss_type+".png")
    
        
    show_or_close_figure()
    
def plot_multiple_losses(loss_type, exp_alias_, filepath, plot_scale='semilogy', fitting_line=False, save_iter = 1, iter_nos = None, epochs=0, ylims=None, physical_time=True):
# from one file, plot a designated type of loss (loss_type) 
# iter_nos = [t_1, t_2, t_3,...] marks dots on the loss value of chosen epochs
    print(f'plot multiple {loss_type}')
    
    # load data
    import os
    import re
    
    r = re.compile(".*pickle")
    directories = os.listdir(filepath)
    for directory in directories[1:]:
        filenames = list(filter(r.match, os.listdir(filepath+'/'+directory)))
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
            
            if physical_time == True:
                dt = param['lr_P']
                if 'lr_Ps' in param.keys():
                    if abs(param['lr_Ps'][-1] - dt) > 1e-8:
                        dt = param['lr_Ps'] 
            else:
                dt = 1  
            if np.isnan(result[loss_type]).any():
                print(f'{filename} loss diverged')   
                result[loss_type] = len(result[loss_type])*[0] 
            if i == 0:
                loss_states = np.array(result[loss_type])
            else:
                loss_states = loss_states + np.array(result[loss_type])
        
            if exp_alias_ != None:
                exp_alias = '%s=%s' % (exp_alias_[i], param[exp_alias_[i]])
        
        if fitting_line == True:
            x_range, line_coefs = fit_line(epochs, loss_states, plot_scale, save_iter, dt)
            
        if loss_type != 'FIDs':
            save_iter = 1
        loss_states = loss_states[:int(epochs/save_iter)]
        print("Last: %s = %f" % (loss_type, loss_states[-1]))
        
        # plot loss
        plot_loss(loss_states, epochs, physical_time, plot_scale, dt=dt, save_iter=save_iter, exp_alias=exp_alias)    
    
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
    
    if type(dt) == list : # decaying dt
        plt.xlim([dt[0]*save_iter, sum(dt)*save_iter]) 
    else: # constant dt
        plt.xlim([dt*save_iter, epochs*dt]) 
    if ylims != None:
        plt.ylim(ylims)    
    if exp_alias_ != None:
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(filepath+loss_type+".png")
        
    show_or_close_figure()
    

# plot data from loading pickled files
if __name__ == "__main__":
    if len(argv) == 2:
        file = argv[1]
        if ".pickle" in file:
            filename = file
            filepath = ""
        else:
            filepath = file
            filename = ""
    
    ## 2D image data
    if 'MNIST' in filename or 'CIFAR10' in filename:
        epochs = 0
        iter_nos = None
        
        #images_to_animation(trajectories=None, N_samples_P=None, dt=None, physical_time=True, pick_samples = None, epochs=epochs, save_gif=True, filename = filename)
        plot_trajectories_img(X_ = None,Y_=None, trajectories = None, dt = None, pick_samples=None, epochs=epochs, iter_nos = iter_nos, physical_time=True, filename=filename)
        plot_trained_img(X_ = None, trajectories = None, pick_samples=None, epochs=0, filename=filename)
        
        plot_losses(loss_type='divergences', loss_state=None, plot_scale='semilogy', fitting_line=False, save_iter = 1, dt = None, iter_nos = None, exp_alias_=None, epochs=0, ylims=None, physical_time=True, filename=filename)
        plot_losses(loss_type='KE_Ps', loss_state=None, plot_scale='semilogy', fitting_line=False, save_iter = 1, dt = None, iter_nos = None, exp_alias_=None, epochs=0, ylims=None, physical_time=True, filename=filename)
        plot_losses(loss_type='FIDs', loss_state=None, plot_scale='semilogy', fitting_line=False, save_iter = 1, dt = None, iter_nos = None, exp_alias_=None, epochs=0, ylims=None, physical_time=True, filename=filename)
        
        if 'all' in filename: # conditional gpa
            plot_tiled_images(print_multiplier=10, last_traj=None, last_digit=None, epochs = 0, data = None, data_label=None, filename=filename)
            
    ## 2D embedded in high dimensions examples 
    elif 'submnfld' in filename:
        iter_nos = None
        exp_alias_ = None
        track_velocity = True
        iscolor = True
        quantile = True
        
        plot_initial_data(proj_axes = [5,6], x_lim = [None,None],y_lim = [None,None], filename=filename)
        plot_trajectories(trajectories=None, dt=None, X_=None, Y_=None, r_param=None, vectorfields = [], mobilities = [], proj_axes = [5,6], pick_samples =None, epochs = 0, iter_nos = None, physical_time=True, save_iter = 1, track_velocity=track_velocity, arrow_scale = 1, iscolor=iscolor, quantile=quantile, exp_alias_ = exp_alias_, x_lim = [None,None],y_lim = [None,None],  filename = filename)
        plot_orth_axes_saturation(N_dim = None, Y_ = None, trajectories = None, save_iter = 1,dt = None, proj_axes=[5,6], epochs=0, iter_nos=None, physical_time = True, filename = filename) 
        plot_speeds(vectorfields =  None, mobilities = None, N_dim = None, save_iter=1, dt=None, plot_scale='semilogy', proj_axes = [5,6], physical_time=True, epochs=0, filename = filename)
        plot_losses(loss_type='divergences', loss_state=None, plot_scale='semilogy', fitting_line=False, save_iter = 1, dt = None, iter_nos = None, exp_alias_=None, epochs=0, ylims=None, physical_time=True, filename=filename)
        plot_losses(loss_type='KE_Ps', loss_state=None, plot_scale='semilogy', fitting_line=False, save_iter = 1, dt = None, iter_nos = None, exp_alias_=None, epochs=0, ylims=None, physical_time=True, filename=filename)
    
    ## low dimensional example
    elif filename != "" :  
        iter_nos = None
        exp_alias_ = None
        track_velocity = False
        iscolor = False
        quantile = True
    
        plot_initial_data(proj_axes = [0,1], x_lim = [None,None],y_lim = [None,None], filename=filename)
        plot_trajectories(trajectories=None, dt=None, X_=None, Y_=None, r_param=None, vectorfields = [], mobilities = [], proj_axes = [0,1], pick_samples =None, epochs = 0, iter_nos = None, physical_time=True, save_iter = 1, track_velocity=track_velocity, arrow_scale = 1, iscolor=iscolor, quantile=quantile, exp_alias_ = exp_alias_, x_lim = [None,None],y_lim = [None,None],  filename = filename)
        plot_losses(loss_type='divergences', loss_state=None, plot_scale='semilogy', fitting_line=False, save_iter = 1, dt = None, iter_nos = None, exp_alias_=None, epochs=0, ylims=None, physical_time=True, filename=filename)
        plot_losses(loss_type='KE_Ps', loss_state=None, plot_scale='semilogy', fitting_line=False, save_iter = 1, dt = None, iter_nos = None, exp_alias_=None, epochs=0, ylims=None, physical_time=True, filename=filename)
        
    # -----------------------------------------
    elif 'MNIST' in filepath or 'CIFAR10' in filepath:
        epochs = 0
        iter_nos = None
        exp_alias_ = ['N_latent_dim','N_latent_dim', 'N_dim']
        
        plot_multiple_trajectories_img(filepath, exp_alias_, proj_axes = [0,1], pick_samples =None, epochs = 0, iter_nos = None, physical_time=True, save_iter = 1)
        plot_multiple_losses(loss_type='divergences', exp_alias_ = exp_alias_, filepath = filepath, plot_scale='semilogy', fitting_line=False, save_iter = 1, iter_nos = None, epochs=0, ylims=None, physical_time=True)
        plot_multiple_losses(loss_type='KE_Ps', exp_alias_ = exp_alias_, filepath = filepath, plot_scale='semilogy', fitting_line=False, save_iter = 1, iter_nos = None, epochs=0, ylims=None, physical_time=True)
        plot_multiple_losses(loss_type='FIDs', exp_alias_ = exp_alias_, filepath = filepath, plot_scale='semilogy', fitting_line=False, save_iter = 1, iter_nos = None, epochs=0, ylims=None, physical_time=True)
        
    else:
        iter_nos = None
        exp_alias_ = None
        track_velocity = True
        iscolor = True
        quantile = True
        
        plot_multiple_trajectories(filepath, exp_alias_, proj_axes = [0,1], pick_samples =None, epochs = 0, iter_nos = None, physical_time=True, save_iter = 1, track_velocity=False, arrow_scale = 1, iscolor=False, quantile=True, x_lim = [None,None],y_lim = [None,None])