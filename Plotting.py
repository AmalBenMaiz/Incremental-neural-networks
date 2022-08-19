# Import libraries
import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import pandas as pd
                    
databases= ['fetal_health_NSP', 'glass', 'hand', 'iris', 'wine', 'zoo']
path_method={'TLFN': 'results/TLFN', 'I-ELM': 'results/I-ELM', 'EM-ELM': 'results/EM-ELM'}
l_dim_tlfn=['loss', 'accuracy', 'Training Time']
l_dim=['nodes', 'loss', 'accuracy', 'Training Time']
mode=['train', 'test']

L_regions = {'1': '1subregions', '3': '3subregions', '5': '5subregions', '10': '10subregions'} #subregions
qt_factor = {'5': '5qt', '10': '10qt', '25': '25qt', '100': '100qt'} 



def extract_data(l_df, dim):
    data = []
    for df in l_df:
        v_min, v_max = df[dim].quantile(0.167), df[dim].quantile(0.833)
        df = df[(df[dim]>=v_min)&(df[dim]<=v_max)][dim]
        data.append(df)
    return data

def method_dfs (path):
    l_df_method = [f for f in listdir(path) if isfile(join(path,f))] #all dataframes of method
    return l_df_method

def plot(data, label, xlabel, dim, fig_titl, mode, method):
    fig, ax = plt.subplots(figsize = (6,5))
    bp = ax.boxplot(data, patch_artist = True, notch ='True', vert = 1)
     
    colors = ['#0000FF', '#00FF00','#FFFF00', '#FF00FF', 'pink', 'lightblue', 'lightgreen']
     
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
     
    # changing color and linewidth of whiskers
    for whisker in bp['whiskers']:
        whisker.set(color ='#8B008B',
                    linewidth = 1.5,
                    linestyle =":")
     
    # changing color and linewidth of caps
    for cap in bp['caps']:
        cap.set(color ='#8B008B',
                linewidth = 1)
     
    # changing color and linewidth of medians
    for median in bp['medians']:
        median.set(color ='red', linewidth = 2)
     
    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker ='D',
                  color ='#e7298a',
                  alpha = 0.3)
         
    # x-axis labels
    ax.set_xticklabels(label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(dim)
    
     
    # Adding title
    if mode == 'train':
        titl={'nodes': 'Epochs', 'loss': 'Training loss', 'accuracy': 'Training accuracy', 'Training Time': 'Training time (sec)'}
    else:
        titl={'loss': 'Testing loss', 'accuracy': 'Testing accuracy'}
    
    plt.title(titl[dim])
     
    # Removing top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    fig.savefig(f'results/figures/{method}/{mode}.{fig_titl}.pdf')
        

def plotting_elm(dim, mode, method): #mode= 'train' or 'test'; method = EM-ELM or I-ELM
    path = path_method[method]
    l_df_method = method_dfs (path)# all dataframes of em-elm
    
    l_df_read=[]
    l_df_mode = [s for s in l_df_method if mode in s]# list of train or test dataframes
    for f in l_df_mode:
        file=pd.read_csv(path + '/' + f)
        l_df_read.append(file)
    
    label= databases
    xlabel='Databases'
    data_m = extract_data(l_df_read, dim)
    fig_titl= '_' + dim 
    plot(data_m, label, xlabel, dim, fig_titl, mode, method)
    
    
#plotting evaluation terms of TLFN with choosen L-T conbination
def plotting_TLFN(mode, method='TLFN'):
    path = path_method[method]
    l_df_method = method_dfs (path)# all dataframes of TLFN
   
    label= databases
    xlabel='Databases'

    if mode=='train':
        l_df_read=[]
        file_mode= [s for s in l_df_method if mode in s]
        l_dim=['nodes', 'loss', 'accuracy', 'Training Time']
        L=10
        T=100     
        ref= L_regions[f'{L}'] + '_' + qt_factor[f'{T}']
        l_df_mode = [s for s in file_mode if ref in s]
        for f in l_df_mode:
            file=pd.read_csv(path + '/' + f)
            l_df_read.append(file)
        for dim in l_dim:
            data_m = extract_data(l_df_read, dim)
            fig_titl= '_' + dim 
            plot(data_m, label, xlabel, dim, fig_titl, mode, method= 'TLFN_agg')
            
    else:
        file_mode= [s for s in l_df_method if mode in s]
        l_dim=['loss', 'accuracy']
        L=1
        
        for dim in l_dim:
            l_df_read=[]
            if dim=='loss':
                T=100
            else:
                T=25
            ref= L_regions[f'{L}'] + '_' + qt_factor[f'{T}']
            l_df_mode = [s for s in file_mode if ref in s]
            for f in l_df_mode:
                  file=pd.read_csv(path + '/' + f)
                  l_df_read.append(file)
            data_m = extract_data(l_df_read, dim)
            fig_titl= '_' + dim 
            plot(data_m, label, xlabel, dim, fig_titl, mode, method= 'TLFN_agg')
        
    
#plot tlfn with all L-T combination     
def plotting_tlfn (dim, mode, method='TLFN'):
   
    path = path_method[method]
    l_df_method = method_dfs(path)# all dataframes of tlfn
    l_value=[1, 3, 5, 10] 
    qt_value=[5, 10, 25, 100]
    for dt in databases:
        for L in l_value:
            l_df_read=[]
            label=[]
            l_df_mode=[]
            for T in qt_value:
                ref = mode + '_' + dt + '_' + L_regions[f'{L}'] + '_' + qt_factor[f'{T}']
                dff=[s for s in l_df_method if ref in s]
                l_df_mode.append (dff[0])# list of train dataframes
                lbl= '(' + f'{L}' + 'L*' + f'{T}' + 'T)'
                label.append(lbl)
            
            for f in l_df_mode:
                file=pd.read_csv(path + '/' + f)
                l_df_read.append(file)
                    
            data_m = extract_data(l_df_read, dim)
            xlabel='L subregions and T quantizer factor _' + f'{dt}'
            fig_titl= '_' + dim + '_' + f'{L}' + 'subregions for each T_' + f'{dt}'
            plot(data_m, label, xlabel, dim, fig_titl, method, mode)
      
def aggregate_tlfn(mode):
    L_regions = {'1': '1subregions', '3': '3subregions', '5': '5subregions', '10': '10subregions'} #subregions
    qt_factor = {'5': '5qt', '10': '10qt', '25': '25qt', '100': '100qt'} 
    path = path_method['TLFN']
    l_df_method = method_dfs(path)# all dataframes of tlfn
    l_value=[1, 3, 5, 10] 
    qt_value=[5, 10, 25, 100]
    for dt in databases:
        l_df = []
        for L in l_value:
            for T in qt_value:
                ref = mode + '_' + dt + '_' + L_regions[f'{L}'] + '_' + qt_factor[f'{T}']
                file=[s for s in l_df_method if ref in s][0]
                df = pd.read_csv(path + f'/{file}')
                del df['sim']
                df['ref'] = f'{L}-{T}'
                df = df.groupby(['ref']).median().reset_index()
                l_df.append(df)
            df_agg = pd.concat(l_df)
            df_agg.to_csv(f'results/TLFN/aggregate/{mode}_{dt}.csv',index=None)

#extraction of median values of all (L-T) Combinaisions
 for m in mode: 
     aggregate_tlfn(m)

#plot evaluation of TLFN model
 for m in mode: 
     plotting_TLFN(m)
    
#plot evaluation of I-ELM or EM-ELM model   
 for dim in l_dim:
     if (dim == 'Training Time') or (dim == 'nodes'):
         m='train'
         plotting_elm(dim, m, 'I-ELM')
     else:
         for m in mode:
             plotting_elm(dim, m, 'I-ELM')
        
    
   
   




