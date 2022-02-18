import numpy as np
import pandas as pd
from scipy import stats
idx = pd.IndexSlice
from os import getcwd,listdir
import matplotlib.pyplot as plt
from random import shuffle, sample, seed
import seaborn as sns
from random import seed, sample, shuffle
from matplotlib.lines import Line2D
import ipywidgets as wid
from other_functions import *

def all_non_consecutive(arr):
    ans = []
    start = arr[0]
    index = arr[0]
    end = arr[0]
    for number in arr:
        if index == number:
            index += 1
            end = number
        else:
            ans.append({'start': start, 'end': end})
            start = number
            index = number + 1
    ans.append({'start':start, 'end':arr[-1]})

    return ans

def partial_shuffle(a, part=0.5):
    '''input array and the fraction you want partially shuffled.
    from eumiro\'s answer at https://stackoverflow.com/questions/8340584/how-to-make-a-random-but-partial-shuffle-in-python'''
    seed(120121)
    # which characters are to be shuffled:
    idx_todo = sample(list(np.arange(0,len(a))), int(len(a) * part))

    # what are the new positions of these to-be-shuffled characters:
    idx_target = idx_todo[:]
    shuffle(idx_target)

    # map all "normal" character positions {0:0, 1:1, 2:2, ...}
    mapper = dict((i, i) for i in np.arange(len(a)))

    # update with all shuffles in the string: {old_pos:new_pos, old_pos:new_pos, ...}
    mapper.update(zip(idx_todo, idx_target))

    # use mapper to modify the string:
    return [a[mapper[i]] for i in np.arange(len(a))]

def supply_curve_plot(df, x_col, stack_cols, ax=0, dividing_line_width=0.2, 
                      price_col='', price_line_width=4,legend=True,legend_fontsize=19,legend_cols=2,
                      title='Cost supply curve',xlabel='Cumulative bauxite production (Mt)',
                      ylabel='Total minesite cost ($/t)',unit_split=True,line_only=False,ylim=(0,71.5),
                      byproduct=False, **kwargs):
    '''Creates a stacked supply curve
    df: dataframe with time index
    x_col: str, name of column in dataframe to use as
      the value along x-axis (not cumulative, the
      function does the cumsum)
    stack_cols: list, the columns comprising the stack,
      their sum creates the supply curve shape
    ax: axis on which to plot
    dividing_line_width: float, linewidth for lines 
      separating the stacks, acts somewhat like shading
    price_col: str, name of column to plot as an additional
      line on the axes
    price_line_width: float, width of the line for price
    legend: bool
    legend_fontsize: float,
    legend_cols: int, number of columns for legend
    title: str
    xlabel: str
    ylabel: str
    unit_split: bool, split to remove units
    line_only: bool, whether to only plot the line
    ylim: tuple, default 0, 71.5
    **kwargs: arguments passed to easy_subplots'''
    if type(ax)==int:
        fig, ax = easy_subplots(1,1,**kwargs)
        ax = ax[0]
    # plt.figure(dpi=300)
    ph = df.copy()
    if len(stack_cols)>1:
        ph.loc[:,'Sort by'] = ph.loc[:,stack_cols].sum(axis=1)
    else:
        ph.loc[:,'Sort by'] = ph.loc[:,stack_cols[0]]
    ph = ph.sort_values('Sort by')
    ph_prod = ph[x_col].cumsum()
    ph.loc[:,'x plot'] = ph_prod
    ph.index=np.arange(2,int(max(ph.index)*2+3),2)
    ph1 = ph.copy().rename(dict(zip(ph.index,[i-1 for i in ph.index])))
    ph1.loc[:,'x plot'] = ph1['x plot'].shift(1).fillna(0)
    
    ph2 = pd.concat([ph,ph1]).sort_index()
    if line_only:
        if byproduct:
            n = 0
            for i in ph2['Byproduct ID'].unique():
                ph4 = ph2.loc[ph2['Byproduct ID']==i]
                ii = all_non_consecutive(ph4.index)
                for j in ii:
                    ph2.loc[j['start']:j['end'],'Cat'] = n
                    n+=1
            colors = dict(zip(np.arange(0,4),['#d7191c','#fdae61','#abd9e9','#2c7bb6']))
            custom_lines = []
            for i in ph2.Cat.unique():
                if i!=ph2.Cat.iloc[-1]:
                    ind = ph2.loc[ph2.Cat==i].index
                    ax.vlines(ph2.loc[ind[-1],'x plot'], ph2.loc[ind[-1],'Sort by'], ph2.loc[ind[-1]+1,'Sort by'],color='k',linewidth=1)
                ax.step(ph2.loc[ph2.Cat==i,'x plot'],ph2.loc[ph2.Cat==i,'Sort by'],color = colors[ph2.loc[ph2.Cat==i,'Byproduct ID'].iloc[0]], label=ph2.loc[ph2.Cat==i,'Byproduct ID'].iloc[0])
                
            for i in np.sort(ph2['Byproduct ID'].unique()):
                custom_lines += [Line2D([0],[0],color=colors[i])]
            ax.legend(custom_lines,np.sort(ph2['Byproduct ID'].unique()))
        else:
            ax.plot(ph2['x plot'], ph2.loc[:,'Sort by'])
    else:
        ax.stackplot(ph2['x plot'],
                 ph2.loc[:,stack_cols].T,
                 labels=[i.split(' (')[0] if unit_split else i for i in stack_cols],
                 colors=['#d53e4f','#f46d43','#fdae61','#fee08b','#e6f598','#abdda4','#66c2a5','#3288bd'])
    if legend and not line_only:
        ax.legend(loc='upper left',fontsize=legend_fontsize,ncol=legend_cols)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if not line_only:
        ax.vlines(ph['x plot'],0,ph['Sort by'],color='k',linewidth=dividing_line_width)
    if len(price_col)>0:
        ax.step(ph1['x plot'],ph[price_col],label=price_col.split(' (')[0],linewidth=price_line_width)
    ax.set_ylim(ylim)
    return ph2

class GeneralizedOperatingMines():
    ''''''
    def __init__(self,byproduct=False,verbose=False):
        ''''''
        self.byproduct = byproduct
        self.verbose = verbose
        self.initialize_hyperparams()
        self.rs = self.hyperparam.Value.random_state
        self.hyperparam.loc['primary_recovery_rate_shuffle_param','Value'] = 1
        
    def plot_relevant_params(self,include=0,exclude=[],plot_recovery_grade_correlation=True, 
                             plot_minesite_supply_curve=True, plot_margin_supply_curve=True,log_scale=False,
                             dontplot=False,byproduct=False):
        mines = self.mines.copy()
        if type(include)==int:
            cols = [i for i in mines.columns if i not in exclude and mines.dtypes[i] not in [object,str]]
        else:
            cols = [i for i in include if mines.dtypes[i] not in [object,str]]
        fig, ax = easy_subplots(len(cols)+plot_recovery_grade_correlation+plot_minesite_supply_curve+plot_margin_supply_curve)
        for i,a in zip(cols,ax):
            colors = ['#d7191c','#fdae61','#abd9e9','#2c7bb6']
            if self.byproduct and i not in ['Byproduct ID','Byproduct cash flow ($M)']:
                colors = [colors[int(i)] for i in mines['Byproduct ID'].unique()]
                try:
                    sns.histplot(mines,x=i,hue='Byproduct ID',palette=colors,bins=50,log_scale=log_scale,ax=a)
                except Exception as e:
                    print(i)
                    print(e)
                a.set(title=i)
            elif self.byproduct and i=='Byproduct cash flow ($M)':
                colors = [colors[int(i)] for i in mines.dropna()['Byproduct ID'].unique()]
                sns.histplot(mines.dropna(),x=i,hue='Byproduct ID',palette=colors,bins=50,log_scale=log_scale,ax=a)
                a.set(title=i)
            else:
                mines[i].plot.hist(ax=a, title=i, bins=50)
            if i=='Recovery rate (%)' and self.hyperparam.loc['primary_rr_negative','Value']:
                a.text(0.05,0.95,'Reset to default,\nnegative values found.\nPrice and grade too low.',
                       va='top',ha='left',transform=a.transAxes)
        
        if plot_recovery_grade_correlation:
            a = ax[-(plot_recovery_grade_correlation+plot_minesite_supply_curve+plot_margin_supply_curve)]
            do_a_regress(mines['Head grade (%)'],mines['Recovery rate (%)'],ax=a)
            a.set(xlabel='Head grade (%)',ylabel='Recovery rate (%)',
                         title=f'Correlation with partial shuffle param\nValue: {self.hyperparam.Value.primary_recovery_rate_shuffle_param}')
            
        if plot_minesite_supply_curve:
            a = ax[-(plot_minesite_supply_curve+plot_margin_supply_curve)]
            minn,maxx = mines['Minesite cost (USD/t)'].min(),mines['Minesite cost (USD/t)'].max()
            minn,maxx = min(minn,mines['Commodity price (USD/t)'].min()),max(maxx,mines['Commodity price (USD/t)'].max())
            supply_curve_plot(mines,'Production (kt)',['Total cash cost (USD/t)'],
                              price_col='Commodity price (USD/t)',width_scale=1.2,line_only=True,ax=a,xlabel='Cumulative production (kt)',
                              ylim=(minn-(maxx-minn)/10, maxx+(maxx-minn)/10),ylabel='Total cash cost (USD/t)',
                              title='Total cash cost supply curve\nMean: {:.2f}'.format(mines['Total cash cost (USD/t)'].mean()),
                              byproduct=byproduct)
            
        if plot_margin_supply_curve:
            a = ax[-(plot_margin_supply_curve)]
            minn,maxx = mines['Total cash margin (USD/t)'].min(),mines['Total cash margin (USD/t)'].max()
            supply_curve_plot(mines,'Production (kt)',['Total cash margin (USD/t)'],
                              width_scale=1.2,line_only=True,ax=a,xlabel='Cumulative production (kt)',
                              ylim=(minn-(maxx-minn)/10, maxx+(maxx-minn)/10),ylabel='Total cash margin (USD/t)',
                              title='Cash margin supply curve\nMean: {:.2f}'.format(mines['Total cash margin (USD/t)'].mean()),
                              byproduct=byproduct)
            a.plot([0,mines['Production (kt)'].sum()],[0,0],'k')
        fig.tight_layout()
        if dontplot:
            plt.close(fig)
        return fig
        
    def initialize_hyperparams(self):
        '''
        for many of the default parameters and their origins, see
        https://countertop.mit.edu:3048/notebooks/SQL/Bauxite-aluminum%20mines.ipynb
        and/or Displacement/04 Presentations/John/Weekly Updates/20210825 Generalization.pptx
        '''
        hyperparameters = pd.DataFrame(np.nan,['primary_commodity_price'],['Value','Notes'])
        
        if 'Parameters for operating mine pool generation, mass':
            hyperparameters.loc['verbose','Value'] = self.verbose
            hyperparameters.loc['byproduct','Value'] = True
            hyperparameters.loc['primary_production','Value'] = 1 # kt
            hyperparameters.loc['primary_production_mean','Value'] = 0.003 # kt
            hyperparameters.loc['primary_production_var','Value'] = 1
            hyperparameters.loc['primary_production_distribution','Value'] = 'lognorm'
            hyperparameters.loc['primary_production_fraction','Value'] = 1
            hyperparameters.loc['primary_ore_grade_mean','Value'] = 0.001
            hyperparameters.loc['primary_ore_grade_var','Value'] = 0.3
            hyperparameters.loc['primary_ore_grade_distribution','Value'] = 'lognorm'
            hyperparameters.loc['primary_cu_mean','Value'] = 0.85
            hyperparameters.loc['primary_cu_var','Value'] = 0.06
            hyperparameters.loc['primary_cu_distribution','Value'] = 'lognorm'
            hyperparameters.loc['primary_payable_percent_mean','Value'] = 0.63
            hyperparameters.loc['primary_payable_percent_var','Value'] = 1.83
            hyperparameters.loc['primary_payable_percent_distribution','Value'] = 'weibull_min'
            hyperparameters.loc['primary_rr_default_mean','Value'] = 13.996
            hyperparameters.loc['primary_rr_default_var','Value'] = 0.675
            hyperparameters.loc['primary_rr_default_distribution','Value'] = 'lognorm'
            
            hyperparameters.loc['primary_rr_alpha','Value'] = 5.3693
            hyperparameters.loc['primary_rr_beta','Value'] = -0.3110
            hyperparameters.loc['primary_rr_gamma','Value'] = -0.3006
            hyperparameters.loc['primary_rr_delta','Value'] = 0
            hyperparameters.loc['primary_rr_epsilon','Value'] = 0
            hyperparameters.loc['primary_rr_theta','Value'] = 0
            hyperparameters.loc['primary_rr_eta','Value'] = 0
            hyperparameters.loc['primary_rr_rho','Value'] = 0
            hyperparameters.loc['primary_rr_negative','Value'] = False

            hyperparameters.loc['primary_recovery_rate_var','Value'] = 0.6056 # default value of 0.6056 comes from the mean across all materials in snl
            hyperparameters.loc['primary_recovery_rate_distribution','Value'] = 'lognorm'
            hyperparameters.loc['primary_recovery_rate_shuffle_param','Value'] = 0.4
            hyperparameters.loc['primary_reserves_mean','Value'] = 11.04953 # these values are from https://countertop.mit.edu:3048/notebooks/SQL/Second%20round%20generalization%20mine%20parameters.ipynb
            hyperparameters.loc['primary_reserves_var','Value'] = 0.902357  # use the ratio between reserves and ore treated in each year, finding lognormal distribution
            hyperparameters.loc['primary_reserves_distribution','Value'] = 'lognorm'
            hyperparameters.loc['primary_reserves_reported','Value'] = 30
            hyperparameters.loc['primary_reserves_reported_basis','Value'] = 'none' # ore, metal, or none basis - ore: mass of ore reported as reserves (SNL style), metal: metal content of reserves reported, none: use the generated values without adjustment
            
            hyperparameters.loc['production_frac_region1','Value'] = 0.2
            hyperparameters.loc['production_frac_region2','Value'] = 0.2
            hyperparameters.loc['production_frac_region3','Value'] = 0.2
            hyperparameters.loc['production_frac_region4','Value'] = 0.2
            hyperparameters.loc['production_frac_region5','Value'] = 1-hyperparameters.loc['production_frac_region1':'production_frac_region4','Value'].sum()

        if 'Parameters for operating mine pool generation, cost':
            hyperparameters.loc['primary_commodity_price','Value'] = 6000 # USD/t
            hyperparameters.loc['primary_minesite_cost_mean','Value'] = 0
            hyperparameters.loc['primary_minesite_cost_var','Value'] = 1
            hyperparameters.loc['primary_minesite_cost_distribution','Value'] = 'lognorm'
            
            hyperparameters.loc['minetype_prod_frac_underground','Value'] = 0.3
            hyperparameters.loc['minetype_prod_frac_openpit','Value'] = 0.7
            hyperparameters.loc['minetype_prod_frac_tailings','Value'] = 0
            hyperparameters.loc['minetype_prod_frac_stockpile','Value'] = 0
            hyperparameters.loc['minetype_prod_frac_placer','Value'] = 1 - hyperparameters.loc['minetype_prod_frac_underground':'minetype_prod_frac_stockpile','Value'].sum()
        
            hyperparameters.loc['primary_minerisk_mean','Value'] = 9.4 # values for copper → ranges from 4 to 20
            hyperparameters.loc['primary_minerisk_var','Value'] = 1.35
            hyperparameters.loc['primary_minerisk_distribution','Value'] = 'norm'
            
            hyperparameters.loc['primary_minesite_cost_regression2use','Value'] = 'linear_112_price_tcm_sx' # options: linear_107, bayesian_108, linear_110_price, linear_111_price_tcm; not used if primary_minesite_cost_mean>0
            hyperparameters.loc['primary_tcm_flag','Value'] = 'tcm' in hyperparameters.Value.primary_minesite_cost_regression2use
            hyperparameters.loc['primary_tcrc_regression2use','Value'] = 'linear_113_reftype' # options: linear_112, linear_112_reftype
            hyperparameters.loc['primary_tcrc_dore_flag','Value'] = False # determines whether the refining process is that for dore or for concentrate
            hyperparameters.loc['primary_sxew_fraction','Value'] = 0 # fraction of primary production coming from sxew mines
            
            hyperparameters.loc['primary_scapex_regression2use','Value'] = 'linear_116_price_cap_sx'
            
            hyperparameters = self.add_minesite_cost_regression_params(hyperparameters)
                        
        if 'Parameters for mine life simulation':
            hyperparameters.loc['primary_oge_s','Value'] = 0.3320346
            hyperparameters.loc['primary_oge_loc','Value'] = 0.757959
            hyperparameters.loc['primary_oge_scale','Value'] = 0.399365
            
            hyperparameters.loc['mine_cu_margin_elas','Value'] = 0.01
            hyperparameters.loc['mine_cost_OG_elas','Value'] = -0.113
            hyperparameters.loc['mine_cost_price_elas','Value'] = 0.125
            hyperparameters.loc['mine_cu0','Value'] = 0.7688729808870376
            hyperparameters.loc['mine_tcm0','Value'] = 14.575211987093567
            hyperparameters.loc['discount_rate','Value'] = 0.10
            hyperparameters.loc['ramp_down_cu','Value'] = 0.4
            hyperparameters.loc['ramp_up_cu','Value'] = 0.4
            hyperparameters.loc['ramp_up_years','Value'] = 3
            hyperparameters.loc['close_price_method','Value']='mean'
            hyperparameters.loc['close_years_back','Value']=3
            hyperparameters.loc['years_for_roi','Value']=10
            hyperparameters.loc['close_probability_split_max','Value']=0.3
            hyperparameters.loc['close_probability_split_mean','Value']=0.5
            hyperparameters.loc['close_probability_split_min','Value']=0.2
            hyperparameters.loc['random_state','Value']=20220208
            
        if 'Parameters for incentive pool':
            hyperparameters.loc['reserve_frac_region1','Value'] = 0.19743337
            hyperparameters.loc['reserve_frac_region2','Value'] = 0.08555446
            hyperparameters.loc['reserve_frac_region3','Value'] = 0.03290556
            hyperparameters.loc['reserve_frac_region4','Value'] = 0.24350115
            hyperparameters.loc['reserve_frac_region5','Value'] = 0.44060546

        if 'Parameters for byproducts' and self.byproduct:
            if 'Parameters for byproduct production and grade':
                hyperparameters.loc['byproduct_pri_production_fraction','Value']   = 0.1
                hyperparameters.loc['byproduct_host3_production_fraction','Value'] = 0
                hyperparameters.loc['byproduct_host2_production_fraction','Value'] = 0.4
                hyperparameters.loc['byproduct_host1_production_fraction','Value'] = 1 - hyperparameters.loc[['byproduct_pri_production_fraction','byproduct_host2_production_fraction','byproduct_host3_production_fraction'],'Value'].sum()

                hyperparameters.loc['byproduct_production','Value'] = 4 # kt
                hyperparameters.loc['byproduct_production_mean','Value'] = 0.03 # kt
                hyperparameters.loc['byproduct_production_var','Value'] = 0.5
                hyperparameters.loc['byproduct_production_distribution','Value'] = 'lognorm'

                hyperparameters.loc['byproduct_host1_grade_ratio_mean','Value'] = 20
                hyperparameters.loc['byproduct_host2_grade_ratio_mean','Value'] = 2
                hyperparameters.loc['byproduct_host3_grade_ratio_mean','Value'] = 10
                hyperparameters.loc['byproduct_host1_grade_ratio_var','Value'] = 1
                hyperparameters.loc['byproduct_host2_grade_ratio_var','Value'] = 1
                hyperparameters.loc['byproduct_host3_grade_ratio_var','Value'] = 1
                hyperparameters.loc['byproduct_host1_grade_ratio_distribution','Value'] = 'norm'
                hyperparameters.loc['byproduct_host2_grade_ratio_distribution','Value'] = 'norm'
                hyperparameters.loc['byproduct_host3_grade_ratio_distribution','Value'] = 'norm'

                hyperparameters.loc['byproduct_pri_ore_grade_mean','Value']   = 0.1
                hyperparameters.loc['byproduct_host1_ore_grade_mean','Value'] = 0.1
                hyperparameters.loc['byproduct_host2_ore_grade_mean','Value'] = 0.1
                hyperparameters.loc['byproduct_host3_ore_grade_mean','Value'] = 0.1
                hyperparameters.loc['byproduct_pri_ore_grade_var','Value']   = 0.3                
                hyperparameters.loc['byproduct_host1_ore_grade_var','Value'] = 0.3                
                hyperparameters.loc['byproduct_host2_ore_grade_var','Value'] = 0.3                
                hyperparameters.loc['byproduct_host3_ore_grade_var','Value'] = 0.3
                hyperparameters.loc['byproduct_pri_ore_grade_distribution','Value']   = 'lognorm'
                hyperparameters.loc['byproduct_host1_ore_grade_distribution','Value'] = 'lognorm'
                hyperparameters.loc['byproduct_host2_ore_grade_distribution','Value'] = 'lognorm'
                hyperparameters.loc['byproduct_host3_ore_grade_distribution','Value'] = 'lognorm'
                
                hyperparameters.loc['byproduct_pri_sxew_fraction','Value']   = 0.5
                hyperparameters.loc['byproduct_host1_sxew_fraction','Value'] = 0.2
                hyperparameters.loc['byproduct_host2_sxew_fraction','Value'] = 0.5
                hyperparameters.loc['byproduct_host3_sxew_fraction','Value'] = 0.5

                hyperparameters.loc['byproduct_pri_cu_mean','Value']   = 0.85
                hyperparameters.loc['byproduct_host1_cu_mean','Value'] = 0.85
                hyperparameters.loc['byproduct_host2_cu_mean','Value'] = 0.85
                hyperparameters.loc['byproduct_host3_cu_mean','Value'] = 0.85
                hyperparameters.loc['byproduct_pri_cu_var','Value']   = 0.06
                hyperparameters.loc['byproduct_host1_cu_var','Value'] = 0.06
                hyperparameters.loc['byproduct_host2_cu_var','Value'] = 0.06
                hyperparameters.loc['byproduct_host3_cu_var','Value'] = 0.06
                hyperparameters.loc['byproduct_pri_cu_distribution','Value']   = 'lognorm'
                hyperparameters.loc['byproduct_host1_cu_distribution','Value'] = 'lognorm'
                hyperparameters.loc['byproduct_host2_cu_distribution','Value'] = 'lognorm'
                hyperparameters.loc['byproduct_host3_cu_distribution','Value'] = 'lognorm'
                
                hyperparameters.loc['byproduct_pri_payable_percent_mean','Value']   = 0.63
                hyperparameters.loc['byproduct_host1_payable_percent_mean','Value'] = 0.63
                hyperparameters.loc['byproduct_host2_payable_percent_mean','Value'] = 0.63
                hyperparameters.loc['byproduct_host3_payable_percent_mean','Value'] = 0.63
                hyperparameters.loc['byproduct_pri_payable_percent_var','Value']   = 1.83
                hyperparameters.loc['byproduct_host1_payable_percent_var','Value'] = 1.83
                hyperparameters.loc['byproduct_host2_payable_percent_var','Value'] = 1.83
                hyperparameters.loc['byproduct_host3_payable_percent_var','Value'] = 1.83
                hyperparameters.loc['byproduct_pri_payable_percent_distribution','Value']   = 'weibull_min'
                hyperparameters.loc['byproduct_host1_payable_percent_distribution','Value'] = 'weibull_min'
                hyperparameters.loc['byproduct_host2_payable_percent_distribution','Value'] = 'weibull_min'
                hyperparameters.loc['byproduct_host3_payable_percent_distribution','Value'] = 'weibull_min'
                
                hyperparameters.loc['byproduct_pri_rr_default_mean','Value']   = 13.996
                hyperparameters.loc['byproduct_host1_rr_default_mean','Value'] = 13.996
                hyperparameters.loc['byproduct_host2_rr_default_mean','Value'] = 13.996
                hyperparameters.loc['byproduct_host3_rr_default_mean','Value'] = 13.996
                hyperparameters.loc['byproduct_pri_rr_default_var','Value']   = 0.675
                hyperparameters.loc['byproduct_host1_rr_default_var','Value'] = 0.675
                hyperparameters.loc['byproduct_host2_rr_default_var','Value'] = 0.675
                hyperparameters.loc['byproduct_host3_rr_default_var','Value'] = 0.675
                hyperparameters.loc['byproduct_pri_rr_default_distribution','Value']   = 'lognorm'
                hyperparameters.loc['byproduct_host1_rr_default_distribution','Value'] = 'lognorm'
                hyperparameters.loc['byproduct_host2_rr_default_distribution','Value'] = 'lognorm'
                hyperparameters.loc['byproduct_host3_rr_default_distribution','Value'] = 'lognorm'
            
            if 'byproduct costs':
                hyperparameters.loc['byproduct_commodity_price','Value'] = 10000 # USD/t
                hyperparameters.loc['byproduct_minesite_cost_mean','Value'] = 0 #USD/t
                hyperparameters.loc['byproduct_minesite_cost_var','Value'] = 1
                hyperparameters.loc['byproduct_minesite_cost_distribution','Value'] = 'lognorm'

                hyperparameters.loc['byproduct_host1_commodity_price','Value'] = 2000
                hyperparameters.loc['byproduct_host2_commodity_price','Value'] = 3000
                hyperparameters.loc['byproduct_host3_commodity_price','Value'] = 1000

                hyperparameters.loc['byproduct_host1_minesite_cost_ratio_mean','Value'] = 20
                hyperparameters.loc['byproduct_host2_minesite_cost_ratio_mean','Value'] = 2
                hyperparameters.loc['byproduct_host3_minesite_cost_ratio_mean','Value'] = 10
                hyperparameters.loc['byproduct_host1_minesite_cost_ratio_var','Value'] = 1
                hyperparameters.loc['byproduct_host2_minesite_cost_ratio_var','Value'] = 1
                hyperparameters.loc['byproduct_host3_minesite_cost_ratio_var','Value'] = 1
                hyperparameters.loc['byproduct_host1_minesite_cost_ratio_distribution','Value'] = 'norm'
                hyperparameters.loc['byproduct_host2_minesite_cost_ratio_distribution','Value'] = 'norm'
                hyperparameters.loc['byproduct_host3_minesite_cost_ratio_distribution','Value'] = 'norm'
            
                hyperparameters.loc['byproduct_host1_sus_capex_ratio_mean','Value'] = 20
                hyperparameters.loc['byproduct_host2_sus_capex_ratio_mean','Value'] = 2
                hyperparameters.loc['byproduct_host3_sus_capex_ratio_mean','Value'] = 10
                hyperparameters.loc['byproduct_host1_sus_capex_ratio_var','Value'] = 1
                hyperparameters.loc['byproduct_host2_sus_capex_ratio_var','Value'] = 1
                hyperparameters.loc['byproduct_host3_sus_capex_ratio_var','Value'] = 1
                hyperparameters.loc['byproduct_host1_sus_capex_ratio_distribution','Value'] = 'norm'
                hyperparameters.loc['byproduct_host2_sus_capex_ratio_distribution','Value'] = 'norm'
                hyperparameters.loc['byproduct_host3_sus_capex_ratio_distribution','Value'] = 'norm'
                
                hyperparameters.loc['byproduct_host1_tcrc_ratio_mean','Value'] = 20
                hyperparameters.loc['byproduct_host2_tcrc_ratio_mean','Value'] = 2
                hyperparameters.loc['byproduct_host3_tcrc_ratio_mean','Value'] = 10
                hyperparameters.loc['byproduct_host1_tcrc_ratio_var','Value'] = 1
                hyperparameters.loc['byproduct_host2_tcrc_ratio_var','Value'] = 1
                hyperparameters.loc['byproduct_host3_tcrc_ratio_var','Value'] = 1
                hyperparameters.loc['byproduct_host1_tcrc_ratio_distribution','Value'] = 'norm'
                hyperparameters.loc['byproduct_host2_tcrc_ratio_distribution','Value'] = 'norm'
                hyperparameters.loc['byproduct_host3_tcrc_ratio_distribution','Value'] = 'norm'
            
                hyperparameters.loc['byproduct_pri_minerisk_mean','Value']   = 9.4 # value for copper → ranges from 4 to 20, cutoffs are enforced
                hyperparameters.loc['byproduct_host1_minerisk_mean','Value'] = 9.4 
                hyperparameters.loc['byproduct_host2_minerisk_mean','Value'] = 9.4 
                hyperparameters.loc['byproduct_host3_minerisk_mean','Value'] = 9.4 
                hyperparameters.loc['byproduct_pri_minerisk_var','Value']   = 1.35
                hyperparameters.loc['byproduct_host1_minerisk_var','Value'] = 1.35
                hyperparameters.loc['byproduct_host2_minerisk_var','Value'] = 1.35
                hyperparameters.loc['byproduct_host3_minerisk_var','Value'] = 1.35
                hyperparameters.loc['byproduct_pri_minerisk_distribution','Value']   = 'norm'
                hyperparameters.loc['byproduct_host1_minerisk_distribution','Value'] = 'norm'
                hyperparameters.loc['byproduct_host2_minerisk_distribution','Value'] = 'norm'
                hyperparameters.loc['byproduct_host3_minerisk_distribution','Value'] = 'norm'

        if 'adding notes':
            # operating pool mass values
            hyperparameters.loc['primary_production','Notes'] = 'Total mine production in whatever units we are using, metal content of the primary (host) commodity'
            hyperparameters.loc['primary_production_mean','Notes'] = 'Mean mine production in whatever units we are using, metal content of the primary (host) commodity'
            hyperparameters.loc['primary_production_var','Notes'] = 'Variance of the mine production in whatever units we are using, metal content of the primary (host) commodity'
            hyperparameters.loc['primary_production_distribution','Notes'] = 'valid stats.distribution distribution name, e.g. lognorm, norm, etc. Should be something with loc and scale parameters'
            hyperparameters.loc['primary_ore_grade_mean','Notes'] = 'mean ore grade (%) for the primary commodity'
            hyperparameters.loc['primary_ore_grade_var','Notes'] = 'ore grade variance (%) for the primary commodity'
            hyperparameters.loc['primary_ore_grade_distribution','Notes'] = 'distribution used for primary commodity ore grade, default lognormal'
            hyperparameters.loc['primary_cu_mean','Notes'] = 'mean capacity utilization of the primary mine (likely not necessary to consider this as just the primary, or at least secondary total CU would be the product of this and the secondary CU)'
            hyperparameters.loc['primary_cu_var','Notes'] = 'capacity utilization of primary mine variance'
            hyperparameters.loc['primary_cu_distribution','Notes'] = 'distirbution for primary mine capacity utilization, default lognormal'
            hyperparameters.loc['primary_payable_percent_mean','Notes'] = 'mean for 100-(payable percent): 0.63 value from https://countertop.mit.edu:3048/notebooks/SQL/Mining%20database%20read.ipynb, section 12.3 Payable Percent, for weibull minimum distribution of 100-(payable percent)'
            hyperparameters.loc['primary_payable_percent_var','Notes'] = 'variance for 100-(payable percent): 1.83 value from https://countertop.mit.edu:3048/notebooks/SQL/Mining%20database%20read.ipynb, section 12.3 Payable Percent, for weibull minimum distribution of 100-(payable percent)'
            hyperparameters.loc['primary_payable_percent_distribution','Notes'] = 'distribution for 100-(payable percent): from https://countertop.mit.edu:3048/notebooks/SQL/Mining%20database%20read.ipynb, section 12.3 Payable Percent, for weibull minimum distribution of 100-(payable percent)'
            hyperparameters.loc['primary_rr_grade_corr_slope','Notes'] = 'slope used for calculating log(100-recovery rate) from log(head grade), referencing slide 41 of 20210825 Generalization.pptx and section 10 Recovery rate, https://countertop.mit.edu:3048/notebooks/SQL/Second%20round%20generalization%20mine%20parameters.ipynb'
            hyperparameters.loc['primary_rr_grade_corr_slope','Notes'] = 'constant used for calculating log(100-recovery rate) from log(head grade), referencing slide 41 of 20210825 Generalization.pptx and section 10 Recovery rate, https://countertop.mit.edu:3048/notebooks/SQL/Second%20round%20generalization%20mine%20parameters.ipynb'
            hyperparameters.loc['primary_recovery_rate_mean','Notes'] = 'mean for 100-(recovery rate): lognormal distribution, using the mean of head grade to calculate the mean recovery rate value with constant standard deviation (average from each material), referencing slide 41 of 20210825 Generalization.pptx and section 10 Recovery rate (https://countertop.mit.edu:3048/notebooks/SQL/Mining%20database%20read.ipynb)as a function of ore grade. '
            hyperparameters.loc['primary_recovery_rate_var','Notes'] = 'variance for 100-(recovery rate): lognormal distribution, using the mean of head grade to calculate the mean recovery rate value with constant standard deviation (average from each material), referencing slide 41 of 20210825 Generalization.pptx and section 10 Recovery rate (https://countertop.mit.edu:3048/notebooks/SQL/Mining%20database%20read.ipynb) as a function of ore grade. '
            hyperparameters.loc['primary_recovery_rate_distribution','Notes'] = 'distribution for (100-recovery rate): lognormal distribution, using the mean of head grade to calculate the mean recovery rate value with constant standard deviation (average from each material), referencing slide 41 of 20210825 Generalization.pptx and section 10 Recovery rate (https://countertop.mit.edu:3048/notebooks/SQL/Mining%20database%20read.ipynb) as a function of ore grade. '
            hyperparameters.loc['primary_recovery_rate_shuffle_param','Notes'] = 'recovery rates are ordered to match the order of grade to retain correlation, then shuffled so the correlation is not perfect. This parameter (called part) passed to the partial shuffle function for the correlation between recovery rate and head grade; higher value = more shuffling'
            hyperparameters.loc['primary_reserves_mean','Notes'] = 'mean for primary reserves divided by ore treated. 11.04953 value from https://countertop.mit.edu:3048/notebooks/SQL/Second%20round%20generalization%20mine%20parameters.ipynb, use the ratio between reserves and ore treated in each year, finding lognormal distribution'
            hyperparameters.loc['primary_reserves_var','Notes'] = 'variance for primary reserves divided by ore treated. 0.902357 value from https://countertop.mit.edu:3048/notebooks/SQL/Second%20round%20generalization%20mine%20parameters.ipynb, use the ratio between reserves and ore treated in each year, finding lognormal distribution'
            hyperparameters.loc['primary_reserves_distribution','Notes'] = 'distribution for primary reserves divided by ore treated. lognormal values from https://countertop.mit.edu:3048/notebooks/SQL/Second%20round%20generalization%20mine%20parameters.ipynb, use the ratio between reserves and ore treated in each year, finding lognormal distribution'
            hyperparameters.loc['primary_reserves_reported','Notes'] = 'reserves reported for that year; tunes the operating mine pool such that its total reserves matches this value. Can be given in terms of metal content or total ore available, just have to adjust the primary_reserves_reported_basis variable in this df. Setting it to zero allows the generated value to be used.'
            hyperparameters.loc['primary_reserves_reported_basis','Notes'] = 'ore, metal, or none basis - ore: mass of ore reported as reserves (SNL style), metal: metal content of reserves reported, none: use the generated values without adjustment'
            
            hyperparameters.loc['production_frac_region1','Notes'] = 'region fraction of global production in 2019'
            hyperparameters.loc['production_frac_region2','Notes'] = 'region fraction of global production in 2019'
            hyperparameters.loc['production_frac_region3','Notes'] = 'region fraction of global production in 2019'
            hyperparameters.loc['production_frac_region4','Notes'] = 'region fraction of global production in 2019'
            hyperparameters.loc['production_frac_region5','Notes'] = 'region fraction of global production in 2019, calculated from the remainder of the other 4'

            # Prices
            hyperparameters.loc['primary_minesite_cost_mean','Notes'] = 'mean minesite cost for the primary commodity. Set to zero to use mine type, risk, and ore grade to generate the cost distribution instead'
            hyperparameters.loc['primary_minesite_cost_var','Notes'] = 'minesite cost variance for the primary commodity. Is not used if the mean value is set to zero'
            hyperparameters.loc['primary_minesite_cost_distribution','Notes'] = 'minesite cost distribution type (e.g. lognormal) for the primary commodity. Is not used if the mean value is set to zero'
            
            hyperparameters.loc['minetype_prod_frac_underground','Notes'] = 'fraction of mines using mine type underground; not used if primary_minesite_cost_mean is nonzero'
            hyperparameters.loc['minetype_prod_frac_openpit','Notes'] = 'fraction of mines using mine type openpit; not used if primary_minesite_cost_mean is nonzero'
            hyperparameters.loc['minetype_prod_frac_tailings','Notes'] = 'fraction of mines using mine type tailings; not used if primary_minesite_cost_mean is nonzero'
            hyperparameters.loc['minetype_prod_frac_stockpile','Notes'] = 'fraction of mines using mine type stockpile; not used if primary_minesite_cost_mean is nonzero'
            hyperparameters.loc['minetype_prod_frac_placer','Notes'] = 'fraction of mines using mine type placer; not used if primary_minesite_cost_mean is nonzero'
        
            hyperparameters.loc['minerisk_mean','Notes'] = 'mean value. Risk indicator is the sum of political, operational, terrorism, and security risks, which range from insignificant (1) to extreme (5). Therefore the minimum value is 4 and the maximum value is 20. Not used if primary_minesite_cost_mean is nonzero'
            hyperparameters.loc['minerisk_var','Notes'] = 'variance. Risk indicator is the sum of political, operational, terrorism, and security risks, which range from insignificant (1) to extreme (5). Therefore the minimum value is 4 and the maximum value is 20. Not used if primary_minesite_cost_mean is nonzero'
            hyperparameters.loc['minerisk_distribution','Notes'] = 'distribution, assumed normal. Risk indicator is the sum of political, operational, terrorism, and security risks, which range from insignificant (1) to extreme (5). Therefore the minimum value is 4 and the maximum value is 20. Not used if primary_minesite_cost_mean is nonzero'
            
            hyperparameters.loc['primary_minesite_cost_regression2use','Notes'] = 'options: linear_107, bayesian_108, linear_110_price, linear_111_price_tcm; not used if primary_minesite_cost_mean>0. First component (linear/bayesian) references regression type, number references slide in Displacement\04 Presentations\John\Weekly Updates\20210825 Generalization.pptx. Inclusion of \'price\' at the end indicates to use the regression that included commodity price. Inclusion of \'tcm\' indicates the regression was performed on total cash margin excl tcrc rather than minesite cost, and determines the primary_minesite_cost_flag'
            hyperparameters.loc['primary_minesite_cost_flag','Notes'] = 'True sets to generate minesite costs and total cash margin using the regressions on minesite cost; False sets to generate the same using the regressions on total cash margin (excl TCRC). Based on whether primary_minesite_cost_regression2use contains str tcm.'
            
            hyperparameters.loc['primary_scapex_slope_capacity','Notes'] = 'slope from regression of sustaining CAPEX on capacity, from 04 Presentations\John\Weekly Updates\20210825 Generalization.pptx, slide 34'
            hyperparameters.loc['primary_scapex_constant_capacity','Notes'] = 'constant from regression of sustaining CAPEX on capacity, from 04 Presentations\John\Weekly Updates\20210825 Generalization.pptx, slide 34'
            
            # Simulation
            hyperparameters.loc['primary_oge_s','Notes'] = 'parameters to generate (1-OGE) for lognormal distribution, found from looking at all mines in https://countertop.mit.edu:3048/notebooks/SQL/Mining%20database%20read.ipynb, section 12.2, where it says \'parameters used for generalization\'.'
            hyperparameters.loc['primary_oge_loc','Notes'] = 'parameters to generate (1-OGE) for lognormal distribution, found from looking at all mines in https://countertop.mit.edu:3048/notebooks/SQL/Mining%20database%20read.ipynb, section 12.2, where it says \'parameters used for generalization\'.'
            hyperparameters.loc['primary_oge_scale','Notes'] = 'parameters to generate (1-OGE) for lognormal distribution, found from looking at all mines in https://countertop.mit.edu:3048/notebooks/SQL/Mining%20database%20read.ipynb, section 12.2, where it says \'parameters used for generalization\'.'
            
            hyperparameters.loc['mine_cu_margin_elas','Notes'] = 'capacity utlization elasticity to total cash margin, current value is approximate result of regression attempts; see slide 42 in Group Research Folder_Olivetti\Displacement\04 Presentations\John\Weekly Updates\20210825 Generalization.pptx',
            hyperparameters.loc['mine_cost_OG_elas','Notes'] = 'minesite cost elasticity to ore grade decline'
            hyperparameters.loc['mine_cost_price_elas','Notes'] = 'minesite cost elasticity to bauxite price'
            hyperparameters.loc['mine_cu0','Notes'] = 'median capacity utlization in 2019, used to determine how mines change CU due to TCM'
            hyperparameters.loc['mine_tcm0','Notes'] = 'median total cash margin in 2019, used to determine how mines change CU due to TCM'
            hyperparameters.loc['discount_rate','Notes'] = 'discount rate (fraction, not percent - 0.1=10%), used for NPV/IRR calculation in mine opening decision'
            hyperparameters.loc['ramp_down_cu','Notes'] = 'capacity utilization during ramp down; float less than 1, default 0.4'
            hyperparameters.loc['ramp_up_cu','Notes'] = 'capacity utilization during ramp up, float less than 1, default 0.4'
            hyperparameters.loc['ramp_up_years','Notes'] = 'number of years allotted for ramp up (currently use total dCAPEX distributed among those years, so shortening would make each year of dCAPEX more expensive), int default 3'
            hyperparameters.loc['close_price_method','Notes'] = 'method used for price expected used for mine closing - mean, max, alonso-ayuso, or probabilistic are supported - if using probabilistic you can adjust the close_probability_split variables'
            hyperparameters.loc['close_years_back','Notes'] = 'number of years to use for rolling mean/max/min values when evaluating mine closing'
            hyperparameters.loc['years_for_roi','Notes'] = 'years for return on investment when evaluating mine opening - number of years of mine life to simulate for IRR calculation'
            hyperparameters.loc['close_probability_split_max','Notes'] = 'for the probabilistic closing method, probability given to the rolling close_years_back max'
            hyperparameters.loc['close_probability_split_mean','Notes'] = 'for the probabilistic closing method, probability given to the rolling close_years_back mean'
            hyperparameters.loc['close_probability_split_min','Notes'] = 'for the probabilistic closing method, probability given to the rolling close_years_back min - make sure these three sum to 1'
            hyperparameters.loc['random_state','Notes'] = 'random state int for sampling'
            hyperparameters.loc['reserve_frac_region1','Notes'] = 'region reserve fraction of global total in 2019'
            hyperparameters.loc['reserve_frac_region2','Notes'] = 'region reserve fraction of global total in 2019'
            hyperparameters.loc['reserve_frac_region3','Notes'] = 'region reserve fraction of global total in 2019'
            hyperparameters.loc['reserve_frac_region4','Notes'] = 'region reserve fraction of global total in 2019'
            hyperparameters.loc['reserve_frac_region5','Notes'] = 'region reserve fraction of global total in 2019'
        
        self.hyperparam = hyperparameters
    
    def update_operation_hyperparams(self):
        hyperparameters = self.hyperparam.copy()
        mines = self.mines.copy()
        hyperparameters.loc['mine_cu0','Value'] = mines['Capacity utilization'].median()
        hyperparameters.loc['mine_tcm0','Value'] = mines['Total cash margin (USD/t)'].median()
        self.hyperparam = hyperparameters.copy()
            
    def add_minesite_cost_regression_params(self, hyperparameters_):
        hyperparameters = hyperparameters_.copy()
        reg2use = hyperparameters.Value.primary_minesite_cost_regression2use
#             log(minesite cost) = alpha + beta*log(commodity price) + gamma*log(head grade) 
#                + delta*(numerical risk) + epsilon*placer (mine type)
#                + theta*stockpile + eta*tailings + rho*underground + zeta*sxew

        if reg2use == 'linear_107': # see slide 107 or 110 left-hand-side table in C:\Users\ryter\Dropbox (MIT)\Group Research Folder_Olivetti\Displacement\04 Presentations\John\Weekly Updates\20210825 Generalization.pptx
            hyperparameters.loc['primary_minesite_cost_alpha','Value'] = 7.4083
            hyperparameters.loc['primary_minesite_cost_beta','Value'] = 0
            hyperparameters.loc['primary_minesite_cost_gamma','Value'] = -1.033
            hyperparameters.loc['primary_minesite_cost_delta','Value'] = 0.0173
            hyperparameters.loc['primary_minesite_cost_epsilon','Value'] = -1.5532
            hyperparameters.loc['primary_minesite_cost_theta','Value'] = 0.5164
            hyperparameters.loc['primary_minesite_cost_eta','Value'] = -0.8997
            hyperparameters.loc['primary_minesite_cost_rho','Value'] = 0.7629
            hyperparameters.loc['primary_minesite_cost_zeta','Value'] = 0
        elif reg2use == 'bayesian_108': # see slide 108 in C:\Users\ryter\Dropbox (MIT)\Group Research Folder_Olivetti\Displacement\04 Presentations\John\Weekly Updates\20210825 Generalization.pptx
            hyperparameters.loc['primary_minesite_cost_alpha','Value'] = 10.4893
            hyperparameters.loc['primary_minesite_cost_beta','Value'] = 0
            hyperparameters.loc['primary_minesite_cost_gamma','Value'] = -0.547
            hyperparameters.loc['primary_minesite_cost_delta','Value'] = 0.121
            hyperparameters.loc['primary_minesite_cost_epsilon','Value'] = -0.5466
            hyperparameters.loc['primary_minesite_cost_theta','Value'] = -0.5837
            hyperparameters.loc['primary_minesite_cost_eta','Value'] = -0.9168
            hyperparameters.loc['primary_minesite_cost_rho','Value'] = 1.4692
            hyperparameters.loc['primary_minesite_cost_zeta','Value'] = 0
        elif reg2use == 'linear_110_price': # see slide 110 right-hand-side table in C:\Users\ryter\Dropbox (MIT)\Group Research Folder_Olivetti\Displacement\04 Presentations\John\Weekly Updates\20210825 Generalization.pptx
            hyperparameters.loc['primary_minesite_cost_alpha','Value'] = 0.5236
            hyperparameters.loc['primary_minesite_cost_beta','Value'] = 0.8453
            hyperparameters.loc['primary_minesite_cost_gamma','Value'] = -0.1932
            hyperparameters.loc['primary_minesite_cost_delta','Value'] = -0.015
            hyperparameters.loc['primary_minesite_cost_epsilon','Value'] = 0
            hyperparameters.loc['primary_minesite_cost_theta','Value'] = 0.2122
            hyperparameters.loc['primary_minesite_cost_eta','Value'] = -0.3076
            hyperparameters.loc['primary_minesite_cost_rho','Value'] = 0.1097
            hyperparameters.loc['primary_minesite_cost_zeta','Value'] = 0
        elif reg2use == 'linear_111_price_tcm': # see slide 111 right-hand-side table in C:\Users\ryter\Dropbox (MIT)\Group Research Folder_Olivetti\Displacement\04 Presentations\John\Weekly Updates\20210825 Generalization.pptx
            hyperparameters.loc['primary_minesite_cost_alpha','Value'] = -2.0374
            hyperparameters.loc['primary_minesite_cost_beta','Value'] = 1.1396
            hyperparameters.loc['primary_minesite_cost_gamma','Value'] = 0.1615
            hyperparameters.loc['primary_minesite_cost_delta','Value'] = 0.0039
            hyperparameters.loc['primary_minesite_cost_epsilon','Value'] = 0.1717
            hyperparameters.loc['primary_minesite_cost_theta','Value'] = -0.2465
            hyperparameters.loc['primary_minesite_cost_eta','Value'] = 0.2974
            hyperparameters.loc['primary_minesite_cost_rho','Value'] = -0.0934
            hyperparameters.loc['primary_minesite_cost_zeta','Value'] = 0
        elif reg2use == 'linear_112_price_sx': # updated total minesite cost see slide 112 left-hand-side table in C:\Users\ryter\Dropbox (MIT)\Group Research Folder_Olivetti\Displacement\04 Presentations\John\Weekly Updates\20210825 Generalization.pptx
            hyperparameters.loc['primary_minesite_cost_alpha','Value'] = 0.4683
            hyperparameters.loc['primary_minesite_cost_beta','Value'] = 0.8456
            hyperparameters.loc['primary_minesite_cost_gamma','Value'] = -0.1924
            hyperparameters.loc['primary_minesite_cost_delta','Value'] = -0.0125
            hyperparameters.loc['primary_minesite_cost_epsilon','Value'] = 0.1004
            hyperparameters.loc['primary_minesite_cost_theta','Value'] = 0.1910
            hyperparameters.loc['primary_minesite_cost_eta','Value'] = -0.3044
            hyperparameters.loc['primary_minesite_cost_rho','Value'] = 0.1288
            hyperparameters.loc['primary_minesite_cost_zeta','Value'] = 0.1285
        elif reg2use == 'linear_112_price_tcm_sx': # updated total cash margin, see slide 112 right-hand-side table in C:\Users\ryter\Dropbox (MIT)\Group Research Folder_Olivetti\Displacement\04 Presentations\John\Weekly Updates\20210825 Generalization.pptx
            hyperparameters.loc['primary_minesite_cost_alpha','Value'] = -2.0096
            hyperparameters.loc['primary_minesite_cost_beta','Value'] = 1.1392
            hyperparameters.loc['primary_minesite_cost_gamma','Value'] = 0.1608
            hyperparameters.loc['primary_minesite_cost_delta','Value'] = 0.0027
            hyperparameters.loc['primary_minesite_cost_epsilon','Value'] = 0.1579
            hyperparameters.loc['primary_minesite_cost_theta','Value'] = -0.2338
            hyperparameters.loc['primary_minesite_cost_eta','Value'] = 0.2975
            hyperparameters.loc['primary_minesite_cost_rho','Value'] = -0.1020
            hyperparameters.loc['primary_minesite_cost_zeta','Value'] = -0.0583
        
        reg2use = hyperparameters.Value.primary_tcrc_regression2use
#             log(tcrc) = alpha + beta*log(commodity price) + gamma*log(head grade) 
#                 + delta*risk + epsilon*sxew + theta*dore (refining type)
        if reg2use == 'linear_113_reftype': # see slide 113 left-hand-side table in C:\Users\ryter\Dropbox (MIT)\Group Research Folder_Olivetti\Displacement\04 Presentations\John\Weekly Updates\20210825 Generalization.pptx
            hyperparameters.loc['primary_tcrc_alpha','Value'] = -2.4186
            hyperparameters.loc['primary_tcrc_beta','Value'] = 0.9314
            hyperparameters.loc['primary_tcrc_gamma','Value'] = -0.0316
            hyperparameters.loc['primary_tcrc_delta','Value'] = 0.0083
            hyperparameters.loc['primary_tcrc_epsilon','Value'] = -0.1199
            hyperparameters.loc['primary_tcrc_theta','Value'] = -2.3439
            hyperparameters.loc['primary_tcrc_eta','Value'] = 0
        elif reg2use == 'linear_113': # see slide 113 upper-right table in C:\Users\ryter\Dropbox (MIT)\Group Research Folder_Olivetti\Displacement\04 Presentations\John\Weekly Updates\20210825 Generalization.pptx
            hyperparameters.loc['primary_tcrc_alpha','Value'] = -2.3363
            hyperparameters.loc['primary_tcrc_beta','Value'] = 0.9322
            hyperparameters.loc['primary_tcrc_gamma','Value'] = -0.0236
            hyperparameters.loc['primary_tcrc_delta','Value'] = 0
            hyperparameters.loc['primary_tcrc_epsilon','Value'] = 0
            hyperparameters.loc['primary_tcrc_theta','Value'] = -2.3104
            hyperparameters.loc['primary_tcrc_eta','Value'] = 0

        reg2use = hyperparameters.Value.primary_scapex_regression2use
        #             log(sCAPEX) = alpha + beta*log(commodity price) + gamma*log(head grade) 
#                 + delta*log(capacity) + epsilon*placer + theta*stockpile + eta*tailings + rho*underground + zeta*sxew
        if reg2use == 'linear_116_price_cap_sx': # see slide 116 right-hand-side table in C:\Users\ryter\Dropbox (MIT)\Group Research Folder_Olivetti\Displacement\04 Presentations\John\Weekly Updates\20210825 Generalization.pptx
            hyperparameters.loc['primary_scapex_alpha','Value'] = -12.5802
            hyperparameters.loc['primary_scapex_beta','Value'] = 0.7334
            hyperparameters.loc['primary_scapex_gamma','Value'] = 0.6660
            hyperparameters.loc['primary_scapex_delta','Value'] = 0.9773
            hyperparameters.loc['primary_scapex_epsilon','Value'] = 0
            hyperparameters.loc['primary_scapex_theta','Value'] = 0
            hyperparameters.loc['primary_scapex_eta','Value'] = 0
            hyperparameters.loc['primary_scapex_rho','Value'] = 0.7989
            hyperparameters.loc['primary_scapex_zeta','Value'] = 0.6115
            
        return hyperparameters
        
    def recalculate_hyperparams(self):
        '''
        With hyperparameters initialized, can then edit any hyperparameters 
        you\'d like then call this function to update any hyperparameters
        that could have been altered by such changes.
        '''
        hyperparameters = self.hyperparam    
        hyperparameters.loc['production_frac_region5','Value'] = 1-hyperparameters.loc['production_frac_region1':'production_frac_region4','Value'].sum()
        hyperparameters.loc['minetype_prod_frac_placer','Value'] = 1 - hyperparameters.loc['minetype_prod_frac_underground':'minetype_prod_frac_stockpile','Value'].sum()
        hyperparameters = self.add_minesite_cost_regression_params(hyperparameters)
        hyperparameters.loc['primary_tcm_flag','Value'] = 'tcm' in hyperparameters.Value.primary_minesite_cost_regression2use
        hyperparameters.loc['primary_rr_negative','Value'] = False
        self.hyperparam = hyperparameters
        
    def generate_production_region(self):
        '''updates self.mines, first function called.'''
        hyperparam = self.hyperparam
        pri_dist = getattr(stats,hyperparam.loc['primary_production_distribution','Value'])
        pri_prod_mean_frac = hyperparam.Value.primary_production_mean / hyperparam.Value.primary_production
        pri_prod_var_frac = hyperparam.Value.primary_production_var / hyperparam.Value.primary_production
        production_fraction = hyperparam.Value.primary_production_fraction
        
        pri_prod_frac_dist = pri_dist.rvs(
            loc=0,
            scale=pri_prod_mean_frac,
            s=pri_prod_var_frac,
            size=int(np.ceil(2/pri_prod_mean_frac)),
            random_state=self.rs)
        mines = pd.DataFrame(
            pri_prod_frac_dist,
            index=np.arange(0,int(np.ceil(2/pri_prod_mean_frac))),
            columns=['Production fraction'])
        mines = mines.loc[mines['Production fraction'].cumsum()<production_fraction,:]
        mines.loc[mines.index[-1]+1,'Production fraction'] = production_fraction-mines['Production fraction'].sum()
        mines.loc[:,'Production (kt)'] = mines['Production fraction']*hyperparam.Value.primary_production
        
        regions = [i for i in hyperparam.index if 'production_frac_region' in i]
        mines.loc[:,'Region'] = np.nan
        for i in regions:
            int_version = int(i.replace('production_frac_region',''))
            ind = mines.loc[(mines.Region.isna()),'Production fraction'].cumsum()
            ind = ind.loc[ind<hyperparam.loc[i,'Value']*production_fraction].index
            mines.loc[ind,'Region'] = int_version
        mines.loc[mines.Region.isna(),'Region'] = int(hyperparam.loc[regions,'Value'].astype(float).idxmax().replace('production_frac_region',''))
        self.mines=mines.copy()
    
    def generate_grade_and_masses(self):
        '''
        Note that the mean ore grade reported will be different than
        the cumulative ore grade of all the ore treated here since
        we are randomly assigning grades and don\'t have a good way
        to correct for this.
        '''
        
        self.assign_mine_types()
        self.mines.loc[:,'Risk indicator'] = self.values_from_dist('primary_minerisk').round(0)
        self.mines.loc[:,'Head grade (%)'] = self.values_from_dist('primary_ore_grade')
        self.mines.loc[:,'Commodity price (USD/t)'] = self.hyperparam.Value.primary_commodity_price

        mines = self.mines.copy()
        
        mines.loc[:,'Capacity utilization'] = self.values_from_dist('primary_cu')
        mines.loc[:,'Production capacity (kt)'] = mines[['Capacity utilization','Production (kt)']].product(axis=1)
        mines.loc[:,'Production capacity fraction'] = mines['Production capacity (kt)']/mines['Production capacity (kt)'].sum()
        mines.loc[:,'Payable percent (%)'] = 100-self.values_from_dist('primary_payable_percent')
        mines.loc[mines['Production fraction'].cumsum()<self.hyperparam.Value.primary_sxew_fraction,'Payable percent (%)'] = 100
        self.mines = mines.copy()
        self.generate_costs_from_regression('Recovery rate (%)')
        mines = self.mines.copy()
        rec_rates = 100-mines['Recovery rate (%)']
        if rec_rates.max()<30 or (rec_rates<0).any():
            rec_rates = 100 - self.values_from_dist('primary_rr_default')
            self.hyperparam.loc['primary_rr_negative','Value'] = True
        mines.loc[mines.sort_values('Head grade (%)').index,'Recovery rate (%)'] = \
            partial_shuffle(np.sort(rec_rates),self.hyperparam.Value.primary_recovery_rate_shuffle_param)
        
        mines.loc[:,'Ore treated (kt)'] = mines['Production (kt)']/(mines['Recovery rate (%)']*mines['Head grade (%)']/1e4)
        mines.loc[:,'Capacity (kt)'] = mines['Ore treated (kt)']/mines['Capacity utilization']
        mines.loc[:,'Paid metal production (kt)'] = mines[
            ['Ore treated (kt)','Capacity utilization','Head grade (%)','Recovery rate (%)','Payable percent (%)']].product(axis=1)/1e6
        mines.loc[:,'Reserves ratio with ore treated'] = self.values_from_dist('primary_reserves')
        
        
        mines.loc[:,'Reserves (kt)'] = mines[['Ore treated (kt)','Reserves ratio with ore treated']].product(axis=1)
        mines.loc[:,'Reserves potential metal content (kt)'] = mines[['Reserves (kt)','Head grade (%)']].product(axis=1)*1e-2
        
        # calibrating reserves to input values if needed
        primary_reserves_reported_basis = self.hyperparam.Value.primary_reserves_reported_basis
        primary_reserves_reported = self.hyperparam.Value.primary_reserves_reported
        if primary_reserves_reported_basis == 'ore' and primary_reserves_reported>0:
            ratio = primary_reserves_reported/mines['Reserves (kt)'].sum()
        elif primary_reserves_reported_basis == 'metal' and primary_reserves_reported>0:
            ratio = primary_reserves_reported/mines['Reserves potential metal content (kt)'].sum()
        else:
            ratio = 1
        mines.loc[:,'Reserves (kt)'] *= ratio
        mines.loc[:,'Reserves potential metal content (kt)'] *= ratio
        self.mines = mines.copy()

    def generate_costs_from_regression(self,param):
        '''Called inside generate_total_cash_margin'''
        mines = self.mines.copy()
        h = self.hyperparam.copy()
        
        if h.Value.primary_minesite_cost_mean > 0 and param=='Minesite cost (USD/t)':
            mines.loc[:,param] = self.values_from_dist('primary_minesite_cost')
        elif param in ['Minesite cost (USD/t)','Total cash margin (USD/t)','Recovery rate (%)']:
#             log(minesite cost) = alpha + beta*log(head grade) + gamma*(head grade) + delta*(numerical risk) + epsilon*placer (mine type)
#                + theta*stockpile + eta*tailings + rho*underground + zeta*sxew
            log_minesite_cost = h.loc['primary_minesite_cost_alpha','Value'] +\
                h.loc['primary_minesite_cost_beta','Value'] * np.log(mines['Commodity price (USD/t)']) +\
                h.loc['primary_minesite_cost_gamma','Value'] * np.log(mines['Head grade (%)']) +\
                h.loc['primary_minesite_cost_delta','Value'] * mines['Risk indicator'] +\
                h.loc['primary_minesite_cost_epsilon','Value'] * (mines['Mine type string']=='placer') +\
                h.loc['primary_minesite_cost_theta','Value'] * (mines['Mine type string']=='stockpile') +\
                h.loc['primary_minesite_cost_eta','Value'] * (mines['Mine type string']=='tailings') +\
                h.loc['primary_minesite_cost_rho','Value'] * (mines['Mine type string']=='underground') +\
                h.loc['primary_minesite_cost_zeta','Value'] * (mines['Payable percent (%)']==100)
            mines.loc[:,param] = np.exp(log_minesite_cost)
        elif param=='TCRC (USD/t)':
#             log(tcrc) = alpha + beta*log(commodity price) + gamma*log(head grade) 
#                 + delta*risk + epsilon*sxew + theta*dore (refining type)
            log_minesite_cost = h.loc['primary_tcrc_alpha','Value'] +\
                h.loc['primary_tcrc_beta','Value'] * np.log(mines['Commodity price (USD/t)']) +\
                h.loc['primary_tcrc_gamma','Value'] * np.log(mines['Head grade (%)']) +\
                h.loc['primary_tcrc_delta','Value'] * mines['Risk indicator'] +\
                h.loc['primary_tcrc_epsilon','Value'] * (mines['Payable percent (%)']==100) +\
                h.loc['primary_tcrc_theta','Value'] * h.Value.primary_tcrc_dore_flag +\
                h.loc['primary_tcrc_eta','Value'] * (mines['Mine type string']=='tailings')
            mines.loc[:,param] = np.exp(log_minesite_cost)
        elif param=='Sustaining CAPEX ($M)':
#         log(sCAPEX) = alpha + beta*log(commodity price) + gamma*log(head grade) 
#            + delta*log(capacity) + epsilon*placer + theta*stockpile + eta*tailings + rho*underground + zeta*sxew
            prefix = 'Primary ' if self.byproduct else ''
            log_minesite_cost = \
                h.loc['primary_scapex_alpha','Value'] +\
                h.loc['primary_scapex_beta','Value'] * np.log(mines[prefix+'Commodity price (USD/t)']) +\
                h.loc['primary_scapex_gamma','Value'] * np.log(mines[prefix+'Head grade (%)']) +\
                h.loc['primary_scapex_delta','Value'] * np.log(mines['Capacity (kt)']) +\
                h.loc['primary_minesite_cost_epsilon','Value'] * (mines['Mine type string']=='placer') +\
                h.loc['primary_minesite_cost_theta','Value'] * (mines['Mine type string']=='stockpile') +\
                h.loc['primary_minesite_cost_eta','Value'] * (mines['Mine type string']=='tailings') +\
                h.loc['primary_minesite_cost_rho','Value'] * (mines['Mine type string']=='underground') +\
                h.loc['primary_minesite_cost_zeta','Value'] * (mines[prefix+'Payable percent (%)']==100)
            mines.loc[:,param] = np.exp(log_minesite_cost)
            
        self.mines = mines.copy()
     
    def generate_total_cash_margin(self):
        h = self.hyperparam
        
        # Risk indicator is the sum of political, operational, terrorism, and security risks, which range from insignificant (1) to extreme (5)
        risk_upper_cutoff = 20
        risk_lower_cutoff = 4
        self.mines.loc[self.mines['Risk indicator']>risk_upper_cutoff,'Risk indicator'] = risk_upper_cutoff
        self.mines.loc[self.mines['Risk indicator']<risk_lower_cutoff,'Risk indicator'] = risk_lower_cutoff
        
        self.generate_costs_from_regression('TCRC (USD/t)')
        
        if h.Value.primary_tcm_flag:
            self.generate_costs_from_regression('Total cash margin (USD/t)')
            self.mines.loc[:,'Minesite cost (USD/t)'] = self.mines['Commodity price (USD/t)'] - self.mines[['TCRC (USD/t)','Total cash margin (USD/t)']].sum(axis=1)
            self.mines.loc[:,'Total cash cost (USD/t)'] = self.mines[['TCRC (USD/t)','Minesite cost (USD/t)']].sum(axis=1)
            if self.verbose:
                print('tcm')
        else:
            self.generate_costs_from_regression('Minesite cost (USD/t)')
            self.mines.loc[:,'Total cash cost (USD/t)'] = self.mines[['TCRC (USD/t)','Minesite cost (USD/t)']].sum(axis=1)
            self.mines.loc[:,'Total cash margin (USD/t)'] = self.mines['Commodity price (USD/t)'] - self.mines['Total cash cost (USD/t)']
            if self.verbose:
                print('tmc')
            
    def assign_mine_types(self):
        mines = self.mines.copy()
        h = self.hyperparam.copy()
        params = [i for i in h.index if 'minetype_prod_frac' in i]
        self.mine_type_mapping = {0:'openpit',1:'placer',2:'stockpile',3:'tailings',4:'underground'}
        self.mine_type_mapping_rev = {'openpit':0,'placer':1,'stockpile':2,'tailings':3,'underground':4}
        mines.loc[:,'Mine type'] = np.nan
        mines.loc[:,'Mine type string'] = np.nan
        
        for i in params:
            map_param = i.split('_')[-1]
            ind = mines.loc[(mines['Mine type'].isna()),'Production fraction'].cumsum()
            ind = ind.loc[ind<h.loc[i,'Value']*h.loc['primary_production_fraction','Value']].index
            mines.loc[ind,'Mine type'] = self.mine_type_mapping_rev[map_param]
            mines.loc[ind,'Mine type string'] = map_param
        mines.loc[mines['Mine type'].isna(),'Mine type'] = self.mine_type_mapping_rev[h.loc[params,'Value'].astype(float).idxmax().split('_')[-1]]
        mines.loc[mines['Mine type string'].isna(),'Mine type string'] = h.loc[params,'Value'].astype(float).idxmax().split('_')[-1]
        self.mines = mines.copy()
        
    def generate_oges(self):
        s, loc, scale = self.hyperparam.loc['primary_oge_s':'primary_oge_scale','Value']
        self.mines.loc[:,'OGE'] = 1-stats.lognorm.rvs(s, loc, scale, size=self.mines.shape[0], random_state=self.rs)
        i = 0
        while (self.mines['OGE']>0).any():
            self.mines.loc[self.mines['OGE']>0,'OGE'] = 1-stats.lognorm.rvs(s, loc, scale, size=(self.mines['OGE']>0).sum(), random_state=self.rs+i)
            i += 1
    
    def generate_annual_costs(self):
        h = self.hyperparam        
        self.generate_costs_from_regression('Sustaining CAPEX ($M)')
        mines = self.mines.copy()
        
        mines.loc[:,'Overhead ($M)'] = 0.1
        if self.verbose:
            print('Overhead assigned to be $0.1M')
        mines.loc[:,'Paid metal profit ($M)'] = mines[['Paid metal production (kt)','Total cash margin (USD/t)']].product(axis=1)/1e3
        mines.loc[:,'Cash flow ($M)'] = mines['Paid metal profit ($M)']-mines[['Sustaining CAPEX ($M)','Overhead ($M)']].sum(axis=1)
        
        
        self.mines = mines.copy()
    
    def generate_byproduct_mines(self):
        if self.byproduct:
            h = self.hyperparam
            mines = pd.DataFrame()
            pri = GeneralizedOperatingMines(byproduct=True)
            pri.hyperparam = self.hyperparam.copy()
            pri.update_hyperparams_from_byproducts('byproduct')
            pri.update_hyperparams_from_byproducts('byproduct_pri')
            pri.byproduct = False
            
            if pri.hyperparam.Value.byproduct_pri_production_fraction>0:
                pri.run()
                pri.mines.loc[:,'Byproduct ID'] = 0
                self.pri = pri
                byproduct_mine_models = [pri]
                byproduct_mines = [pri.mines]
            else:
                byproduct_mine_models = []
                byproduct_mines = []
                
            for param in np.unique([i.split('_')[1] for i in h.index if 'host' in i]):
                if self.hyperparam.loc['byproduct_'+param+'_production_fraction','Value'] != 0:
                    byproduct_model = self.generate_byproduct_params(param)
                    byproduct_mine_models += [byproduct_model]
                    byproduct_mines += [byproduct_model.mines]
            self.byproduct_mine_models = byproduct_mine_models
            self.mines = pd.concat(byproduct_mines).reset_index(drop=True)
                
    def update_hyperparams_from_byproducts(self,param):
        h = self.hyperparam
        if param == 'byproduct':
            replace_h = [i for i in h.index if 'byproduct' in i and 'host' not in i and i!='byproduct']
        else:
            replace_h = [i for i in h.index if param in i]
        replace_h_split = [i.split(param)[1] for i in replace_h]
        to_replace_h = [i for i in h.index if 'primary' in i and i.split('primary')[1] in replace_h_split]            
        if param == 'byproduct':
            matched = dict([(i,j) for i in replace_h for j in to_replace_h if '_'.join(i.split('_')[1:])=='_'.join(j.split('_')[1:])])            
        else:
            matched = dict([(i,j) for i in replace_h for j in to_replace_h if '_'.join(i.split('_')[2:])=='_'.join(j.split('_')[1:])])
        self.hyperparam.loc[to_replace_h,'Value'] = self.hyperparam.drop(to_replace_h).rename(matched).loc[to_replace_h,'Value']
        if self.verbose:
            display(matched)
        
    def generate_byproduct_params(self,param):
        ''' '''
        self.generate_byproduct_production(param)
        self.generate_byproduct_costs(param)
        self.correct_byproduct_production(param)
        self.generate_byproduct_total_costs(param)
        return getattr(self,param)
        
    def generate_byproduct_production(self,param):
        by_param = 'byproduct_'+param
        host1 = GeneralizedOperatingMines(byproduct=True)
        h = self.hyperparam.copy()
        host1.hyperparam = h.copy()
        host1_params = [i for i in h.index if param in i]
        
        host1.update_hyperparams_from_byproducts(by_param)
        
        production_fraction = h.loc[by_param+'_production_fraction','Value']
        by_dist = getattr(stats,h.loc['byproduct_production_distribution','Value'])
        prod_mean_frac = h.Value.byproduct_production_mean / h.Value.byproduct_production
        prod_var_frac = h.Value.byproduct_production_var / h.Value.byproduct_production

        host1_prod_frac_dist = by_dist.rvs(
            loc=0,
            scale=prod_mean_frac,
            s=prod_var_frac,
            size=int(np.ceil(2/prod_mean_frac)),
            random_state=self.rs)
        mines = pd.DataFrame(
            host1_prod_frac_dist,
            index=np.arange(0,int(np.ceil(2/prod_mean_frac))),
            columns=['Production fraction'])
        mines = mines.loc[mines['Production fraction'].cumsum()<production_fraction,:]
        mines.loc[mines.index[-1]+1,'Production fraction'] = production_fraction-mines['Production fraction'].sum()
        mines.loc[:,'Production (kt)'] = mines['Production fraction']*h.Value.byproduct_production
        
        regions = [i for i in h.index if 'production_frac_region' in i]
        mines.loc[:,'Region'] = np.nan
        for i in regions:
            int_version = int(i.replace('production_frac_region',''))
            ind = mines.loc[(mines.Region.isna()),'Production fraction'].cumsum()
            ind = ind.loc[ind<h.loc[i,'Value']*production_fraction].index
            mines.loc[ind,'Region'] = int_version
        mines.loc[mines.Region.isna(),'Region'] = int(h.loc[regions,'Value'].astype(float).idxmax().replace('production_frac_region',''))
    
        host1.mines = mines.copy()
        host1.assign_mine_types()
        host1.mines.loc[:,'Risk indicator'] = host1.values_from_dist(by_param+'_minerisk').round(0)
        host1.mines.loc[:,'Head grade (%)'] = host1.values_from_dist(by_param+'_ore_grade')

        mines = host1.mines.copy()
        mines.loc[:,'Capacity utilization'] = host1.values_from_dist(by_param+'_cu')
        mines.loc[:,'Recovery rate (%)'] = 100-host1.values_from_dist(by_param+'_rr_default')
        mines.loc[:,'Production capacity (kt)'] = mines[['Capacity utilization','Production (kt)']].product(axis=1)
        mines.loc[:,'Production capacity fraction'] = mines['Production capacity (kt)']/mines['Production capacity (kt)'].sum()
        mines.loc[:,'Payable percent (%)'] = 100-host1.values_from_dist(by_param+'_payable_percent')
        mines.loc[mines['Production fraction'].cumsum()<h.loc[by_param+'_sxew_fraction','Value'],'Payable percent (%)'] = 100       
        mines.loc[:,'Commodity price (USD/t)'] = h.loc[by_param+'_commodity_price','Value']   
        
        host1.mines = mines.copy()
        setattr(self,param,host1)
        
    def generate_byproduct_costs(self,param):
        by_param = 'byproduct_'+param
        host1 = getattr(self,param)
        h = host1.hyperparam
        
        host1.generate_total_cash_margin() 
        mines = host1.mines.copy()
        
        mines.loc[:,'Overhead ($M)'] = 0.1
        if self.verbose:
            print('Overhead assigned to be $0.1M')
        
        mines.loc[:,'Byproduct ID'] = int(param.split('host')[1])
        host1.mines = mines.copy()
        
        pri_main = [i for i in h.index if 'byproduct' in i and 'host' not in i and 'pri' not in i and i!='byproduct']
        setattr(self,param,host1)
        return host1
    
    def correct_byproduct_production(self,param):
        by_param = 'byproduct_'+param
        host1 = getattr(self,param)
        h = host1.hyperparam
        mines = host1.mines.copy()
        
        primary_relocate = ['Commodity price (USD/t)','Recovery rate (%)','Head grade (%)','Payable percent (%)',
                           'Minesite cost (USD/t)','Total cash margin (USD/t)','Total cash cost (USD/t)','TCRC (USD/t)']
        for i in primary_relocate:
            mines.loc[:,'Primary '+i] = mines.loc[:,i]
        mines.loc[:,'Commodity price (USD/t)'] = h.loc['byproduct_commodity_price','Value']
        mines.loc[:,'Recovery rate (%)'] = 95
        mines.loc[:,'Byproduct grade ratio'] = host1.values_from_dist(by_param+'_grade_ratio')
        mines.loc[mines['Byproduct grade ratio']<0,'Byproduct grade ratio'] = mines['Byproduct grade ratio'].sample(n=(mines['Byproduct grade ratio']<0).sum()).values
        mines.loc[:,'Head grade (%)'] = mines['Primary Head grade (%)']/mines['Byproduct grade ratio']
        mines.loc[:,'Payable percent (%)'] = 100
        
        mines.loc[:,'Byproduct minesite cost ratio'] = host1.values_from_dist(by_param+'_minesite_cost_ratio')
        mines.loc[mines['Byproduct minesite cost ratio']<0,'Byproduct minesite cost ratio'] = mines['Byproduct minesite cost ratio'].sample(n=(mines['Byproduct minesite cost ratio']<0).sum()).values
        mines.loc[:,'Minesite cost (USD/t)'] = mines['Primary Minesite cost (USD/t)']/mines['Byproduct minesite cost ratio']
        
        mines.loc[:,'Byproduct TCRC ratio'] = host1.values_from_dist(by_param+'_tcrc_ratio')
        mines.loc[mines['Byproduct TCRC ratio']<0,'Byproduct TCRC ratio'] = mines['Byproduct TCRC ratio'].sample(n=(mines['Byproduct TCRC ratio']<0).sum()).values
        mines.loc[:,'TCRC (USD/t)'] = mines['Primary TCRC (USD/t)']/mines['Byproduct TCRC ratio']
        
        mines.loc[:,'Total cash cost (USD/t)'] = mines[['TCRC (USD/t)','Minesite cost (USD/t)']].sum(axis=1)
        mines.loc[:,'Total cash margin (USD/t)'] = mines['Commodity price (USD/t)'] - mines['Total cash cost (USD/t)']
        
        
        mines.loc[:,'Total cash margin (USD/t)'] = mines['Commodity price (USD/t)']-mines['Minesite cost (USD/t)']
        
        if self.verbose:
            print('Currently assuming 95% byproduct recovery rate and 100% byproduct payable percent. Byproduct recovery rate is multiplied by primary recovery rate in future calculations.')
        
        mines.loc[:,'Ore treated (kt)'] = mines['Production (kt)']/(mines[['Recovery rate (%)','Head grade (%)','Primary Recovery rate (%)']].product(axis=1)/1e6)
        mines.loc[:,'Primary Production (kt)'] = mines[['Ore treated (kt)','Primary Recovery rate (%)','Primary Head grade (%)']].product(axis=1)/1e4
        mines.loc[:,'Capacity (kt)'] = mines['Ore treated (kt)']/mines['Capacity utilization']
        mines.loc[:,'Paid metal production (kt)'] = mines[
            ['Ore treated (kt)','Capacity utilization','Head grade (%)','Recovery rate (%)','Primary Recovery rate (%)','Payable percent (%)']].product(axis=1)/1e8
        mines.loc[:,'Primary Paid metal production (kt)'] = mines[
            ['Ore treated (kt)','Capacity utilization','Primary Head grade (%)','Primary Recovery rate (%)','Primary Payable percent (%)']].product(axis=1)/1e6
        
        mines.loc[:,'Reserves ratio with ore treated'] = host1.values_from_dist('primary_reserves')
        
        mines.loc[:,'Reserves (kt)'] = mines[['Ore treated (kt)','Reserves ratio with ore treated']].product(axis=1)
        mines.loc[:,'Reserves potential metal content (kt)'] = mines[['Reserves (kt)','Head grade (%)']].product(axis=1)*1e-2
        
        # calibrating reserves to input values if needed
        primary_reserves_reported_basis = h.Value.primary_reserves_reported_basis
        primary_reserves_reported = h.Value.primary_reserves_reported
        if primary_reserves_reported_basis == 'ore' and primary_reserves_reported>0:
            ratio = primary_reserves_reported/mines['Reserves (kt)'].sum()
        elif primary_reserves_reported_basis == 'metal' and primary_reserves_reported>0:
            ratio = primary_reserves_reported/mines['Reserves potential metal content (kt)'].sum()
        else:
            ratio = 1
        mines.loc[:,'Reserves (kt)'] *= ratio
        mines.loc[:,'Reserves potential metal content (kt)'] *= ratio     
        
        host1.mines = mines.copy()
        host1.generate_oges()

        pri_main = [i for i in h.index if 'byproduct' in i and 'host' not in i and 'pri' not in i and i!='byproduct']
        setattr(self,param,host1)
        
    def generate_byproduct_total_costs(self,param):
        by_param = 'byproduct_'+param
        host1 = getattr(self,param)
        host1.generate_costs_from_regression('Sustaining CAPEX ($M)')
        h = host1.hyperparam
        mines = host1.mines.copy()
        
        mines.loc[:,'Primary Sustaining CAPEX ($M)'] = mines['Sustaining CAPEX ($M)']
        mines.loc[:,'Byproduct sCAPEX ratio'] = host1.values_from_dist(by_param+'_sus_capex_ratio')
        mines.loc[mines['Byproduct sCAPEX ratio']<0,'Byproduct sCAPEX ratio'] = mines['Byproduct sCAPEX ratio'].sample(n=(mines['Byproduct sCAPEX ratio']<0).sum()).values
        mines.loc[:,'Sustaining CAPEX ($M)'] = mines['Primary Sustaining CAPEX ($M)']/mines['Byproduct sCAPEX ratio']
                
        mines.loc[:,'Paid metal profit ($M)'] = \
            mines[['Paid metal production (kt)','Total cash margin (USD/t)']].product(axis=1)/1e3 +\
            mines[['Primary Paid metal production (kt)','Primary Total cash margin (USD/t)']].product(axis=1)/1e3
            
        mines.loc[:,'Cash flow ($M)'] = mines['Paid metal profit ($M)']-mines[['Sustaining CAPEX ($M)','Primary Sustaining CAPEX ($M)','Overhead ($M)']].sum(axis=1)
        mines.loc[:,'Byproduct cash flow ($M)'] = mines[['Paid metal production (kt)','Total cash margin (USD/t)']].product(axis=1)/1e3 - mines['Sustaining CAPEX ($M)']
        
        mines.loc[:,'Byproduct ID'] = int(param.split('host')[1])
        host1.mines = mines.copy()
        
        pri_main = [i for i in h.index if 'byproduct' in i and 'host' not in i and 'pri' not in i and i!='byproduct']
        setattr(self,param,host1)
        return host1
            
    def values_from_dist(self,param):
        hyperparam = self.hyperparam
        params = [i for i in hyperparam.index if param in i]
        if len(params)==0:
            raise Exception('invalid param value given in values_from_dist call')
        else:
            dist_name = [i for i in params if 'distribution' in i][0]
            if len([i for i in params if 'distribution' in i])>1:
                raise Exception('993'+param+str(params))
            mean_name = [i for i in params if 'mean' in i][0]
            var_name = [i for i in params if 'var' in i][0]
            pri_dist = getattr(stats,hyperparam.loc[dist_name,'Value'])
            pri_mean = hyperparam.loc[mean_name,'Value']
            pri_var = hyperparam.loc[var_name,'Value'] 

            if hyperparam.loc[dist_name,'Value'] == 'norm':
                dist_rvs =  pri_dist.rvs(
                    loc=pri_mean,
                    scale=pri_var,
                    size=self.mines.shape[0],
                    random_state=self.rs)
                dist_rvs[dist_rvs<0] = np.random.choice(dist_rvs[dist_rvs>0],len(dist_rvs[dist_rvs<0]))
            else:
                dist_rvs =  pri_dist.rvs(
                    pri_var,
                    loc=0,
                    scale=pri_mean,
                    size=self.mines.shape[0],
                    random_state=self.rs)
                dist_rvs[dist_rvs<0] = np.random.choice(dist_rvs[dist_rvs>0],len(dist_rvs[dist_rvs<0]))
            if 'ratio' in param:
                dist_rvs[dist_rvs<1] = np.random.choice(dist_rvs[dist_rvs>1],len(dist_rvs[dist_rvs<1]))
            return dist_rvs
        
    def run(self):
        if self.byproduct:
            self.generate_byproduct_mines()
        else:
            self.recalculate_hyperparams()
            self.generate_production_region()
            self.generate_grade_and_masses()
            self.generate_total_cash_margin()
            self.generate_oges()
            self.generate_annual_costs()
        
            self.update_operation_hyperparams()
