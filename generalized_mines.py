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
                      title='Cost supply curve',xlabel='Cumulative bauxite production (kt)',
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
        ph.loc[:,'Sort by'] = ph[stack_cols].sum(axis=1)
    else:
        ph.loc[:,'Sort by'] = ph[stack_cols[0]]
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
    def __init__(self,byproduct=False,verbosity=0,price_change_yoy=0):
        ''''''
        self.byproduct = byproduct
        self.verbosity = verbosity
        self.initialize_hyperparams()
        self.rs = self.hyperparam['Value']['random_state']
        self.hyperparam.loc['primary_recovery_rate_shuffle_param','Value'] = 1
        self.simulation_time = self.hyperparam['Value']['simulation_time']
        self.price_change_yoy = price_change_yoy
        self.i = self.simulation_time[0]
        self.minesite_cost_response_to_grade_price = self.hyperparam['Value']['minesite_cost_response_to_grade_price']
        
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
                sns.histplot(mines,x=i,hue='Byproduct ID',palette=colors,bins=50,log_scale=log_scale,ax=a)
                a.set(title=i)
            elif self.byproduct and i=='Byproduct cash flow ($M)':
                colors = [colors[int(i)] for i in mines.dropna()['Byproduct ID'].unique()]
                sns.histplot(mines.dropna(),x=i,hue='Byproduct ID',palette=colors,bins=50,log_scale=log_scale,ax=a)
                a.set(title=i)
            else:
                mines[i].plot.hist(ax=a, title=i, bins=50)
            if i=='Recovery rate (%)' and self.hyperparam['Value']['primary_rr_negative']:
                a.text(0.05,0.95,'Reset to default,\nnegative values found.\nPrice and grade too low.',
                       va='top',ha='left',transform=a.transAxes)
        
        if plot_recovery_grade_correlation:
            a = ax[-(plot_recovery_grade_correlation+plot_minesite_supply_curve+plot_margin_supply_curve)]
            do_a_regress(mines['Head grade (%)'],mines['Recovery rate (%)'],ax=a)
            a.set(xlabel='Head grade (%)',ylabel='Recovery rate (%)',
                         title='Correlation with partial shuffle param\nValue: {:.2f}'.format(self.hyperparam['Value']['primary_recovery_rate_shuffle_param']))
            
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
        
        if 'parameters for operating mine pool generation, mass':
            hyperparameters.loc['verbosity','Value'] = self.verbosity
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
            hyperparameters.loc['primary_ot_cumu_mean','Value'] = 14.0018
            hyperparameters.loc['primary_ot_cumu_var','Value'] = 0.661
            hyperparameters.loc['primary_ot_cumu_distribution','Value'] = 'lognorm'
            
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

        if 'parameters for operating mine pool generation, cost':
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
            hyperparameters.loc['primary_tcm_flag','Value'] = 'tcm' in hyperparameters['Value']['primary_minesite_cost_regression2use']
            hyperparameters.loc['primary_tcrc_regression2use','Value'] = 'linear_113_reftype' # options: linear_112, linear_112_reftype
            hyperparameters.loc['primary_tcrc_dore_flag','Value'] = False # determines whether the refining process is that for dore or for concentrate
            hyperparameters.loc['primary_sxew_fraction','Value'] = 0 # fraction of primary production coming from sxew mines
            
            hyperparameters.loc['primary_scapex_regression2use','Value'] = 'linear_116_price_cap_sx'
            
            hyperparameters.loc['primary_reclamation_constant',:] = 1.321,'for use in np.exp(1.321+0.671*np.log(mines_cor_adj[Capacity (kt)]))'
            hyperparameters.loc['primary_reclamation_slope',:]    = 0.671,'for use in np.exp(1.321+0.671*np.log(mines_cor_adj[Capacity (kt)]))'
            
            hyperparameters = self.add_minesite_cost_regression_params(hyperparameters)
                        
        if 'parameters for mine life simulation':
            hyperparameters.loc['primary_oge_s','Value'] = 0.3320346
            hyperparameters.loc['primary_oge_loc','Value'] = 0.757959
            hyperparameters.loc['primary_oge_scale','Value'] = 0.399365
            
            hyperparameters.loc['mine_cu_margin_elas','Value'] = 0.01
            hyperparameters.loc['mine_cost_og_elas','Value'] = -0.113
            hyperparameters.loc['mine_cost_price_elas','Value'] = 0.125
            hyperparameters.loc['mine_cu0','Value'] = 0.7688729808870376
            hyperparameters.loc['mine_tcm0','Value'] = 14.575211987093567
            hyperparameters.loc['discount_rate','Value'] = 0.10
            hyperparameters.loc['ramp_down_cu','Value'] = 0.4
            hyperparameters.loc['ramp_up_cu','Value'] = 0.4
            hyperparameters.loc['ramp_up_years','Value'] = 3
            hyperparameters.loc['byproduct_ramp_down_rr','Value'] = 0.4
            hyperparameters.loc['byproduct_ramp_up_rr','Value'] = 0.4
            hyperparameters.loc['byproduct_ramp_up_years','Value'] = 1
            hyperparameters.loc['close_price_method','Value']='mean'
            hyperparameters.loc['close_years_back','Value']=3
            hyperparameters.loc['years_for_roi','Value']=10
            hyperparameters.loc['close_probability_split_max','Value']=0.3
            hyperparameters.loc['close_probability_split_mean','Value']=0.5
            hyperparameters.loc['close_probability_split_min','Value']=0.2
            
            hyperparameters.loc['reinitialize',['Value','Notes']] = np.array([True, 'bool, True runs the setup fn initialize_mine_life instead of pulling from init_mine_life.pkl'],dtype='object')
            hyperparameters.loc['simulation_time',['Value','Notes']] = np.array([np.arange(2019,2041),'years for the simulation'],dtype='object')
            hyperparameters.loc['minesite_cost_response_to_grade_price',['Value','Notes']] = np.array([False,'bool, True,minesite costs respond to ore grade decline as per slide 10 here: Group Research Folder_Olivetti/Displacement/04 Presentations/John/Weekly Updates/20210825 Generalization.pptx'],dtype='object')
            hyperparameters.loc['use_reserves_for_closure',['Value','Notes']] = np.array([False, 'bool, True forces mines to close when they run out of reserves, False allows otherwise. Should always use False'],dtype='object')
            hyperparameters.loc['forever_sim',['Value','Notes']] = np.array([False,'bool, if True allows the simulation to run until all mines have closed (or bauxite price series runs out of values), False only goes until set point'],dtype='object')
            hyperparameters.loc['simulate_closure',['Value','Notes']] = np.array([True,'bool, whether to simulate 2019 operating mines and their closure, default Truebut can be set to False to test mine opening.'],dtype='object')
            hyperparameters.loc['simulate_opening',['Value','Notes']] = np.array([False,'bool, whether to simulate new mine opening, default True but set False during mine opening evaluation so we dont end up in an infinite loop'],dtype='object')
            hyperparameters.loc['reinitialize_incentive_mines',['Value','Notes']] = np.array([False,'bool, default False and True is not set up yet. Whether to create a new  incentive pool of mines or to use the pre-generated one, passing True requires supplying incentive_mine_hyperparameters, which can be accessed by calling self.output_incentive_mine_hyperparameters()'],dtype='object')
            hyperparameters.loc['continuous_incentive',['Value','Notes']] = np.array([False,'bool, if True, maintains the same set of incentive pool mines the entire time, dropping them from the incentive pool and adding them to the operating pool as they open. Hopefully this will eventually also include adding new mines to the incentive as reserves expand. If False, does not drop & samples from incentive pool each time; recommend changing incentive_mine_hyperparameters and setting reinitialize_incentive_mines=True if that is the case. Current default is False'],dtype='object')
            hyperparameters.loc['years_for_roi',['Value','Notes']] = np.array([10,'int, default 10, number of years simulated in simulate_incentive_mines() to determine NPV of incentive mines'],dtype='object')
            hyperparameters.loc['follow_copper_opening_method',['Value','Notes']] = np.array([True, 'bool, if True, generates an incentive pool for each year of the simulation, creates alterable subsample_series to track how many from the pool are sampled in each year'],dtype='object')
            hyperparameters.loc['calibrate_copper_opening_method',['Value','Notes']] = np.array([True, 'bool, should be False once the subsample_series is set up (sets up the subsample series for mine opening evaluation.'],dtype='object')
            
            hyperparameters.loc['primary_commodity_price_option',['Value','Notes']] = np.array(['constant','str, how commodity prices are meant to evolve. Options: constant, yoy, step, input. Input requires setting the variable primary_price_series after model initialization'],dtype='object')
            hyperparameters.loc['byproduct_commodity_price_option',['Value','Notes']] = np.array(['constant','str, how commodity prices are meant to evolve. Options: constant, yoy, step, input. Input requires setting the variable byproduct_price_series after model initialization'],dtype='object')
            hyperparameters.loc['primary_commodity_price_change',['Value','Notes']] = np.array([10,'percentage value, percent change in commodity price year-over-year (yoy) or in its one-year step.'],dtype='object')
            hyperparameters.loc['byproduct_commodity_price_change',['Value','Notes']] = np.array([10,'percentage value, percent change in commodity price year-over-year (yoy) or in its one-year step.'],dtype='object')
#             hyperparameters.loc['',['Value','Notes']] = np.array([],dtype='object')
#             hyperparameters.loc['',['Value','Notes']] = np.array([],dtype='object')
#             hyperparameters.loc['',['Value','Notes']] = np.array([],dtype='object')
            
            hyperparameters.loc['random_state','Value']=20220208
            
        if 'parameters for incentive pool':
            hyperparameters.loc['reserve_frac_region1','Value'] = 0.19743337
            hyperparameters.loc['reserve_frac_region2','Value'] = 0.08555446
            hyperparameters.loc['reserve_frac_region3','Value'] = 0.03290556
            hyperparameters.loc['reserve_frac_region4','Value'] = 0.24350115
            hyperparameters.loc['reserve_frac_region5','Value'] = 0.44060546

        if 'parameters for byproducts' and self.byproduct:
            if 'parameters for byproduct production and grade':
                hyperparameters.loc['byproduct_pri_production_fraction','Value']   = 0.1
                hyperparameters.loc['byproduct_host3_production_fraction','Value'] = 0
                hyperparameters.loc['byproduct_host2_production_fraction','Value'] = 0.4
                hyperparameters.loc['byproduct_host1_production_fraction','Value'] = 1 - hyperparameters.loc[['byproduct_pri_production_fraction','byproduct_host2_production_fraction','byproduct_host3_production_fraction'],'Value'].sum()

                hyperparameters.loc['byproduct0_rr0',:] = 80,'byproduct median recovery rate at simulation start'
                hyperparameters.loc['byproduct1_rr0',:] = 80,'byproduct median recovery rate at simulation start'
                hyperparameters.loc['byproduct1_mine_rrmax',:] = 90,'byproduct median recovery rate at simulation start'
                hyperparameters.loc['byproduct1_mine_tcm0',:] = 25,'byproduct median total cash margin at simulation start'
                hyperparameters.loc['byproduct2_rr0',:] = 85,'byproduct median recovery rate at simulation start'
                hyperparameters.loc['byproduct2_mine_rrmax',:] = 95,'byproduct median recovery rate at simulation start'
                hyperparameters.loc['byproduct2_mine_tcm0',:] = 25,'byproduct median total cash margin at simulation start'
                hyperparameters.loc['byproduct3_rr0',:] = 80,'byproduct median recovery rate at simulation start'
                hyperparameters.loc['byproduct3_mine_rrmax',:] = 95,'byproduct median recovery rate at simulation start'
                hyperparameters.loc['byproduct3_mine_tcm0',:] = 25,'byproduct median total cash margin at simulation start'
                hyperparameters.loc['byproduct_rr_margin_elas',:] = 0.01,'byproduct recovery rate elasticity to total cash margin'
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
                hyperparameters.loc['byproduct_commodity_price','Value'] = 1000 # USD/t
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
        self.mine_cu_margin_elas, self.mine_cost_og_elas, self.mine_cost_price_elas, self.mine_cu0, self.mine_tcm0, self.discount_rate, self.ramp_down_cu, self.ramp_up_cu, self.ramp_up_years = \
            self.hyperparam['Value'][['mine_cu_margin_elas','mine_cost_og_elas','mine_cost_price_elas',
                                 'mine_cu0','mine_tcm0','discount_rate','ramp_down_cu','ramp_up_cu','ramp_up_years']]
        
    def update_operation_hyperparams(self,innie=0):
        hyperparameters = self.hyperparam.copy()
        if type(innie)==int:
            mines = self.mines.copy()
        else:
            mines = innie.copy()
        hyperparameters.loc['mine_cu0','Value'] = mines['Capacity utilization'].median()
        hyperparameters.loc['mine_tcm0','Value'] = mines['Total cash margin (USD/t)'].median()
        if self.byproduct:
            byp = 'Byproduct ' if 'Byproduct Total cash margin (USD/t)' in mines.columns else ''
            for i in np.arange(1,4):
                hyperparameters.loc['byproduct'+str(i)+'_rr0','Value'] = mines.loc[(mines['Byproduct ID']==i)&(mines[byp+'Total cash margin (USD/t)']>0),'Recovery rate (%)'].median()
                hyperparameters.loc['byproduct'+str(i)+'_mine_tcm0','Value'] = mines.loc[(mines['Byproduct ID']==i)&(mines[byp+'Total cash margin (USD/t)']>0),byp+'Total cash margin (USD/t)'].median()
        self.hyperparam = hyperparameters.copy()
            
    def add_minesite_cost_regression_params(self, hyperparameters_):
        hyperparameters = hyperparameters_.copy()
        reg2use = hyperparameters['Value']['primary_minesite_cost_regression2use']
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
        
        reg2use = hyperparameters['Value']['primary_tcrc_regression2use']
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

        reg2use = hyperparameters['Value']['primary_scapex_regression2use']
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
        hyperparameters.loc['primary_tcm_flag','Value'] = 'tcm' in hyperparameters['Value']['primary_minesite_cost_regression2use']
        hyperparameters.loc['primary_rr_negative','Value'] = False
        self.hyperparam = hyperparameters
        
    def generate_production_region(self):
        '''updates self.mines, first function called.'''
        hyperparam = self.hyperparam
        pri_dist = getattr(stats,hyperparam['Value']['primary_production_distribution'])
        pri_prod_mean_frac = hyperparam['Value']['primary_production_mean'] / hyperparam['Value']['primary_production']
        pri_prod_var_frac = hyperparam['Value']['primary_production_var'] / hyperparam['Value']['primary_production']
        production_fraction = hyperparam['Value']['primary_production_fraction']
        
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
        mines.loc[:,'Production (kt)'] = mines['Production fraction']*hyperparam['Value']['primary_production']
        
        regions = [i for i in hyperparam.index if 'production_frac_region' in i]
        mines.loc[:,'Region'] = np.nan
        for i in regions:
            int_version = int(i.replace('production_frac_region',''))
            ind = mines.loc[(mines.Region.isna()),'Production fraction'].cumsum()
            ind = ind.loc[ind<hyperparam['Value'][i]*production_fraction].index
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
        self.mines.loc[:,'Commodity price (USD/t)'] = self.hyperparam['Value']['primary_commodity_price']

        mines = self.mines.copy()
        
        mines.loc[:,'Capacity utilization'] = self.values_from_dist('primary_cu')
        mines.loc[:,'Production capacity (kt)'] = mines[['Capacity utilization','Production (kt)']].product(axis=1)
        mines.loc[:,'Production capacity fraction'] = mines['Production capacity (kt)']/mines['Production capacity (kt)'].sum()
        mines.loc[:,'Payable percent (%)'] = 100-self.values_from_dist('primary_payable_percent')
        mines.loc[mines['Production fraction'].cumsum()<self.hyperparam['Value']['primary_sxew_fraction'],'Payable percent (%)'] = 100
        self.mines = mines.copy()
        self.generate_costs_from_regression('Recovery rate (%)')
        mines = self.mines.copy()
        rec_rates = 100-mines['Recovery rate (%)']
        if rec_rates.max()<30 or (rec_rates<0).any():
            rec_rates = 100 - self.values_from_dist('primary_rr_default')
            self.hyperparam.loc['primary_rr_negative','Value'] = True
        mines.loc[mines.sort_values('Head grade (%)').index,'Recovery rate (%)'] = \
            partial_shuffle(np.sort(rec_rates),self.hyperparam['Value']['primary_recovery_rate_shuffle_param'])
        
        mines.loc[:,'Ore treated (kt)'] = mines['Production (kt)']/(mines['Recovery rate (%)']*mines['Head grade (%)']/1e4)
        mines.loc[:,'Capacity (kt)'] = mines['Ore treated (kt)']/mines['Capacity utilization']
        mines.loc[:,'Paid metal production (kt)'] = mines[
            ['Ore treated (kt)','Capacity utilization','Head grade (%)','Recovery rate (%)','Payable percent (%)']].product(axis=1)/1e6
        mines.loc[:,'Reserves ratio with ore treated'] = self.values_from_dist('primary_reserves')
        
        mines.loc[:,'Reserves (kt)'] = mines[['Ore treated (kt)','Reserves ratio with ore treated']].product(axis=1)
        mines.loc[:,'Reserves potential metal content (kt)'] = mines[['Reserves (kt)','Head grade (%)']].product(axis=1)*1e-2
        
        # calibrating reserves to input values if needed
        primary_reserves_reported_basis = self.hyperparam['Value']['primary_reserves_reported_basis']
        primary_reserves_reported = self.hyperparam['Value']['primary_reserves_reported']
        if primary_reserves_reported_basis == 'ore' and primary_reserves_reported>0:
            ratio = primary_reserves_reported/mines['Reserves (kt)'].sum()
        elif primary_reserves_reported_basis == 'metal' and primary_reserves_reported>0:
            ratio = primary_reserves_reported/mines['Reserves potential metal content (kt)'].sum()
        else:
            ratio = 1
        mines.loc[:,'Reserves (kt)'] *= ratio
        mines.loc[:,'Reserves potential metal content (kt)'] *= ratio
        
        # setting up cumulative ore treated for use with calculating initial grades
        mines.loc[:,'Cumulative ore treated ratio with ore treated'] = self.values_from_dist('primary_ot_cumu')
        mines.loc[:,'Cumulative ore treated (kt)'] = mines['Cumulative ore treated ratio with ore treated']*mines['Ore treated (kt)']
        mines.loc[:,'Opening'] = self.simulation_time[0] - mines['Cumulative ore treated ratio with ore treated'].round(0)
        mines.loc[:,'Initial ore treated (kt)'] = mines['Ore treated (kt)']/self.hyperparam['Value']['ramp_up_years']
        self.mines = mines.copy()

    def generate_costs_from_regression(self,param):
        '''Called inside generate_total_cash_margin'''
        mines = self.mines.copy()
        h = self.hyperparam.copy()
        
        if h['Value']['primary_minesite_cost_mean'] > 0 and param=='Minesite cost (USD/t)':
            mines.loc[:,param] = self.values_from_dist('primary_minesite_cost')
        elif param in ['Minesite cost (USD/t)','Total cash margin (USD/t)','Recovery rate (%)']:
#             log(minesite cost) = alpha + beta*log(head grade) + gamma*(head grade) + delta*(numerical risk) + epsilon*placer (mine type)
#                + theta*stockpile + eta*tailings + rho*underground + zeta*sxew
            log_minesite_cost = h['Value']['primary_minesite_cost_alpha'] +\
                h['Value']['primary_minesite_cost_beta'] * np.log(mines['Commodity price (USD/t)']) +\
                h['Value']['primary_minesite_cost_gamma'] * np.log(mines['Head grade (%)']) +\
                h['Value']['primary_minesite_cost_delta'] * mines['Risk indicator'] +\
                h['Value']['primary_minesite_cost_epsilon'] * (mines['Mine type string']=='placer') +\
                h['Value']['primary_minesite_cost_theta'] * (mines['Mine type string']=='stockpile') +\
                h['Value']['primary_minesite_cost_eta'] * (mines['Mine type string']=='tailings') +\
                h['Value']['primary_minesite_cost_rho'] * (mines['Mine type string']=='underground') +\
                h['Value']['primary_minesite_cost_zeta'] * (mines['Payable percent (%)']==100)
            mines.loc[:,param] = np.exp(log_minesite_cost)
        elif param=='TCRC (USD/t)':
#             log(tcrc) = alpha + beta*log(commodity price) + gamma*log(head grade) 
#                 + delta*risk + epsilon*sxew + theta*dore (refining type)
            log_minesite_cost = h['Value']['primary_tcrc_alpha'] +\
                h['Value']['primary_tcrc_beta'] * np.log(mines['Commodity price (USD/t)']) +\
                h['Value']['primary_tcrc_gamma'] * np.log(mines['Head grade (%)']) +\
                h['Value']['primary_tcrc_delta'] * mines['Risk indicator'] +\
                h['Value']['primary_tcrc_epsilon'] * (mines['Payable percent (%)']==100) +\
                h['Value']['primary_tcrc_theta'] * h['Value']['primary_tcrc_dore_flag'] +\
                h['Value']['primary_tcrc_eta'] * (mines['Mine type string']=='tailings')
            mines.loc[:,param] = np.exp(log_minesite_cost)
        elif param=='Sustaining CAPEX ($M)':
#         log(sCAPEX) = alpha + beta*log(commodity price) + gamma*log(head grade) 
#            + delta*log(capacity) + epsilon*placer + theta*stockpile + eta*tailings + rho*underground + zeta*sxew
            prefix = 'Primary ' if self.byproduct else ''
            log_minesite_cost = \
                h['Value']['primary_scapex_alpha'] +\
                h['Value']['primary_scapex_beta'] * np.log(mines[prefix+'Commodity price (USD/t)']) +\
                h['Value']['primary_scapex_gamma'] * np.log(mines[prefix+'Head grade (%)']) +\
                h['Value']['primary_scapex_delta'] * np.log(mines['Capacity (kt)']) +\
                h['Value']['primary_minesite_cost_epsilon'] * (mines['Mine type string']=='placer') +\
                h['Value']['primary_minesite_cost_theta'] * (mines['Mine type string']=='stockpile') +\
                h['Value']['primary_minesite_cost_eta'] * (mines['Mine type string']=='tailings') +\
                h['Value']['primary_minesite_cost_rho'] * (mines['Mine type string']=='underground') +\
                h['Value']['primary_minesite_cost_zeta'] * (mines[prefix+'Payable percent (%)']==100)
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
        
        if h['Value']['primary_tcm_flag']:
            self.generate_costs_from_regression('Total cash margin (USD/t)')
            self.mines.loc[:,'Minesite cost (USD/t)'] = self.mines['Commodity price (USD/t)'] - self.mines[['TCRC (USD/t)','Total cash margin (USD/t)']].sum(axis=1)
            self.mines.loc[:,'Total cash cost (USD/t)'] = self.mines[['TCRC (USD/t)','Minesite cost (USD/t)']].sum(axis=1)
            if self.verbosity > 1:
                print('tcm')
        else:
            self.generate_costs_from_regression('Minesite cost (USD/t)')
            self.mines.loc[:,'Total cash cost (USD/t)'] = self.mines[['TCRC (USD/t)','Minesite cost (USD/t)']].sum(axis=1)
            self.mines.loc[:,'Total cash margin (USD/t)'] = self.mines['Commodity price (USD/t)'] - self.mines['Total cash cost (USD/t)']
            if self.verbosity > 1:
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
            ind = ind.loc[ind<h['Value'][i]*h['Value']['primary_production_fraction']].index
            mines.loc[ind,'Mine type'] = self.mine_type_mapping_rev[map_param]
            mines.loc[ind,'Mine type string'] = map_param
        mines.loc[mines['Mine type'].isna(),'Mine type'] = self.mine_type_mapping_rev[h['Value'][params].astype(float).idxmax().split('_')[-1]]
        mines.loc[mines['Mine type string'].isna(),'Mine type string'] = h['Value'][params].astype(float).idxmax().split('_')[-1]
        self.mines = mines.copy()
        
    def generate_oges(self):
        s, loc, scale = self.hyperparam.loc['primary_oge_s':'primary_oge_scale','Value']
        self.mines.loc[:,'OGE'] = 1-stats.lognorm.rvs(s, loc, scale, size=self.mines.shape[0], random_state=self.rs)
        i = 0
        while (self.mines['OGE']>0).any():
            self.mines.loc[self.mines['OGE']>0,'OGE'] = 1-stats.lognorm.rvs(s, loc, scale, size=(self.mines['OGE']>0).sum(), random_state=self.rs+i)
            i += 1
        if self.byproduct:
            self.mines.loc[:,'Primary OGE'] = self.mines['OGE']
    
    def generate_annual_costs(self):
        h = self.hyperparam        
        self.generate_costs_from_regression('Sustaining CAPEX ($M)')
        mines = self.mines.copy()
        
        mines.loc[:,'Overhead ($M)'] = 0.1
        if self.verbosity > 1:
            print('Overhead assigned to be $0.1M')
        mines.loc[:,'Paid metal profit ($M)'] = mines[['Paid metal production (kt)','Total cash margin (USD/t)']].product(axis=1)/1e3
        mines.loc[:,'Cash flow ($M)'] = mines['Paid metal profit ($M)']-mines[['Sustaining CAPEX ($M)','Overhead ($M)']].sum(axis=1)
        mines.loc[:,'Total reclamation cost ($M)'] = np.exp(h['Value']['primary_reclamation_constant'] + 
                                                            h['Value']['primary_reclamation_slope']*np.log(mines['Capacity (kt)']/1e3))
        
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
            
            if pri.hyperparam['Value']['byproduct_pri_production_fraction']>0:
                pri.initialize_mines()
                pri.mines.loc[:,'Byproduct ID'] = 0
                self.pri = pri
                byproduct_mine_models = [pri]
                byproduct_mines = [pri.mines]
            else:
                byproduct_mine_models = []
                byproduct_mines = []
                
            for param in np.unique([i.split('_')[1] for i in h.index if 'host' in i]):
                if self.hyperparam['Value']['byproduct_'+param+'_production_fraction'] != 0:
                    byproduct_model = self.generate_byproduct_params(param)
                    byproduct_mine_models += [byproduct_model]
                    byproduct_mines += [byproduct_model.mines]
            self.byproduct_mine_models = byproduct_mine_models
            self.mines = pd.concat(byproduct_mines).reset_index(drop=True)
            self.update_operation_hyperparams()
                
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
        if self.verbosity > 1:
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
        
        production_fraction = h['Value'][by_param+'_production_fraction']
        by_dist = getattr(stats,h['Value']['byproduct_production_distribution'])
        prod_mean_frac = h['Value']['byproduct_production_mean'] / h['Value']['byproduct_production']
        prod_var_frac = h['Value']['byproduct_production_var'] / h['Value']['byproduct_production']

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
        mines.loc[:,'Production (kt)'] = mines['Production fraction']*h['Value']['byproduct_production']
        
        regions = [i for i in h.index if 'production_frac_region' in i]
        mines.loc[:,'Region'] = np.nan
        for i in regions:
            int_version = int(i.replace('production_frac_region',''))
            ind = mines.loc[(mines.Region.isna()),'Production fraction'].cumsum()
            ind = ind.loc[ind<h['Value'][i]*production_fraction].index
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
        mines.loc[mines['Production fraction'].cumsum()<h['Value'][by_param+'_sxew_fraction'],'Payable percent (%)'] = 100       
        mines.loc[:,'Commodity price (USD/t)'] = h['Value'][by_param+'_commodity_price']   
        
        host1.mines = mines.copy()
        setattr(self,param,host1)
        
    def generate_byproduct_costs(self,param):
        by_param = 'byproduct_'+param
        host1 = getattr(self,param)
        h = host1.hyperparam
        
        host1.generate_total_cash_margin() 
        mines = host1.mines.copy()
        
        mines.loc[:,'Overhead ($M)'] = 0.1
        if self.verbosity > 1:
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
        mines.loc[:,'Commodity price (USD/t)'] = h['Value']['byproduct_commodity_price']
        for j in np.arange(0,4):
            mines.loc[mines['Byproduct ID']==j,'Recovery rate (%)'] = h['Value']['byproduct'+str(j)+'_rr0']
        mines.loc[:,'Byproduct grade ratio'] = host1.values_from_dist(by_param+'_grade_ratio')
        mines.loc[mines['Byproduct grade ratio']<0,'Byproduct grade ratio'] = mines['Byproduct grade ratio'].sample(n=(mines['Byproduct grade ratio']<0).sum(),random_state=self.rs).values
        mines.loc[:,'Head grade (%)'] = mines['Primary Head grade (%)']/mines['Byproduct grade ratio']
        mines.loc[:,'Payable percent (%)'] = 100
        
        mines.loc[:,'Byproduct minesite cost ratio'] = host1.values_from_dist(by_param+'_minesite_cost_ratio')
        mines.loc[mines['Byproduct minesite cost ratio']<0,'Byproduct minesite cost ratio'] = mines['Byproduct minesite cost ratio'].sample(n=(mines['Byproduct minesite cost ratio']<0).sum(),random_state=self.rs).values
        mines.loc[:,'Minesite cost (USD/t)'] = mines['Primary Minesite cost (USD/t)']/mines['Byproduct minesite cost ratio']
        
        mines.loc[:,'Byproduct TCRC ratio'] = host1.values_from_dist(by_param+'_tcrc_ratio')
        mines.loc[mines['Byproduct TCRC ratio']<0,'Byproduct TCRC ratio'] = mines['Byproduct TCRC ratio'].sample(n=(mines['Byproduct TCRC ratio']<0).sum(),random_state=self.rs).values
        mines.loc[:,'TCRC (USD/t)'] = mines['Primary TCRC (USD/t)']/mines['Byproduct TCRC ratio']
        
#         mines.loc[:,'Total cash cost (USD/t)'] = mines[['TCRC (USD/t)','Minesite cost (USD/t)']].sum(axis=1)
#         mines.loc[:,'Total cash margin (USD/t)'] = mines['Commodity price (USD/t)'] - mines['Total cash cost (USD/t)']
        
        if self.verbosity > 1:
            print('Currently assuming 95% byproduct recovery rate and 100% byproduct payable percent. Byproduct recovery rate is multiplied by primary recovery rate in future calculations.')
        
        mines.loc[mines['Byproduct ID']==0,'Primary Recovery rate (%)'] = 100
        mines.loc[:,'Ore treated (kt)'] = mines['Production (kt)']/(mines[['Recovery rate (%)','Head grade (%)','Primary Recovery rate (%)']].product(axis=1)/1e6)
        mines.loc[:,'Primary Production (kt)'] = mines[['Ore treated (kt)','Primary Recovery rate (%)','Primary Head grade (%)']].product(axis=1)/1e4
        mines.loc[:,'Capacity (kt)'] = mines['Ore treated (kt)']/mines['Capacity utilization']
        mines.loc[:,'Paid metal production (kt)'] = mines[
            ['Ore treated (kt)','Capacity utilization','Head grade (%)','Recovery rate (%)','Primary Recovery rate (%)','Payable percent (%)']].product(axis=1)/1e8
        mines.loc[:,'Primary Paid metal production (kt)'] = mines[
            ['Ore treated (kt)','Capacity utilization','Primary Head grade (%)','Primary Recovery rate (%)','Primary Payable percent (%)']].product(axis=1)/1e6
        mines.loc[:,'Byproduct Total cash margin (USD/t)'] = mines['Commodity price (USD/t)'].values - mines['Minesite cost (USD/t)'].values - mines['TCRC (USD/t)'].values
        mines.loc[:,'Primary Total cash margin (USD/t)'] = mines['Primary Commodity price (USD/t)'].values - mines['Primary Minesite cost (USD/t)'].values - mines['Primary TCRC (USD/t)'].values
        mines.loc[:,'Total cash margin (USD/t)'] = (mines['Byproduct Total cash margin (USD/t)'].values*mines['Paid metal production (kt)'].values + mines['Primary Total cash margin (USD/t)'].values*mines['Primary Paid metal production (kt)'].values)/mines['Primary Paid metal production (kt)']

        mines.loc[:,'Reserves ratio with ore treated'] = host1.values_from_dist('primary_reserves')
        
        mines.loc[:,'Reserves (kt)'] = mines[['Ore treated (kt)','Reserves ratio with ore treated']].product(axis=1)
        mines.loc[:,'Reserves potential metal content (kt)'] = mines[['Reserves (kt)','Head grade (%)']].product(axis=1)*1e-2
        
        mines.loc[:,'Cumulative ore treated ratio with ore treated'] = host1.values_from_dist('primary_ot_cumu')
        mines.loc[:,'Cumulative ore treated (kt)'] = mines['Cumulative ore treated ratio with ore treated']*mines['Ore treated (kt)']
        mines.loc[:,'Initial ore treated (kt)'] = mines['Ore treated (kt)']/h['Value']['ramp_up_years']
        mines.loc[:,'Opening'] = host1.simulation_time[0] - mines['Cumulative ore treated ratio with ore treated'].round(0)
        
        # calibrating reserves to input values if needed
        primary_reserves_reported_basis = h['Value']['primary_reserves_reported_basis']
        primary_reserves_reported = h['Value']['primary_reserves_reported']
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
        mines.loc[mines['Byproduct sCAPEX ratio']<0,'Byproduct sCAPEX ratio'] = mines['Byproduct sCAPEX ratio'].sample(n=(mines['Byproduct sCAPEX ratio']<0).sum(),random_state=self.rs).values
        mines.loc[:,'Sustaining CAPEX ($M)'] = mines['Primary Sustaining CAPEX ($M)']/mines['Byproduct sCAPEX ratio']
                
        mines.loc[:,'Paid metal profit ($M)'] = \
            mines[['Paid metal production (kt)','Total cash margin (USD/t)']].product(axis=1)/1e3 +\
            mines[['Primary Paid metal production (kt)','Primary Total cash margin (USD/t)']].product(axis=1)/1e3
            
        mines.loc[:,'Cash flow ($M)'] = mines['Paid metal profit ($M)']-mines[['Sustaining CAPEX ($M)','Primary Sustaining CAPEX ($M)','Overhead ($M)']].sum(axis=1)
        mines.loc[:,'Byproduct cash flow ($M)'] = mines[['Paid metal production (kt)','Total cash margin (USD/t)']].product(axis=1)/1e3 - mines['Sustaining CAPEX ($M)']
        mines.loc[:,'Total reclamation cost ($M)'] = np.exp(h['Value']['primary_reclamation_constant'] + 
                                                            h['Value']['primary_reclamation_slope']*np.log(mines['Capacity (kt)']/1e3))
        
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
            pri_dist = getattr(stats,hyperparam['Value'][dist_name])
            pri_mean = hyperparam['Value'][mean_name]
            pri_var = hyperparam['Value'][var_name] 

            np.random.seed(self.rs)
            if hyperparam['Value'][dist_name] == 'norm':
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
        
    def initialize_mines(self):
        '''out: self.mine_life_init
        (copy of self.mines).
        function of hyperparam[reinitialize]'''
        if self.hyperparam['Value']['reinitialize']:
            if self.byproduct:
                self.generate_byproduct_mines()
                # out: self.mines
            else:
                self.recalculate_hyperparams()
                self.generate_production_region()
                self.generate_grade_and_masses()
                self.generate_total_cash_margin()
                self.generate_oges()
                self.generate_annual_costs()
                # out: self.mines

            self.update_operation_hyperparams()
            self.mine_life_init = self.mines.copy()
        else:
            if self.byproduct:
                try:
                    self.mine_life_init = pd.read_pickle('data/mine_life_init_byproduct.pkl')
                except:
                    raise Exception('Save an initialized mine file as data/mine_life_init_byproduct.pkl or set hyperparam.loc[\'reinitialize\',\'Value\'] to True')
            else:
                try:
                    self.mine_life_init = pd.read_pickle('data/mine_life_init_primary.pkl')
                except:
                    raise Exception('Save an initialized mine file as data/mine_life_init_primary.pkl or set hyperparam.loc[\'reinitialize\',\'Value\'] to True')

    def op_initialize_prices(self):
        '''
        relevant hyperparams:
        - primary_commodity_price/byproduct
        - primary_commodity_price_option/byproduct
        - primary_commodity_price_change/byproduct
        
        Can bypass price series generation for pri/by
        by assigning series to primary_price_series
        or byproduct_price_series.
        '''
        h = self.hyperparam
        if self.byproduct and not hasattr(self,'primary_price_series') and not hasattr(self,'byproduct_price_series'):
            strings = ['primary_','byproduct_']
        elif self.byproduct and not hasattr(self,'byproduct_price_series'):
            strings = ['byproduct_']
        elif not hasattr(self,'primary_price_series'):
            strings = ['primary_']
        else:
            strings = []
        for string in strings:
            price = h['Value'][string+'commodity_price']
            price_option = h['Value'][string+'commodity_price_option']
            price_series = pd.Series(price,self.simulation_time)
            price_change = h['Value'][string+'commodity_price_change']

            if price_option == 'yoy':
                price_series.loc[:] = [price*(1+price_change/100)**n for n in np.arange(0,len(self.simulation_time))]
            elif price_option == 'step':
                price_series.iloc[1:] = price*(1+price_change/100)

            setattr(self,string+'price_series',price_series)

    def op_initialize_mine_life(self):
        self.initialize_mines()
        h = self.hyperparam
        if h['Value']['reinitialize']:
            mine_life_init = self.mine_life_init.copy()
            mine_life_init.loc[:,'Ramp up flag'] = False
            mine_life_init.loc[:,'Ramp down flag'] = False
            mine_life_init.loc[:,'Closed flag'] = False
            mine_life_init.loc[:,'Operate with negative cash flow'] = False
            mine_life_init.loc[:,'Total cash margin expect (USD/t)'] = np.nan
            mine_life_init.loc[:,'Cash flow ($M)'] = np.nan
            mine_life_init.loc[:,'Cash flow expect ($M)'] = np.nan
            mine_life_init.loc[:,'NPV ramp next ($M)'] = np.nan
            mine_life_init.loc[:,'NPV ramp following ($M)'] = np.nan
            mine_life_init.loc[:,'Close method'] = np.nan
            mine_life_init.loc[:,'Simulated closure'] = np.nan
            mine_life_init.loc[:,'Initial head grade (%)'] = mine_life_init['Head grade (%)'] / (mine_life_init['Cumulative ore treated (kt)']/mine_life_init['Initial ore treated (kt)'])**mine_life_init['OGE']
            mine_life_init.loc[:,'Discount'] = 1
            
            if self.byproduct:
                mine_life_init.loc[:,'Primary Total cash margin expect (USD/t)'] = np.nan
                mine_life_init.loc[:,'Byproduct Cash flow ($M)'] = np.nan
                mine_life_init.loc[:,'Byproduct Cash flow expect ($M)'] = np.nan
                mine_life_init.loc[:,'Byproduct production flag'] = True
                mine_life_init.loc[:,'Byproduct Ramp up flag'] = False
                mine_life_init.loc[:,'Byproduct Ramp down flag'] = False
                mine_life_init.loc[:,'Primary Initial head grade (%)'] = mine_life_init['Primary Head grade (%)'] / (mine_life_init['Cumulative ore treated (kt)']/mine_life_init['Initial ore treated (kt)'])**mine_life_init['OGE']
                mine_life_init = mine_life_init.fillna(0)
                mine_life_init.loc[:,'Primary Recovery rate (%)'] = mine_life_init['Primary Recovery rate (%)'].replace(0,100)
                mine_life_init.loc[mine_life_init['Primary OGE']==0,'Primary OGE'] = mine_life_init['OGE']
            
            to_drop = ['Byproduct TCRC ratio','Byproduct minesite cost ratio','Byproduct sCAPEX ratio','Cumulative ore treated ratio with ore treated','Reserves ratio with ore treated']
            to_drop = mine_life_init.columns[mine_life_init.columns.isin(to_drop)]
            for j in to_drop:
                mine_life_init.drop(columns=j,inplace=True)
            self.mine_life_init = mine_life_init.copy()
            
        self.ml_yr = self.mine_life_init.copy()
        self.ml = pd.concat([self.ml_yr],keys=[self.i])
        
        if h['Value']['forever_sim']:
            self.simulation_end = self.primary_price_series.index[-1]
        else:
            self.simulation_end = self.simulation_time[-1]
        
    def op_simulate_mine_life(self):
        simulation_time = self.simulation_time
        h = self.hyperparam
        i = self.i
        
        self.hstrings = ['primary_','byproduct_'] if self.byproduct else ['primary_']
        self.istrings = ['Primary ',''] if self.byproduct else ['Primary ']
        primary_price_series = self.primary_price_series
        byproduct_price_series = self.byproduct_price_series if self.byproduct else 0
            
        ml_yr = self.ml.copy().loc[i] if i==simulation_time[0] else self.ml.copy().loc[i-1]
        ml_last = ml_yr.copy()
        
        # No longer include closed mines in the calculations → they won't have any data available after closure
        closed_index = ml_last['Closed flag'][ml_last['Closed flag']].index
        ml_yr = ml_yr.loc[ml_yr.index.isin(ml_last.index)]
        ml_yr = ml_yr.loc[~ml_last.index.isin(closed_index)]
        ml_last = ml_last.loc[~ml_last.index.isin(closed_index)]

        if self.byproduct:
            ml_yr.loc[:,'Primary Commodity price (USD/t)'] *= (primary_price_series.pct_change().fillna(0)+1)[i]
            ml_yr.loc[:,'Commodity price (USD/t)'] *= (byproduct_price_series.pct_change().fillna(0)+1)[i]
        else: 
            ml_yr.loc[:,'Commodity price (USD/t)'] *= (primary_price_series.pct_change().fillna(0)+1)[i]
        
        closing_mines = ml_last['Ramp down flag'][ml_last['Ramp down flag']].index
        opening_mines = ml_yr['Ramp up flag']
        govt_mines = ml_yr['Operate with negative cash flow'][ml_yr['Operate with negative cash flow']].index
        end_ramp_up = ml_yr.loc[(opening_mines)&(ml_yr['Opening']+h['Value']['ramp_up_years']<=i)&(ml_yr['Opening']>simulation_time[0]-1)].index
        if opening_mines.sum()>0:
            ml_yr.loc[(opening_mines)&(ml_yr['Opening'].isna()),'Opening'] = i
            ml_yr.loc[opening_mines,'Capacity utilization'] = h['Value']['ramp_down_cu']
        if len(end_ramp_up)>0:
            ml_yr.loc[end_ramp_up,'Capacity utilization'] = self.calculate_cu(h['Value']['mine_cu0'],ml_last.loc[end_ramp_up,'Total cash margin (USD/t)'],govt=False)
            ml_yr.loc[end_ramp_up,'Ramp up flag'] = False
        ml_yr.loc[~opening_mines,'Development CAPEX ($M)'] = 0
        opening_mines = list(opening_mines[opening_mines].index)
        # Correcting to deal with government mines → 'Operate with negative cash flow' mines. Sets to enter ramp down if reserves have become smaller than prior year's ore treated.
        closing_mines = [i for i in closing_mines if i not in govt_mines or ml_last.loc[i,'Reserves (kt)']<ml_last.loc[i,'Ore treated (kt)']]

        ml_yr.loc[~ml_yr.index.isin(closing_mines+opening_mines),'Capacity utilization'] = self.calculate_cu(ml_last['Capacity utilization'],ml_last['Total cash margin (USD/t)'])
        ml_yr.loc[closing_mines,'Capacity utilization'] = h['Value']['ramp_down_cu']
        ml_yr.loc[:,'Ore treated (kt)'] = ml_yr['Capacity utilization']*ml_yr['Capacity (kt)']
        ml_yr.loc[:,'Reserves (kt)'] -= ml_yr['Ore treated (kt)']
        ml_yr.loc[ml_yr['Initial ore treated (kt)']==0,'Initial ore treated (kt)'] = ml_yr['Ore treated (kt)']
        ml_yr.loc[:,'Cumulative ore treated (kt)'] += ml_yr['Ore treated (kt)']
        ml_yr.loc[:,'Head grade (%)'] = self.calculate_grade(ml_yr['Initial head grade (%)'],ml_yr['Cumulative ore treated (kt)'], ml_yr['Initial ore treated (kt)'], ml_yr['OGE'])

        if self.byproduct:
            if i!=simulation_time[0]:
                for j in np.arange(1,4):
                    ml_yr.loc[ml_yr['Byproduct ID']==j,'Recovery rate (%)'] = (ml_last['Recovery rate (%)']*(ml_last['Byproduct Total cash margin (USD/t)']/h['Value']['byproduct'+str(j)+'_mine_tcm0'])**h['Value']['byproduct_rr_margin_elas']).fillna(-1)
                    ml_yr.loc[(ml_yr['Byproduct ID']==j)&(ml_yr['Recovery rate (%)']>h['Value']['byproduct'+str(j)+'_mine_rrmax']),'Recovery rate (%)'] = h['Value']['byproduct'+str(j)+'_mine_rrmax']
                    problem = ml_yr.loc[ml_yr['Byproduct ID']==j,'Recovery rate (%)']==-1
                    problem = problem[problem].index
                    if len(problem)>0:
                        ml_yr.loc[problem,'Recovery rate (%)'] = ml_last['Recovery rate (%)']
#                         print('Last recovery')
#                         display(ml_last.loc[problem,'Recovery rate (%)'])
#                         print('last tcm')
#                         display(ml_last.loc[problem,'Byproduct Total cash margin (USD/t)'])
#                         print('other')
#                         print(h['Value']['byproduct'+str(j)+'_mine_tcm0'],h['Value']['byproduct_rr_margin_elas'])
            ml_yr.loc[:,'Primary Head grade (%)'] = self.calculate_grade(ml_yr['Primary Initial head grade (%)'],ml_yr['Cumulative ore treated (kt)'], ml_yr['Initial ore treated (kt)'], ml_yr['OGE']).fillna(0)
            ml_yr.loc[ml_yr['Byproduct ID']!=0,'Head grade (%)'] = ml_yr['Primary Head grade (%)']/ml_yr['Byproduct grade ratio']

        ml_yr.loc[closing_mines,'Closed flag'] = True
        ml_yr.loc[closing_mines,'Simulated closure'] = i
        ml_yr.loc[ml_yr['Close method']=='NPV following','Ramp down flag'] = True

        if h['Value']['minesite_cost_response_to_grade_price']:
            ml_yr.loc[:,'Minesite cost (USD/t)'] = ml_last['Minesite cost (USD/t)']*(ml_yr['Head grade (%)']/ml_yr['Initial head grade (%)'])**h['Value']['mine_cost_og_elas']
            if self.byproduct:
                ml_yr.loc[:,'Primary Minesite cost (USD/t)'] = ml_last['Primary Minesite cost (USD/t)']*((ml_yr['Primary Head grade (%)']/ml_yr['Primary Initial head grade (%)'])**h['Value']['mine_cost_og_elas']).fillna(0)
        
        ml_yr.loc[:,'Production (kt)'] = self.calculate_production(ml_yr)            
        ml_yr.loc[:,'Paid metal production (kt)'] = self.calculate_paid_metal_prod(ml_yr)
        if self.byproduct:
            ml_yr.loc[:,'Production (kt)'] *= ml_yr['Primary Recovery rate (%)']/100
            ml_yr.loc[:,'Paid metal production (kt)'] *= ml_yr['Primary Recovery rate (%)']/100
        
            ml_yr.loc[:,'Primary Production (kt)'] = ml_yr['Ore treated (kt)']*ml_yr['Primary Recovery rate (%)']*ml_yr['Primary Head grade (%)']/1e4
            ml_yr.loc[:,'Primary Paid metal production (kt)'] = ml_yr['Ore treated (kt)']*ml_yr['Primary Recovery rate (%)']*ml_yr['Primary Head grade (%)']*ml_yr['Primary Payable percent (%)']/1e6
            ml_yr.loc[:,'Byproduct Total cash margin (USD/t)'] = ml_yr['Commodity price (USD/t)'] - ml_yr['Minesite cost (USD/t)'] - ml_yr['TCRC (USD/t)']
            ml_yr.loc[:,'Byproduct Cash flow ($M)'] = ml_yr['Paid metal production (kt)']*ml_yr['Byproduct Total cash margin (USD/t)'] - ml_yr['Sustaining CAPEX ($M)']
            ml_yr.loc[:,'Primary Total cash margin (USD/t)'] = ml_yr['Primary Commodity price (USD/t)'] - ml_yr['Primary Minesite cost (USD/t)'] - ml_yr['Primary TCRC (USD/t)']
            ml_yr.loc[:,'Total cash margin (USD/t)'] = (ml_yr['Byproduct Total cash margin (USD/t)']*ml_yr['Paid metal production (kt)'] + ml_yr['Primary Total cash margin (USD/t)']*ml_yr['Primary Paid metal production (kt)'])/ml_yr['Primary Paid metal production (kt)']
            by_only = ml_yr['Byproduct ID'][ml_yr['Byproduct ID']==0].index
            ml_yr.loc[by_only,'Total cash margin (USD/t)'] = ml_yr['Byproduct Total cash margin (USD/t)']
            ml_yr.loc[:,'Cash flow ($M)'] = ml_yr['Byproduct Cash flow ($M)'] + ml_yr['Primary Paid metal production (kt)']*ml_yr['Primary Total cash margin (USD/t)'] - ml_yr['Primary Sustaining CAPEX ($M)'] - ml_yr['Overhead ($M)'] - ml_yr['Development CAPEX ($M)']
            ml_yr.loc[by_only,'Cash flow ($M)'] = ml_yr['Byproduct Cash flow ($M)'] - ml_yr['Overhead ($M)'] - ml_yr['Development CAPEX ($M)']
            
            pri_price_df = self.ml['Primary Commodity price (USD/t)'].unstack()
            pri_price_df.loc[i,:] = ml_yr['Primary Commodity price (USD/t)']
            pri_price_expect = self.calculate_price_expect(pri_price_df, i)
            self.pri_price_df = pri_price_df.copy()
            self.no = pri_price_expect
            ml_yr.loc[:,'Primary Price expect (USD/t)'] = pri_price_expect
            
            ml_yr.loc[:,'Primary Revenue ($M)'] = ml_yr['Primary Paid metal production (kt)']*ml_yr['Primary Total cash margin (USD/t)']
            ml_yr.loc[:,'Byproduct Revenue ($M)'] = ml_yr['Paid metal production (kt)']*ml_yr['Byproduct Total cash margin (USD/t)']
            ml_yr.loc[:,'Byproduct revenue fraction'] = ml_yr['Byproduct Revenue ($M)']/(ml_yr['Byproduct Revenue ($M)']+ml_yr['Primary Revenue ($M)'])
            ml_yr.loc[(ml_yr['Byproduct revenue fraction']>1)|(ml_yr['Byproduct revenue fraction']<0),'Byproduct revenue fraction'] = np.nan
        else:
            ml_yr.loc[:,'Total cash margin (USD/t)'] = ml_yr['Commodity price (USD/t)'] - ml_yr['Minesite cost (USD/t)'] - ml_yr['TCRC (USD/t)']
            ml_yr.loc[:,'Cash flow ($M)'] = ml_yr['Paid metal production (kt)']*ml_yr['Total cash margin (USD/t)'] - ml_yr['Overhead ($M)'] - ml_yr['Sustaining CAPEX ($M)'] - ml_yr['Development CAPEX ($M)']
            
        self.govt_mines = ml_yr['Operate with negative cash flow'][ml_yr['Operate with negative cash flow']].index

        price_df = self.ml['Commodity price (USD/t)'].unstack()
        price_df.loc[i,:] = ml_yr['Commodity price (USD/t)']
        price_expect = self.calculate_price_expect(price_df, i)
        ml_yr.loc[:,'Price expect (USD/t)'] = price_expect
        
        # Simplistic byproduct production approach: no byprod prod when byprod cash flow<0 for that year
        if self.byproduct: ml_yr = self.byproduct_closure(ml_yr)
        
        # Check for mines with negative cash flow that should ramp down next year
        ml_yr = self.check_ramp_down(ml_yr, price_df, price_expect)
        ml_yr.loc[ml_yr['Close method']=='NPV next','Ramp down flag'] = True
        
        self.ml_yr = pd.concat([ml_yr],keys=[i])
        if i>self.simulation_time[0]:
            self.ml = pd.concat([self.ml,self.ml_yr])
        else:
            self.ml = self.ml_yr.copy()
        
    def simulate_mine_life_one_year(self):
        h = self.hyperparam
        simulation_time = self.simulation_time
        if self.i == simulation_time[0]:
            self.op_initialize_mine_life()
            self.op_simulate_mine_life()
            self.update_operation_hyperparams(innie=self.ml_yr)
        else: self.op_simulate_mine_life()
    
    def calculate_cu(self, cu_last, tcm_last, govt=True):
        cu = (cu_last*(tcm_last/self.hyperparam['Value']['mine_tcm0'])**self.hyperparam['Value']['mine_cu_margin_elas']).fillna(0.7)
#         ind = np.intersect1d(cu.index,self.govt_mines)
#         ind = np.intersect1d(ind, cu.loc[cu==0.7].index)

#         if govt:
#             cu.loc[ind] = cu_last.loc[ind]
        # government mines (operating under negative cash flow) may have negative tcm and therefore 
        # get their CU set to 0.7. Decided it is better to set them to their previous value instead
        return cu
    
    def calculate_grade(self, initial_grade, cumu_ot, initial_ot, oge=0):
        ''' '''
        grade = initial_grade * (cumu_ot/initial_ot)**oge
        grade.loc[cumu_ot==0] = initial_grade
        grade[grade<0] = 1e-6
        return grade
    
    def calculate_minesite_cost(self, minesite_cost_last, grade, initial_grade, price, initial_price):
        return minesite_cost_last * (grade/initial_grade)**self.hyperparam['Value']['mine_cost_og_elas'] *\
                    (price/initial_price)**self.hyperparam['Value']['mine_cost_price_elas']
    
    def calculate_paid_metal_prod(self, ml_yr):
        return ml_yr['Ore treated (kt)']*ml_yr['Recovery rate (%)']*ml_yr['Head grade (%)']*ml_yr['Payable percent (%)']/1e6
            
    def calculate_production(self, ml_yr):
        return ml_yr['Ore treated (kt)']*ml_yr['Recovery rate (%)']*ml_yr['Head grade (%)']/1e4
        
    def calculate_cash_flow(self, ml_yr):
        '''Intended to skip over some steps and just give cash flows
        for ramp down evaluation. Returns cash flow series.'''
        paid_metal = self.calculate_paid_metal_prod(ml_yr)
        tcm = ml_yr['Commodity price (USD/t)'] - ml_yr['Minesite cost (USD/t)']
        return paid_metal*tcm
    
    def calculate_price_expect(self, ml, i):
        '''i is year index'''
        close_price_method = self.hyperparam['Value']['close_price_method']
        close_years_back = self.hyperparam['Value']['close_years_back']
        close_probability_split_max = self.hyperparam['Value']['close_probability_split_max']
        close_probability_split_mean = self.hyperparam['Value']['close_probability_split_mean']
        close_probability_split_min = self.hyperparam['Value']['close_probability_split_min']
        
        # Process the dataframe of mines to return mine-level price expectation info (series of expected prices for each mine)
        if close_price_method == 'mean':
            if len(ml.index)<=close_years_back:
                price_expect = ml.mean()
            else:
                price_expect = ml.rolling(close_years_back).mean().loc[i]
        elif close_price_method == 'max':
            if len(ml.index)<=close_years_back:
                price_expect = ml.max()
            else:
                price_expect = ml.rolling(close_years_back).max().loc[i]
        elif close_price_method == 'probabilistic':
            if len(ml.index)<=close_years_back:
                price_expect_min = ml.min()
                price_expect_mean = ml.mean()
                price_expect_max = ml.max()
            else:
                price_expect_min = ml.rolling(close_years_back).min().loc[i]
                price_expect_mean = ml.rolling(close_years_back).mean().loc[i]
                price_expect_max = ml.rolling(close_years_back).max().loc[i]
            price_expect = close_probability_split_max*price_expect_max +\
                close_probability_split_mean*price_expect_mean +\
                close_probability_split_min*price_expect_min
        elif close_price_method == 'alonso-ayuso':
            # from Alonso-Ayuso et al (2014). Medium range optimization of copper extraction planning under uncertainty in future copper prices
            # slide 67 in C:\Users\ryter\Dropbox (MIT)\Group Research Folder_Olivetti\Displacement\04 Presentations\John\Weekly Updates\20210825 Generalization.pptx
            price_base = ml.loc[i]
            price_expect = price_base*1.35*0.5 + price_base*0.65*1/6 + price_base*1/3
        return price_expect
    
    def check_ramp_down(self, ml_yr_, price_df, price_expect):
        ml_yr = ml_yr_.copy()
        discount_rate = self.hyperparam['Value']['discount_rate']
        use_reserves_for_closure = self.hyperparam['Value']['use_reserves_for_closure']
        first_yr = self.simulation_time[0]
        i = self.i
        
        overhead = ml_yr['Overhead ($M)']
        sustaining_capex = ml_yr['Sustaining CAPEX ($M)']
        development_capex = ml_yr['Development CAPEX ($M)']
        
        capacity = ml_yr['Capacity (kt)']
        initial_grade = ml_yr['Initial head grade (%)']
        initial_ore_treated = ml_yr['Initial ore treated (kt)']
        initial_price = self.ml.loc[first_yr]['Commodity price (USD/t)']
        oge = ml_yr['OGE']
        
        cu_expect = self.calculate_cu(ml_yr['Capacity utilization'], ml_yr['Total cash margin (USD/t)'])
        ot_expect = cu_expect * capacity
        cumu_ot_expect = ml_yr['Cumulative ore treated (kt)'] + ot_expect
        
        if self.byproduct:
            pri_price_df = self.ml['Primary Commodity price (USD/t)'].unstack()
            pri_price_df.loc[i,:] = ml_yr['Primary Commodity price (USD/t)']
            pri_price_expect = self.calculate_price_expect(pri_price_df, i)
            pri_initial_price = self.ml.loc[first_yr]['Primary Commodity price (USD/t)']
            ml_yr.loc[:,'Primary Price expect (USD/t)'] = pri_price_expect
        else:
            pri_initial_price, pri_price_expect = 0, 0
        
        cash_flow_expect, by_cash_flow_expect, tcm_expect, by_tcm_expect = self.get_cash_flow(ml_yr, cumu_ot_expect, ot_expect, initial_ore_treated, initial_grade, 
                                              price_expect, initial_price, overhead, sustaining_capex, 
                                              development_capex, pri_initial_price, pri_price_expect, 
                                              neg_cash_flow=0)
        
        ml_yr.loc[:,'Cash flow expect ($M)'] = cash_flow_expect
        ml_yr.loc[:,'Total cash margin expect (USD/t)'] = tcm_expect
        ml_yr.loc[:,'Byproduct Total cash margin expect (USD/t)'] = by_tcm_expect
        if self.byproduct: ml_yr.loc[:,'Byproduct Cash flow expect ($M)'] = by_cash_flow_expect
        
        if ml_yr.shape[0]==0 or ml_yr['Reserves (kt)'].notna().sum()==0:
            return ml_yr
        
        exclude_this_yr_reserves = ml_yr.loc[ml_yr['Reserves (kt)']<ot_expect, 'Ramp down flag'].index
        exclude_already_ramping = ml_yr.loc[ml_yr['Ramp down flag']].index
        exclude_ramp_up = ml_yr.loc[ml_yr['Ramp up flag']].index
        neg_cash_flow = ml_yr.loc[ml_yr['Cash flow expect ($M)']<0].index
        if use_reserves_for_closure:
            exclude = list(exclude_this_yr_reserves) + list(exclude_already_ramping) + list(exclude_ramp_up)
        else:
            exclude = list(exclude_already_ramping) + list(exclude_ramp_up)
            
        neg_cash_flow = [i for i in neg_cash_flow if i not in exclude]
        
        ml_yr.loc[:,'CU ramp following'] = np.nan
        ml_yr.loc[:,'Ore treat ramp following'] = np.nan
        ml_yr.loc[:,'Ore treat expect'] = np.nan
        ml_yr.loc[:,'NPV ramp next ($M)'] = np.nan
        ml_yr.loc[:,'NPV ramp following ($M)'] = np.nan
        
        if len(neg_cash_flow)>0:
            if self.verbosity > 1:
                print('len neg cash flow >0', self.i)
            reclamation = ml_yr.loc[neg_cash_flow,'Total reclamation cost ($M)']
            
            # those with reserves likely to be depleted in the year following
            if use_reserves_for_closure:
                reserve_violation = ml_yr.loc[neg_cash_flow,'Reserves (kt)']<ot_expect.loc[neg_cash_flow] + capacity.loc[neg_cash_flow]*self.ramp_down_cu
                reserve_violation = reserve_violation[reserve_violation].index
                ml_yr.loc[reserve_violation,'Ore treat ramp following'] = ml_yr.loc[reserve_violation,'Reserves (kt)']-ot_expect.loc[reserve_violation]
                ml_yr.loc[reserve_violation,'CU ramp following'] = ml_yr.loc[reserve_violation,'Ore treat ramp following']/capacity.loc[reserve_violation]
            
            # those with reserves ok in the year following
            if use_reserves_for_closure:
                reserve_ok = [i for i in neg_cash_flow if i not in reserve_violation]
            else:
                reserve_ok = neg_cash_flow
            ot_ramp_following = self.ramp_down_cu*capacity.loc[reserve_ok]
            
            # Back to all neg_cash_flow mines, evaluate cash flow for ramp following
            cumu_ot_ramp_following = cumu_ot_expect + ot_ramp_following
            cash_flow_ramp_following, by_cash_flow_ramp_following, tcm_rf, by_tcm_rf = self.get_cash_flow(ml_yr, cumu_ot_ramp_following, ot_ramp_following, initial_ore_treated, initial_grade, 
                                              price_expect, initial_price, overhead, sustaining_capex, 
                                              development_capex, pri_initial_price, pri_price_expect, 
                                              neg_cash_flow=neg_cash_flow)
            
            
            # More all neg_cash_flow mines, evaluating cash flow for ramp down in the next year
            ot_ramp_next = self.ramp_down_cu * capacity
            cumu_ot_ramp_next = ml_yr['Ore treated (kt)'] + ot_ramp_next
            cash_flow_ramp_next, by_cash_flow_ramp_next, tcm_rn, by_tcm_rn = self.get_cash_flow(ml_yr, cumu_ot_ramp_next, ot_ramp_next, initial_ore_treated, initial_grade, 
                                              price_expect, initial_price, overhead, sustaining_capex, 
                                              development_capex, pri_initial_price, pri_price_expect, 
                                              neg_cash_flow=neg_cash_flow)
            npv_ramp_following = cash_flow_expect.loc[neg_cash_flow] + cash_flow_ramp_following/(1+discount_rate) - reclamation/(1+discount_rate)**2
            npv_ramp_next = cash_flow_ramp_next - reclamation/(1+discount_rate)
            
            ml_yr.loc[neg_cash_flow,'NPV ramp next ($M)'] = npv_ramp_next
            ml_yr.loc[neg_cash_flow,'NPV ramp following ($M)'] = npv_ramp_following
            
            ramp_down_next = npv_ramp_next>npv_ramp_following
            ramp_down_next = ramp_down_next[ramp_down_next].index
            ramp_down_following = npv_ramp_next<npv_ramp_following
            ramp_down_following = ramp_down_following[ramp_down_following].index
            ml_yr.loc[ramp_down_next, 'Ramp down flag'] = True
            ml_yr.drop(columns=['CU ramp following','Ore treat ramp following','Ore treat expect'],inplace=True)
            ml_yr.loc[ramp_down_next,'Close method'] = 'NPV next'
            ml_yr.loc[ramp_down_following,'Close method'] = 'NPV following'
        return ml_yr
    
    def byproduct_closure(self, ml_yr_):
        ml_yr = ml_yr_.copy()
        byp1 = ml_yr['Byproduct Cash flow ($M)'][ml_yr['Byproduct Cash flow ($M)']<0].index
        byp2 = ml_yr['Byproduct ID'][ml_yr['Byproduct ID']!=0].index
        byp = np.intersect1d(byp1,byp2)
        
        if len(byp)!=0: 
            ml_yr.loc[byp,'Production (kt)'] = 0
            ml_yr.loc[byp,'Paid metal production (kt)'] = 0

            ml_yr.loc[byp,'Byproduct Cash flow ($M)'] = 0
            ml_yr.loc[byp,'Total cash margin (USD/t)'] = ml_yr['Primary Total cash margin (USD/t)']
            ml_yr.loc[byp,'Cash flow ($M)'] = ml_yr['Primary Paid metal production (kt)']*ml_yr['Primary Total cash margin (USD/t)'] - ml_yr['Overhead ($M)'] - ml_yr['Primary Sustaining CAPEX ($M)'] - ml_yr['Development CAPEX ($M)']
        return ml_yr
        
    def get_cash_flow(self, ml_yr_, cumu_ot_expect, ot_expect, initial_ore_treated, initial_grade, price_expect, 
                      initial_price, overhead, sustaining_capex, development_capex,
                      pri_initial_price, pri_price_expect, neg_cash_flow):
        ml_yr = ml_yr_.copy()
        h = self.hyperparam
        grade_expect = self.calculate_grade(ml_yr['Initial head grade (%)'],cumu_ot_expect,initial_ore_treated,ml_yr['OGE'])    
        
        if self.byproduct:
            if self.i!=self.simulation_time[0]:
                for j in np.arange(1,4):
                    ml_yr.loc[ml_yr['Byproduct ID']==j,'Recovery rate (%)'] = (ml_yr['Recovery rate (%)']*(ml_yr['Byproduct Total cash margin (USD/t)']/h['Value']['byproduct'+str(j)+'_mine_tcm0'])**h['Value']['byproduct_rr_margin_elas']).fillna(0)
                    ml_yr.loc[(ml_yr['Byproduct ID']==j)&(ml_yr['Recovery rate (%)']>h['Value']['byproduct'+str(j)+'_mine_rrmax']),'Recovery rate (%)'] = h['Value']['byproduct'+str(j)+'_mine_rrmax']
            pri_initial_grade = ml_yr['Primary Initial head grade (%)']
            pri_grade_expect = self.calculate_grade(ml_yr['Primary Initial head grade (%)'],cumu_ot_expect,ml_yr['Initial ore treated (kt)'],ml_yr['OGE'])
            grade_expect.loc[ml_yr['Byproduct ID'][ml_yr['Byproduct ID']!=0].index] = pri_grade_expect/ml_yr['Byproduct grade ratio']
            
        if self.hyperparam['Value']['minesite_cost_response_to_grade_price']:
            minesite_cost_expect = self.calculate_minesite_cost(ml_yr['Minesite cost (USD/t)'], grade_expect, initial_grade, price_expect, initial_price)
        else:
            minesite_cost_expect = ml_yr['Minesite cost (USD/t)']
        
        paid_metal_expect = ot_expect * grade_expect * ml_yr['Recovery rate (%)'] * ml_yr['Payable percent (%)'] * 1e-6
        tcm_expect = price_expect - minesite_cost_expect - ml_yr['TCRC (USD/t)']   
        cash_flow_expect = paid_metal_expect*tcm_expect - overhead - sustaining_capex - development_capex
        by_cash_flow_expect = 0
        
        if self.byproduct:
            paid_metal_expect *= ml_yr['Primary Recovery rate (%)']/100
            by_cash_flow_expect = paid_metal_expect*tcm_expect - sustaining_capex

            pri_paid_metal_expect = ot_expect * pri_grade_expect * ml_yr['Primary Recovery rate (%)'] * ml_yr['Primary Payable percent (%)'] /1e6
            if self.hyperparam['Value']['minesite_cost_response_to_grade_price']:
                pri_minesite_cost_expect = self.calculate_minesite_cost(ml_yr['Primary Minesite cost (USD/t)'], pri_grade_expect, pri_initial_grade, pri_price_expect, pri_initial_price)
            else: 
                pri_minesite_cost_expect = ml_yr['Primary Minesite cost (USD/t)']
            pri_tcm_expect = pri_price_expect - pri_minesite_cost_expect - ml_yr['Primary TCRC (USD/t)']
            by_tcm_expect = tcm_expect.copy()
            tcm_expect = (by_tcm_expect*paid_metal_expect + pri_tcm_expect*pri_paid_metal_expect)/pri_paid_metal_expect
            by_only = ml_yr['Byproduct ID'][ml_yr['Byproduct ID']==0].index
            tcm_expect.loc[by_only] = by_tcm_expect
            cash_flow_expect = by_cash_flow_expect + pri_paid_metal_expect*pri_tcm_expect - ml_yr['Primary Sustaining CAPEX ($M)'] - overhead - development_capex
            cash_flow_expect.loc[by_only] = by_cash_flow_expect - overhead - development_capex
        if type(neg_cash_flow)!=int:
            cash_flow_expect = cash_flow_expect.loc[neg_cash_flow]
        return cash_flow_expect, by_cash_flow_expect, tcm_expect, by_tcm_expect
