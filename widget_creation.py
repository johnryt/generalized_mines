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
import functools

from generalized_mines import *
from other_functions import *
m = GeneralizedOperatingMines(byproduct=True)
m.run()

init_plot2()

def define_hyperparam_list():
    hyperparam_list = pd.DataFrame(np.nan,['byproduct'],['Value','Min','Max','Type','Name','Tab'])

    if 'Primary parameters':
        hyperparam_list.loc['byproduct'] = True, False, True, 'Dropdown-bool','Byproduct:','Primary only'
        hyperparam_list.loc['primary_production'] = 1, 1, 1000, 'IntSlider', 'Primary production (kt):','Primary only'
        hyperparam_list.loc['primary_production_mean'] = 0.003, 1e-4, 1e3, 'FloatSlider', 'Mean mine production (kt):', 'Primary only'
        hyperparam_list.loc['primary_production_var'] = 1, 0.1, 2, 'FloatSlider', 'Mine production var (kt)', 'Primary only'
        hyperparam_list.loc['primary_production_distribution'] = 'lognorm','','','Dropdown-dist','Mine prod. dist.:','Primary only'
        hyperparam_list.loc['primary_production_fraction'] = 1, 0.01, 1, 'FloatSlider', 'Primary prod. frac.:', 'Primary only'
        hyperparam_list.loc['primary_ore_grade_mean'] = 0.001, 1e-5, 40, 'FloatSlider', 'Grade mean:', 'Primary only'
        hyperparam_list.loc['primary_ore_grade_var'] = 0.3, 0.01, 2, 'FloatSlider', 'Grade var:', 'Primary only'
        hyperparam_list.loc['primary_ore_grade_distribution'] = 'lognorm','','','Dropdown-dist','Ore grade dist.:','Primary only'
        hyperparam_list.loc['primary_cu_mean'] = 0.85, 0.5, 1, 'FloatSlider', 'CU mean:', 'Primary only'
        hyperparam_list.loc['primary_cu_var'] = 0.06, 0.01, 0.1, 'FloatSlider', 'CU var:', 'Primary only'
        hyperparam_list.loc['primary_cu_distribution'] = 'lognorm','','','Dropdown-dist','CU dist.:','Primary only'
        hyperparam_list.loc['primary_payable_percent_mean'] = 0.63, 0.5, 1, 'FloatSlider', 'Payable % mean:', 'Primary only'
        hyperparam_list.loc['primary_payable_percent_var'] = 1.83, 1, 3, 'FloatSlider', 'Payable % var:', 'Primary only'
        hyperparam_list.loc['primary_payable_percent_distribution'] = 'weibull_min','','','Dropdown-dist','Payable % dist.:','Primary only'
        hyperparam_list.loc['primary_rr_default_mean'] = 13.996, 10, 20, 'FloatSlider', '100-Rec. rate mean:', 'Primary only'
        hyperparam_list.loc['primary_rr_default_var'] = 0.675, 0.1, 1, 'FloatSlider', 'Rec. rate var:', 'Primary only'
        hyperparam_list.loc['primary_rr_default_distribution'] = 'lognorm','','','Dropdown-dist','Rec. rate dist.:','Primary only'
        hyperparam_list.loc['primary_commodity_price'] = 6000, 1000, 20000, 'IntSlider','Commodity price (USD/t):','Primary only'
        hyperparam_list.loc['primary_minerisk_mean'] = 9.4, 4, 20, 'FloatSlider','Risk mean:','Primary only'
        hyperparam_list.loc['primary_minerisk_var'] = 1.35, 0.1, 10, 'FloatSlider','Risk var:','Primary only'
        hyperparam_list.loc['primary_minerisk_distribution'] = 'norm','','','Dropdown-dist','Risk dist.:','Primary only'

    if 'Mine life sim':
        hyperparam_list.loc['primary_oge_s'] = 0.3320346, 0.01, 1, 'FloatSlider', 'OGE elas. shape:', 'Mine life sim.'
        hyperparam_list.loc['primary_oge_loc'] = 0.757959, 0.01, 1, 'FloatSlider', 'OGE elas. loc:', 'Mine life sim.'
        hyperparam_list.loc['primary_oge_scale'] = 0.399365, 0.01, 1, 'FloatSlider', 'OGE elas. scale:', 'Mine life sim.'
        hyperparam_list.loc['mine_cu_margin_elas'] = 0.01, 0, 1, 'FloatSlider', 'Mine CU margin elas:', 'Mine life sim.'
        hyperparam_list.loc['mine_cost_OG_elas'] = -0.113, -1, 0, 'FloatSlider', 'Mine cost OG elas:', 'Mine life sim.'
        hyperparam_list.loc['mine_cost_price_elas'] = 0.125, 0, 1, 'FloatSlider', 'Mine cost price elas:', 'Mine life sim.'
        hyperparam_list.loc['discount_rate'] = 0.10, 0.01, 1, 'FloatSlider', 'Discount rate (frac):', 'Mine life sim.'
        hyperparam_list.loc['ramp_down_cu'] = 0.4, 0.1, 1, 'FloatSlider', 'Ramp down CU:', 'Mine life sim.'
        hyperparam_list.loc['ramp_up_cu'] = 0.4, 0.1, 1, 'FloatSlider', 'Ramp up CU:', 'Mine life sim.'
        hyperparam_list.loc['ramp_up_years'] = 3, 1, 5, 'IntSlider', 'Ramp up years:', 'Mine life sim.'
        hyperparam_list.loc['close_price_method']='mean', '','', 'Dropdown-price', 'Close price method:','Mine life sim.'
        hyperparam_list.loc['close_years_back']=3, 1, 5, 'IntSlider', 'Close years back:', 'Mine life sim.'
        hyperparam_list.loc['years_for_roi']=10, 1, 50, 'IntSlider', 'Years for ROI:', 'Mine life sim.'
        hyperparam_list.loc['close_probability_split_max']=0.3, 0, 1, 'FloatSlider', 'Close prob. split max:', 'Mine life sim.'
        hyperparam_list.loc['close_probability_split_mean']=0.5, 0, 1, 'FloatSlider', 'Close prob. split mean:', 'Mine life sim.'
        hyperparam_list.loc['close_probability_split_min']=0.2, 0, 1, 'FloatSlider', 'Close prob. split min (NO ADJUST):', 'Mine life sim.'
        hyperparam_list.loc['random_state']=20220208, 20200101, 20260000, 'IntSlider', 'Random state:', 'Mine life sim.'

    if 'Byproducts production and grade':
        hyperparam_list.loc['byproduct_commodity_price'] = 10000, 1000, 40000, 'IntSlider', 'Byproduct price (USD/t):', 'Byproduct main'

        hyperparam_list.loc['byproduct_host1_commodity_price'] = 2000, 500, 40000, 'IntSlider', 'Host 1 price (USD/t):', 'Byproduct main'
        hyperparam_list.loc['byproduct_host2_commodity_price'] = 3000, 500, 40000, 'IntSlider', 'Host 2 price (USD/t):', 'Byproduct main'
        hyperparam_list.loc['byproduct_host3_commodity_price'] = 1000, 500, 40000, 'IntSlider', 'Host 3 price (USD/t):', 'Byproduct main'

        hyperparam_list.loc['byproduct_pri_production_fraction']   = 0.1, 0, 1, 'FloatSlider', 'Byproduct prod. pri. frac.:','Byproduct main'
        hyperparam_list.loc['byproduct_host1_production_fraction'] = 0.5, 0, 1, 'FloatSlider', 'Byproduct prod. host 1 frac.:','Byproduct main'
        hyperparam_list.loc['byproduct_host2_production_fraction'] = 0.4, 0, 1, 'FloatSlider', 'Byproduct prod. host 2 frac.:','Byproduct main'
        hyperparam_list.loc['byproduct_host3_production_fraction'] = 0, 0, 1, 'FloatSlider', 'Byproduct prod. host 3 frac. (NO ADJUST):','Byproduct main'

        hyperparam_list.loc['byproduct_production'] = 4, 1, 1000, 'IntSlider', 'Byproduct production (kt):','Byproduct main'
        hyperparam_list.loc['byproduct_production_mean'] = 0.03, 1e-4, 1e3, 'FloatSlider', 'Mean mine production (kt):', 'Byproduct main'
        hyperparam_list.loc['byproduct_production_var'] = 0.5, 0.1, 2, 'FloatSlider', 'Mine production var (kt)', 'Byproduct main'
        hyperparam_list.loc['byproduct_production_distribution'] = 'lognorm', '','','Dropdown-dist','Mine prod. dist.:','Byproduct main'

        hyperparam_list.loc['byproduct_pri_ore_grade_mean']   = 0.1, 0.001, 40, 'FloatSlider', 'Pri ore grade mean:', 'Primary mines'
        hyperparam_list.loc['byproduct_host1_ore_grade_mean'] = 0.1, 0.001, 40, 'FloatSlider', 'Host 1 ore grade mean:', 'Host 1'
        hyperparam_list.loc['byproduct_host2_ore_grade_mean'] = 0.1, 0.001, 40, 'FloatSlider', 'Host 2 ore grade mean:', 'Host 2'
        hyperparam_list.loc['byproduct_host3_ore_grade_mean'] = 0.1, 0.001, 40, 'FloatSlider', 'Host 3 ore grade mean:', 'Host 3'
        hyperparam_list.loc['byproduct_pri_ore_grade_var']   = 0.3, 0.001, 4, 'FloatSlider', 'Pri ore grade var:', 'Primary mines'
        hyperparam_list.loc['byproduct_host1_ore_grade_var'] = 0.3, 0.001, 4, 'FloatSlider', 'Host 1 ore grade var:', 'Host 1'                
        hyperparam_list.loc['byproduct_host2_ore_grade_var'] = 0.3, 0.001, 4, 'FloatSlider', 'Host 2 ore grade var:', 'Host 2'                
        hyperparam_list.loc['byproduct_host3_ore_grade_var'] = 0.3, 0.001, 4, 'FloatSlider', 'Host 3 ore grade var:', 'Host 3'
        hyperparam_list.loc['byproduct_pri_ore_grade_distribution']   = 'lognorm', '', '', 'Dropdown-dist', 'Pri ore grade dist.:', 'Primary mines'
        hyperparam_list.loc['byproduct_host1_ore_grade_distribution'] = 'lognorm', '', '', 'Dropdown-dist', 'Host 1 ore grade dist.:', 'Host 1'
        hyperparam_list.loc['byproduct_host2_ore_grade_distribution'] = 'lognorm', '', '', 'Dropdown-dist', 'Host 2 ore grade dist.:', 'Host 2'
        hyperparam_list.loc['byproduct_host3_ore_grade_distribution'] = 'lognorm', '', '', 'Dropdown-dist', 'Host 3 ore grade dist.:', 'Host 3'

        hyperparam_list.loc['byproduct_pri_cu_mean']   = 0.85, 0.001, 1, 'FloatSlider', 'Pri CU mean:', 'Primary mines'
        hyperparam_list.loc['byproduct_host1_cu_mean'] = 0.85, 0.001, 1, 'FloatSlider', 'Host 1 CU mean:', 'Host 1'
        hyperparam_list.loc['byproduct_host2_cu_mean'] = 0.85, 0.001, 1, 'FloatSlider', 'Host 2 CU mean:', 'Host 2'
        hyperparam_list.loc['byproduct_host3_cu_mean'] = 0.85, 0.001, 1, 'FloatSlider', 'Host 3 CU mean:', 'Host 3'
        hyperparam_list.loc['byproduct_pri_cu_var']   = 0.06, 0.001, 0.2, 'FloatSlider', 'Pri CU var:', 'Primary mines'
        hyperparam_list.loc['byproduct_host1_cu_var'] = 0.06, 0.001, 0.2, 'FloatSlider', 'Host 1 CU var:', 'Host 1'
        hyperparam_list.loc['byproduct_host2_cu_var'] = 0.06, 0.001, 0.2, 'FloatSlider', 'Host 2 CU var:', 'Host 2'
        hyperparam_list.loc['byproduct_host3_cu_var'] = 0.06, 0.001, 0.2, 'FloatSlider', 'Host 3 CU var:', 'Host 3'
        hyperparam_list.loc['byproduct_pri_cu_distribution']   = 'lognorm', '', '', 'Dropdown-dist', 'Pri CU dist.:', 'Primary mines'
        hyperparam_list.loc['byproduct_host1_cu_distribution'] = 'lognorm', '', '', 'Dropdown-dist', 'Host 1 CU dist.:', 'Host 1'
        hyperparam_list.loc['byproduct_host2_cu_distribution'] = 'lognorm', '', '', 'Dropdown-dist', 'Host 2 CU dist.:', 'Host 2'
        hyperparam_list.loc['byproduct_host3_cu_distribution'] = 'lognorm', '', '', 'Dropdown-dist', 'Host 3 CU dist.:', 'Host 3'

        hyperparam_list.loc['byproduct_pri_payable_percent_mean']   = 0.63, 0.001, 1, 'FloatSlider', 'Pri payable % mean:', 'Primary mines'
        hyperparam_list.loc['byproduct_host1_payable_percent_mean'] = 0.63, 0.001, 1, 'FloatSlider', 'Host 1 payable % mean:', 'Host 1'
        hyperparam_list.loc['byproduct_host2_payable_percent_mean'] = 0.63, 0.001, 1, 'FloatSlider', 'Host 2 payable % mean:', 'Host 2'
        hyperparam_list.loc['byproduct_host3_payable_percent_mean'] = 0.63, 0.001, 1, 'FloatSlider', 'Host 3 payable % mean:', 'Host 3'
        hyperparam_list.loc['byproduct_pri_payable_percent_var']   = 1.83, 0.001, 4, 'FloatSlider', 'Pri payable % var:', 'Primary mines'
        hyperparam_list.loc['byproduct_host1_payable_percent_var'] = 1.83, 0.001, 4, 'FloatSlider', 'Host 1 payable % var:', 'Host 1'
        hyperparam_list.loc['byproduct_host2_payable_percent_var'] = 1.83, 0.001, 4, 'FloatSlider', 'Host 2 payable % var:', 'Host 2'
        hyperparam_list.loc['byproduct_host3_payable_percent_var'] = 1.83, 0.001, 4, 'FloatSlider', 'Host 3 payable % var:', 'Host 3'
        hyperparam_list.loc['byproduct_pri_payable_percent_distribution']   = 'weibull_min', '', '', 'Dropdown-dist', 'Pri payable % dist.:', 'Primary mines'
        hyperparam_list.loc['byproduct_host1_payable_percent_distribution'] = 'weibull_min', '', '', 'Dropdown-dist', 'Host 1 payable % dist.:', 'Host 1'
        hyperparam_list.loc['byproduct_host2_payable_percent_distribution'] = 'weibull_min', '', '', 'Dropdown-dist', 'Host 2 payable % dist.:', 'Host 2'
        hyperparam_list.loc['byproduct_host3_payable_percent_distribution'] = 'weibull_min', '', '', 'Dropdown-dist', 'Host 3 payable % dist.:', 'Host 3'

        hyperparam_list.loc['byproduct_pri_rr_default_mean']   = 13.996, 1, 20, 'FloatSlider', 'Pri rec. rate mean:', 'Primary mines'
        hyperparam_list.loc['byproduct_host1_rr_default_mean'] = 13.996, 1, 20, 'FloatSlider', 'Host 1 rec. rate mean:', 'Host 1'
        hyperparam_list.loc['byproduct_host2_rr_default_mean'] = 13.996, 1, 20, 'FloatSlider', 'Host 2 rec. rate mean:', 'Host 2'
        hyperparam_list.loc['byproduct_host3_rr_default_mean'] = 13.996, 1, 20, 'FloatSlider', 'Host 3 rec. rate mean:', 'Host 3'
        hyperparam_list.loc['byproduct_pri_rr_default_var']   = 0.675, 0.001, 4, 'FloatSlider', 'Pri rec. rate var:', 'Primary mines'
        hyperparam_list.loc['byproduct_host1_rr_default_var'] = 0.675, 0.001, 4, 'FloatSlider', 'Host 1 rec. rate var:', 'Host 1'
        hyperparam_list.loc['byproduct_host2_rr_default_var'] = 0.675, 0.001, 4, 'FloatSlider', 'Host 2 rec. rate var:', 'Host 2'
        hyperparam_list.loc['byproduct_host3_rr_default_var'] = 0.675, 0.001, 4, 'FloatSlider', 'Host 3 rec. rate var:', 'Host 3'
        hyperparam_list.loc['byproduct_pri_rr_default_distribution']   = 'lognorm', '', '', 'Dropdown-dist', 'Pri rec. rate dist.:', 'Byproduct main'
        hyperparam_list.loc['byproduct_host1_rr_default_distribution'] = 'lognorm', '', '', 'Dropdown-dist', 'Host 1 rec. rate dist.:', 'Host 1'
        hyperparam_list.loc['byproduct_host2_rr_default_distribution'] = 'lognorm', '', '', 'Dropdown-dist', 'Host 2 rec. rate dist.:', 'Host 2'
        hyperparam_list.loc['byproduct_host3_rr_default_distribution'] = 'lognorm', '', '', 'Dropdown-dist', 'Host 3 rec. rate dist.:', 'Host 3'

        hyperparam_list.loc['byproduct_pri_minerisk_mean']   = 9.4, 4, 20, 'FloatSlider', 'Pri risk mean:', 'Primary mines'
        hyperparam_list.loc['byproduct_host1_minerisk_mean'] = 9.4 , 4, 20, 'FloatSlider', 'Host 1 risk mean:', 'Host 1'
        hyperparam_list.loc['byproduct_host2_minerisk_mean'] = 9.4 , 4, 20, 'FloatSlider', 'Host 2 risk mean:', 'Host 2'
        hyperparam_list.loc['byproduct_host3_minerisk_mean'] = 9.4 , 4, 20, 'FloatSlider', 'Host 3 risk mean:', 'Host 3'
        hyperparam_list.loc['byproduct_pri_minerisk_var']   = 1.35, 0.001, 4, 'FloatSlider', 'Pri risk var:', 'Primary mines'
        hyperparam_list.loc['byproduct_host1_minerisk_var'] = 1.35, 0.001, 4, 'FloatSlider', 'Host 1 risk var:', 'Host 1'
        hyperparam_list.loc['byproduct_host2_minerisk_var'] = 1.35, 0.001, 4, 'FloatSlider', 'Host 2 risk var:', 'Host 2'
        hyperparam_list.loc['byproduct_host3_minerisk_var'] = 1.35, 0.001, 4, 'FloatSlider', 'Host 3 risk var:', 'Host 3'
        hyperparam_list.loc['byproduct_pri_minerisk_distribution']   = 'norm', '', '', 'Dropdown-dist', 'Pri risk dist.:', 'Byproduct main'
        hyperparam_list.loc['byproduct_host1_minerisk_distribution'] = 'norm', '', '', 'Dropdown-dist', 'Host 1 risk dist.:', 'Host 1'
        hyperparam_list.loc['byproduct_host2_minerisk_distribution'] = 'norm', '', '', 'Dropdown-dist', 'Host 2 risk dist.:', 'Host 2'
        hyperparam_list.loc['byproduct_host3_minerisk_distribution'] = 'norm', '', '', 'Dropdown-dist', 'Host 3 risk dist.:', 'Host 3'

        hyperparam_list.loc['byproduct_host1_minesite_cost_ratio_mean'] = 20, 1, 100, 'FloatSlider', 'Host 1 : byprod. cost mean:', 'Host 1'
        hyperparam_list.loc['byproduct_host2_minesite_cost_ratio_mean'] = 2, 1, 100, 'FloatSlider', 'Host 2 : byprod. cost mean:', 'Host 2'
        hyperparam_list.loc['byproduct_host3_minesite_cost_ratio_mean'] = 10, 1, 100, 'FloatSlider', 'Host 3 : byprod. cost mean:', 'Host 3'
        hyperparam_list.loc['byproduct_host1_minesite_cost_ratio_var'] = 1, 0.001, 4, 'FloatSlider', 'Host 1 : byprod. cost var:', 'Host 1'
        hyperparam_list.loc['byproduct_host2_minesite_cost_ratio_var'] = 1, 0.001, 4, 'FloatSlider', 'Host 2 : byprod. cost var:', 'Host 2'
        hyperparam_list.loc['byproduct_host3_minesite_cost_ratio_var'] = 1, 0.001, 4, 'FloatSlider', 'Host 3 : byprod. cost var:', 'Host 3'
        hyperparam_list.loc['byproduct_host1_minesite_cost_ratio_distribution'] = 'norm', '', '', 'Dropdown-dist', 'Host 1 : byprod. cost dist.:', 'Host 1'
        hyperparam_list.loc['byproduct_host2_minesite_cost_ratio_distribution'] = 'norm', '', '', 'Dropdown-dist', 'Host 2 : byprod. cost dist.:', 'Host 2'
        hyperparam_list.loc['byproduct_host3_minesite_cost_ratio_distribution'] = 'norm', '', '', 'Dropdown-dist', 'Host 3 : byprod. cost dist.:', 'Host 3'

        hyperparam_list.loc['byproduct_host1_sus_capex_ratio_mean'] = 20, 1, 100, 'FloatSlider', 'Host 1 : byprod. sCAPEX mean:', 'Host 1'
        hyperparam_list.loc['byproduct_host2_sus_capex_ratio_mean'] = 2, 1, 100, 'FloatSlider', 'Host 2 : byprod. sCAPEX mean:', 'Host 2'
        hyperparam_list.loc['byproduct_host3_sus_capex_ratio_mean'] = 10, 1, 100, 'FloatSlider', 'Host 3 : byprod. sCAPEX mean:', 'Host 3'
        hyperparam_list.loc['byproduct_host1_sus_capex_ratio_var'] = 1, 0.001, 4, 'FloatSlider', 'Host 1 : byprod. sCAPEX var:', 'Host 1'
        hyperparam_list.loc['byproduct_host2_sus_capex_ratio_var'] = 1, 0.001, 4, 'FloatSlider', 'Host 2 : byprod. sCAPEX var:', 'Host 2'
        hyperparam_list.loc['byproduct_host3_sus_capex_ratio_var'] = 1, 0.001, 4, 'FloatSlider', 'Host 3 : byprod. sCAPEX var:', 'Host 3'
        hyperparam_list.loc['byproduct_host1_sus_capex_ratio_distribution'] = 'norm', '', '', 'Dropdown-dist', 'Host 1 : byprod. sCAPEX dist.:', 'Host 1'
        hyperparam_list.loc['byproduct_host2_sus_capex_ratio_distribution'] = 'norm', '', '', 'Dropdown-dist', 'Host 2 : byprod. sCAPEX dist.:', 'Host 2'
        hyperparam_list.loc['byproduct_host3_sus_capex_ratio_distribution'] = 'norm', '', '', 'Dropdown-dist', 'Host 3 : byprod. sCAPEX dist.:', 'Host 3'

        hyperparam_list.loc['byproduct_host1_tcrc_ratio_mean'] = 20, 1, 100, 'FloatSlider', 'Host 1 : byprod TCRC mean:', 'Host 1'
        hyperparam_list.loc['byproduct_host2_tcrc_ratio_mean'] = 2, 1, 100, 'FloatSlider', 'Host 2 : byprod TCRC mean:', 'Host 2'
        hyperparam_list.loc['byproduct_host3_tcrc_ratio_mean'] = 10, 1, 100, 'FloatSlider', 'Host 3 : byprod TCRC mean:', 'Host 3'
        hyperparam_list.loc['byproduct_host1_tcrc_ratio_var'] = 1, 0.001, 4, 'FloatSlider', 'Host 1 : byprod TCRC var:', 'Host 1'
        hyperparam_list.loc['byproduct_host2_tcrc_ratio_var'] = 1, 0.001, 4, 'FloatSlider', 'Host 2 : byprod TCRC var:', 'Host 2'
        hyperparam_list.loc['byproduct_host3_tcrc_ratio_var'] = 1, 0.001, 4, 'FloatSlider', 'Host 3 : byprod TCRC var:', 'Host 3'
        hyperparam_list.loc['byproduct_host1_tcrc_ratio_distribution'] = 'norm', '', '', 'Dropdown-dist', 'Host 1 : byprod TCRC dist.:', 'Host 1'
        hyperparam_list.loc['byproduct_host2_tcrc_ratio_distribution'] = 'norm', '', '', 'Dropdown-dist', 'Host 2 : byprod TCRC dist.:', 'Host 2'
        hyperparam_list.loc['byproduct_host3_tcrc_ratio_distribution'] = 'norm', '', '', 'Dropdown-dist', 'Host 3 : byprod TCRC dist.:', 'Host 3'

        hyperparam_list.loc['byproduct_host1_grade_ratio_mean'] = 20, 1,100,'FloatSlider','Host : byprod. grade (mean):', 'Host 1'
        hyperparam_list.loc['byproduct_host2_grade_ratio_mean'] = 2, 1,100,'FloatSlider','Host : byprod. grade (mean):', 'Host 2'
        hyperparam_list.loc['byproduct_host3_grade_ratio_mean'] = 10, 1,100,'FloatSlider','Host : byprod. grade (mean):', 'Host 3'
        hyperparam_list.loc['byproduct_host1_grade_ratio_var'] = 1, 1, 10,'FloatSlider','Host : byprod. grade (var):', 'Host 1'
        hyperparam_list.loc['byproduct_host2_grade_ratio_var'] = 1, 1, 10,'FloatSlider','Host : byprod. grade (var):', 'Host 2'
        hyperparam_list.loc['byproduct_host3_grade_ratio_var'] = 1, 1, 10,'FloatSlider','Host : byprod. grade (var):', 'Host 3'
        hyperparam_list.loc['byproduct_host1_grade_ratio_distribution'] = 'norm','','','Dropdown-dist','Host : byprod. grade (dist.):', 'Host 1'
        hyperparam_list.loc['byproduct_host2_grade_ratio_distribution'] = 'norm','','','Dropdown-dist','Host : byprod. grade (dist.):', 'Host 2'
        hyperparam_list.loc['byproduct_host3_grade_ratio_distribution'] = 'norm','','','Dropdown-dist','Host : byprod. grade (dist.):', 'Host 3'

        hyperparam_list.loc['byproduct_pri_sxew_fraction']   = 0.5, 0, 1, 'FloatSlider', 'Pri SX-EW frac.:', 'Primary mines'
        hyperparam_list.loc['byproduct_host1_sxew_fraction'] = 0.2, 0, 1, 'FloatSlider', 'Host 1 SX-EW frac.:', 'Host 1'
        hyperparam_list.loc['byproduct_host2_sxew_fraction'] = 0.5, 0, 1, 'FloatSlider', 'Host 2 SX-EW frac.:', 'Host 2'
        hyperparam_list.loc['byproduct_host3_sxew_fraction'] = 0.5, 0, 1, 'FloatSlider', 'Host 3 SX-EW frac.:', 'Host 3'
    return hyperparam_list

def set_all(b,buttons,boo):
    for button in buttons:
        button.value = boo

def setup_toggles():
    toggles = list(np.sort([i for i in m.host1.mines.columns if m.host1.mines.dtypes[i] != str]))
    toggles += ['Rec. rate - grade corr.','Cost supply curve','Margin supply curve','Log scale?','Select all','Clear selection']
    buttons = []
    for toggle in toggles:
        if toggle in ['Select all','Clear selection']:
            button = wid.Button(description=toggle,layout=wid.Layout(width='auto', height='40px'))
            if toggle=='Select all':
                button.on_click(functools.partial(set_all,buttons=buttons[:-1],boo=True))
            else:
                button.on_click(functools.partial(set_all,buttons=buttons[:-2],boo=False))
        else:
            button = wid.ToggleButton(
                value=False,
                description=toggle,
                disabled=False,
                button_style='', # 'success', 'info', 'warning', 'danger' or ''
                tooltip='Description',
                icon='',layout=wid.Layout(width='auto', height='40px')
            )
        buttons += [button]
    print('Values to plot:')
    toggles = pd.DataFrame(buttons,toggles,['Result'])
    display(wid.GridBox(list(buttons),layout = wid.Layout(grid_template_columns="repeat(4, 200px)")))
    return toggles
    
def setup_tabs(hyperparam_list):
    tabs = hyperparam_list.Tab.unique()
    children = []

    hyperparam_list.loc[:,'Result'] = np.nan
    width = '350px'
    
    tab_tab = wid.Tab()
    
    for tab in tabs:
        hyper = hyperparam_list.loc[hyperparam_list.Tab==tab]
        items = []
        for i in hyper.index:
            h = hyper.loc[i]

            if h.Type=='Dropdown-bool':
                item = wid.Dropdown(options=[True, False],
                                value=h.Value,
                                description=h.Name,
                                disabled=False,style={'description_width': 'initial'},
                                layout = wid.Layout(width=width,height='40px'))
            elif h.Type=='Dropdown-dist':
                item = wid.Dropdown(options=['norm', 'lognorm', 'weibull_min'],
                                value=h.Value,
                                description=h.Name,
                                disabled=False,style={'description_width': 'initial'},
                                layout = wid.Layout(width=width,height='40px'))
            elif h.Type == 'Dropdown-price':
                item = wid.Dropdown(options=['mean', 'max', 'probabilistic','alonso-ayuso'],
                                value=h.Value,
                                description=h.Name,
                                disabled=False,style={'description_width': 'initial'},
                                layout = wid.Layout(width=width,height='40px'))
            elif h.Type == 'FloatSlider':
                item = wid.FloatSlider(
                                min=h.Min,max=h.Max,step=(h.Max-h.Min)/100,description=h.Name,value=h.Value,
                                style={'description_width': 'initial'},readout_format='.3f',
                                layout = wid.Layout(width=width,height='40px'))
            elif h.Type == 'IntSlider':
                item = wid.IntSlider(
                                min=h.Min,max=h.Max,step=1,description=h.Name,value=h.Value,
                                style={'description_width': 'initial'},
                                layout = wid.Layout(width=width,height='40px'))

            if 'dist' in i:
                pass
            else:
                items += [item]
                hyperparam_list.loc[i,'Result'] = item

        if type(hyperparam_list.loc['close_probability_split_min','Result'])!=float:
            def update_value_min(*args):
                hyperparam_list.loc['close_probability_split_min','Result'].value = 1 -\
                    hyperparam_list.loc['close_probability_split_mean','Result'].value -\
                    hyperparam_list.loc['close_probability_split_max','Result'].value
            hyperparam_list.loc['close_probability_split_mean','Result'].observe(update_value_min,'value')
            hyperparam_list.loc['close_probability_split_max','Result'].observe(update_value_min,'value')
        if type(hyperparam_list.loc['byproduct_host3_production_fraction','Result'])!=float:
            p, h1, h2, h3 = 'byproduct_pri_production_fraction', 'byproduct_host1_production_fraction', 'byproduct_host2_production_fraction', 'byproduct_host3_production_fraction'
            def update_value(*args):
                hyperparam_list.loc[h3,'Result'].value = 1 -\
                    hyperparam_list.loc[p,'Result'].value -\
                    hyperparam_list.loc[h1,'Result'].value -\
                    hyperparam_list.loc[h2,'Result'].value
            hyperparam_list.loc[p,'Result'].observe(update_value,'value')
            hyperparam_list.loc[h1,'Result'].observe(update_value,'value')
            hyperparam_list.loc[h2,'Result'].observe(update_value,'value')
        hyperparam_list = add_tab_reset_button(hyperparam_list, tab, tab_tab)
    
    
        children += [wid.GridBox(list(hyperparam_list.loc[hyperparam_list.Tab==tab].Result.dropna().values), layout=wid.Layout(grid_template_columns="repeat(2, 400px)"))]

    
    tab_tab.children = children
    for i,j in zip(range(len(tabs)),tabs):
        tab_tab.set_title(i, j)
    display(tab_tab)

def add_tab_reset_button(hyperparam_list, tab, tab_tab):
    button = wid.Button(description='Reset this tab to default values',
                       layout=wid.Layout(width='auto', height='40px'))
    def btn_eventhandler(obj):
        for i in hyperparam_list.loc[hyperparam_list.Tab==tabs[tab_tab.selected_index]].dropna().index:
            hyperparam_list.loc[i,'Result'].value = hyperparam_list.loc[i,'Value']
    button.on_click(btn_eventhandler)
    hyperparam_list.loc['button_'+tab,'Result'] = button
    hyperparam_list.loc['button_'+tab,'Tab'] = tab
    return hyperparam_list
    
def add_overall_reset_button(hyperparam_list):
    button = wid.Button(description='Reset all values to default',
                       layout=wid.Layout(width='auto', height='40px'))
    def btn_eventhandler(obj):
        for i in hyperparam_list.dropna().index:
            hyperparam_list.loc[i,'Result'].value = hyperparam_list.loc[i,'Value']
#     display(button)
    button.on_click(btn_eventhandler)
    return button

def make_plot_button(toggles, m, hyperparam_list, out):
    button = wid.Button(description='Plot selected',
                       layout=wid.Layout(width='auto', height='40px'))
#     display(button)
    display(out)
    def btn_eventhandler(obj,m=m,toggles_=toggles):
        with out:
            toggles = toggles_.copy()
            toggles_list = [i for i in toggles.index if 'value' in toggles.loc[i].Result.keys]
            for i in toggles_list:
                toggles.loc[i,'value'] = toggles.loc[i,'Result'].value
            
            toggles = toggles.dropna()
            bools = ['Rec. rate - grade corr.','Cost supply curve','Margin supply curve','Log scale?']
            plot_values = toggles.loc[(toggles.value),'value'].index
            if len(plot_values)>0 or (len(plot_values)>1 and toggles.loc['Log scale?','value']):
                m = GeneralizedOperatingMines(byproduct=True)
                ind = np.intersect1d(m.hyperparam.index,hyperparam_list.dropna().index)
                m.hyperparam.loc[ind,'Value'] = [i.value for i in hyperparam_list.loc[ind].Result]
                m.run()
                fig = m.plot_relevant_params(
                    include=[i for i in plot_values if i not in bools],
                    plot_recovery_grade_correlation = toggles.loc['Rec. rate - grade corr.','value'],
                    plot_minesite_supply_curve = toggles.loc['Cost supply curve','value'],
                    plot_margin_supply_curve = toggles.loc['Margin supply curve','value'],
                    log_scale = toggles.loc['Log scale?','value'],dontplot=True,byproduct=True)
                out.clear_output()
                display(fig)
            elif toggles.loc['Log scale?','value']:
                out.clear_output()
                print('Must select more than just \'Log scale\'')
            else:
                out.clear_output()
                print('Must select at least one parameter')
    button.on_click(functools.partial(btn_eventhandler,m=m,toggles_=toggles))
    return button

def make_run_button(hyperparam_list, out):
    button = wid.Button(description='Run model setup',
                       layout=wid.Layout(width='auto', height='40px'))
#     display(button)
#     display(out)
    def btn_eventhandler(obj):
        with out:
#             out.clear_output()
            print('Mines now exist')
        newmod = GeneralizedOperatingMines(byproduct=True)
        ind = np.intersect1d(newmod.hyperparam.index,hyperparam_list.dropna().index)
        newmod.hyperparam.loc[ind,'Value'] = [i.value for i in hyperparam_list.loc[ind].Result]
        newmod.run()
        return newmod
    newmod = button.on_click(btn_eventhandler)
    return button, newmod

def display_multiple_buttons(buttons):
    display(wid.GridBox(list(buttons),layout = wid.Layout(grid_template_columns="repeat(3, 200px)")))
    
def generate_widgets():
    hyperparam_list = define_hyperparam_list()
    toggles = setup_toggles()
    setup_tabs(hyperparam_list)
    out = wid.Output()
    reset_button = add_overall_reset_button(hyperparam_list)
    run_button, m = make_run_button(hyperparam_list,out)
    plot_button = make_plot_button(toggles, m, hyperparam_list, out)
    display_multiple_buttons([
        reset_button,
        run_button,
        plot_button])