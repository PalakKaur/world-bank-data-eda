# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 14:40:49 2018

@author: Palak Kaur

Working Directory:
/Users/palakkaur/Desktop/PyCourse    

The purpose of this project is to analyze World Bank data of countries in the
Northern Latin America and Caribbean region, flag outliers and investigate
their distinct characteristics. 

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


file = 'world_data_hult_regions(1).xlsx'
old_world= pd.read_excel(file)
file = 'population_df.xls'
df = pd.read_excel(file, header = None)

population_df = df.iloc[3:,]
pop_df = population_df.iloc[1:,]
pop_df.columns = list(population_df.iloc[0])
pop = pop_df.iloc[:, [1,-4]]

merged_df = pd.merge(left = old_world, right = pop, left_on = 'country_code', 
                 right_on='Country Code')

world_df = merged_df.drop('Country Code', axis=1)
world_df['GDP_per_capita'] = (world_df.loc[:, 'gdp_usd']/world_df.iloc[:,-1])
world_df.rename(columns={world_df.columns[-2]:'total_pop'}, inplace=True)


###############################################################################
##### MISSING VALUES
###############################################################################

### SUBSET REGION AS A VARIABLE REGION
region = 'Northern Latin America / Caribbean'
region_df= pd.DataFrame.copy(world_df[world_df['Hult_Team_Regions'] == region])


##### FLAGGING MISSING VALUES

for col in region_df:
    if region_df[col].isnull().astype(int).sum() > 0:
        region_df['m_'+col] = region_df[col].isnull().astype(int)


##### SUBSET WORLD BY INCOME GROUPS

high_inc = 'High income'
whigh_inc_df = world_df[world_df['income_group'] == high_inc]

up_mid_inc = 'Upper middle income'
wup_mid_inc_df = world_df[world_df['income_group'] == up_mid_inc]

low_mid_inc = 'Lower middle income'
wlow_mid_inc_df = world_df[world_df['income_group'] == low_mid_inc]

low_inc = 'Low income'
wlow_inc_df = world_df[world_df['income_group'] == low_inc]


##### SUBSET REGION BY INCOME GROUPS

rhigh_inc_df = pd.DataFrame.copy(
        region_df[region_df['income_group'] == high_inc])

rup_mid_inc_df = pd.DataFrame.copy(
        region_df[region_df['income_group'] == up_mid_inc])

rlow_mid_inc_df = pd.DataFrame.copy(
        region_df[region_df['income_group'] ==low_mid_inc])

rlow_inc_df = pd.DataFrame.copy(
        region_df[region_df['income_group']== low_inc])


##### LIST FOR COLUMNS WHERE WE ARE USING MEDIAN DEPENDING ON INCOME GROUP

ls_world_inc = ['access_to_electricity_pop',
                'access_to_electricity_rural',
                'access_to_electricity_urban',
                'CO2_emissions_per_capita)',
                'pct_agriculture_employment',
                'pct_services_employment',
                'internet_usage_pct',
                'child_mortality_per_1k',
                'exports_pct_gdp']


##### LOOPS TO IMPUT MEDIAN DEPENDING ON INCOME GROUP

for col in rhigh_inc_df[ls_world_inc]:
    if rhigh_inc_df[col].isnull().astype(int).sum() > 0:
        col_median = whigh_inc_df[col].median()
        rhigh_inc_df[col] = rhigh_inc_df[col].fillna(col_median).round(2)

for col in rup_mid_inc_df[ls_world_inc]:
    if rup_mid_inc_df[col].isnull().astype(int).sum() > 0:
        col_median = wup_mid_inc_df[col].median()
        rup_mid_inc_df[col] = rup_mid_inc_df[col].fillna(col_median).round(2)

for col in rlow_mid_inc_df[ls_world_inc]:
    if rlow_mid_inc_df[col].isnull().astype(int).sum() > 0:
        col_median = wlow_mid_inc_df[col].median()
        rlow_mid_inc_df[col] = rlow_mid_inc_df[col].fillna(col_median).round(2)

for col in rlow_inc_df[ls_world_inc]:
    if rlow_inc_df[col].isnull().astype(int).sum() > 0:
        col_median = wlow_inc_df[col].median()
        rlow_inc_df[col] = rlow_inc_df[col].fillna(col_median).round(2)


##### LIST FOR TAX HEAVEN

ls_havens = ['Bermuda', 
             'Netherlands', 
             'Luxembourg', 
             'Cayman Islands', 
              'Singapore', 
              'Channel Islands', 
              'Isle of Man', 
              'Ireland',
              'Bahamas, The', 
              'Switzerland', 
              'Monaco', 
              'Mauritius']


##### IMPUT CAYMAN

tax_havens = world_df[world_df['country_name'].isin(ls_havens)]
cayman_median = int(tax_havens['GDP_per_capita'].median())
rhigh_inc_df.loc[119,'GDP_per_capita'] = int(
        tax_havens['GDP_per_capita'].median())


##### IMPUTE MEDIAN OF gdp_usd and GDP_per_capita

gdp_m = rhigh_inc_df['GDP_per_capita'].median()
rhigh_inc_df['GDP_per_capita'] = rhigh_inc_df[
        'GDP_per_capita'].fillna(gdp_m).round(2)

rhigh_inc_df['gdp_usd'] = (
        rhigh_inc_df.loc[:, 'GDP_per_capita'] * rhigh_inc_df.iloc[:,30])


##### CONCAT SUBSET OF INCOMES

region_df = pd.concat([
        rhigh_inc_df, rup_mid_inc_df, rlow_mid_inc_df, rlow_inc_df])


##### CALCULATION FOR SERVICES

sum_employment = (region_df.loc[:,'pct_agriculture_employment'] 
    + region_df.loc[:,'pct_services_employment'])

region_df['pct_industry_employment'] = (100-sum_employment)


##### HOMICIDES
##### COUNTRY WITH DRUG AND GANG ISSUES

ls_drug = ['Puerto Rico', 
           'Dominican Republic', 
           'Belize', 
           'Jamaica', 
           'Mexico', 
           'Honduras',
           'Guatemala', 
           'Haiti', 
           'Nicaragua']

drug = pd.DataFrame.copy(region_df[region_df['country_name'].isin(ls_drug)])

if drug['homicides_per_100k'].isnull().astype(int).sum() > 0:
   col_median = drug['homicides_per_100k'].median()
   drug['homicides_per_100k'] = (
           drug['homicides_per_100k'].fillna(col_median).round(2))


##### TOURIST HEAVEN

ls_tour = ['British Virgin Islands',
           'Curacao',
           'Turks and Caicos Islands', 
           'Aruba',
           'Cayman Islands', 
           'St. Kitts and Nevis',
           'Virgin Islands (U.S.)',
           'Bahamas, The', 
           'Cuba', 
           'St. Vincent and the Grenadines', 
           'Barbados',
           'Costa Rica']

tourism = pd.DataFrame.copy(region_df[region_df['country_name'].isin(ls_tour)])

if tourism['homicides_per_100k'].isnull().astype(int).sum() > 0:
   col_median = tourism['homicides_per_100k'].median()
   tourism['homicides_per_100k'] = (
           tourism['homicides_per_100k'].fillna(col_median).round(2))

region_df = pd.concat([drug,tourism]) #CONCAT TO HAVE WHOLE REGION


##### LIST FOR THE REST OF COLUMNS

ls=['compulsory_edu_yrs',
    'pct_female_employment',
    'pct_male_employment',
    'fdi_pct_gdp',
    'incidence_hiv',
    'adult_literacy_pct',
    'avg_air_pollution',
    'women_in_parliament',
    'tax_revenue_pct_gdp',
    'unemployment_pct', 
    'gdp_growth_pct']


#####lOOPS TO IMPUTE MEDIAN OF REGION

for col in region_df[ls]:
    if region_df[col].isnull().astype(int).sum() > 0:
        col_median = region_df[col].median()
        region_df[col] = region_df[col].fillna(col_median).round(2)


###############################################################################
##### PLOTS - EXPLORATORY ANALYSIS
###############################################################################
##### BOXPLOTS   

for col in region_df.iloc[:, 5:32]:
   region_df.boxplot(column = col, vert = False)
   plt.title(f"{col}")
   plt.tight_layout()
   plt.show()
    
##### By income group

for col in region_df.iloc[:, 5:32]:
   region_df.boxplot(column = col, by = 'income_group')
   plt.title(f"{col} by income group")
   plt.xticks(rotation = 45)
   plt.tight_layout()
   plt.suptitle("")
   plt.show()
   
###### Histograms with distribution plots

for col in region_df.iloc[:, 5:32]:
    sns.distplot(region_df[col], bins = 'fd')
    plt.tight_layout()
    plt.show()
    
    
###############################################################################
##### PLOTS - OUTLIERS
###############################################################################

##### Total electricity - mean is outside the box

region_df.boxplot(column = ['access_to_electricity_pop'],
                 vert = False,
                 meanline = True,
                 showmeans = True)
plt.axvline(89, color='purple', linestyle = '--')
plt.show()

sns.distplot(region_df['access_to_electricity_pop'])
plt.show()


##### Rural electricity - mean is outside the box

region_df.boxplot(column = ['access_to_electricity_rural'],
                 vert = False,
                 meanline = True,
                 showmeans = True)
plt.axvline(88, color='purple', linestyle = '--')
plt.show()

sns.distplot(region_df['access_to_electricity_rural'])
plt.axvline(88, color='purple', linestyle = '--')
plt.show()


##### Urban electricity - mean way outside the box

region_df.boxplot(column = ['access_to_electricity_urban'],
                 vert = False,
                 meanline = True,
                 showmeans = True)
plt.axvline(97, color = 'purple', linestyle = '--')
plt.show()

sns.distplot(region_df['access_to_electricity_urban'])
plt.axvline(97, color='purple', linestyle = '--')
plt.show()


##### CO2 emission  

region_df.boxplot(column = ['CO2_emissions_per_capita)'],
                 vert = False,
                 meanline = True,
                 showmeans = True)
plt.axvline(9, color='purple', linestyle = '--')
plt.show()

sns.distplot(region_df['CO2_emissions_per_capita)'])
plt.axvline(9, color='purple', linestyle = '--')
plt.show()


##### Compulsory education

region_df.boxplot(column = ['compulsory_edu_yrs'],
                 vert = False,
                 meanline = True,
                 showmeans = True)
plt.show()

sns.distplot(region_df['compulsory_edu_yrs'])
plt.axvline(11.5, color='green', linestyle = '--')
plt.show()      


##### Female employement

region_df.boxplot(column = ['pct_female_employment'],
                 vert = False,
                 meanline = True,
                 showmeans = True)
plt.show()

sns.distplot(region_df['pct_female_employment'])
plt.axvline(8, color='green', linestyle = '--')
plt.show()  


##### Male employement

region_df.boxplot(column = ['pct_male_employment'],
                 vert = False,
                 meanline = True,
                 showmeans = True)
plt.show()

sns.distplot(region_df['pct_male_employment'])
plt.tight_layout()
plt.axvline(3.5, color='green', linestyle = '--')
plt.show()  


##### Agriculture employement

region_df.boxplot(column = ['pct_agriculture_employment'],
                 vert = False,
                 meanline = True,
                 showmeans = True)
plt.axvline(28.5, color='red', linestyle = '--')
plt.show()

sns.distplot(region_df['pct_agriculture_employment'], bins = 25, kde = False)
plt.tight_layout()
plt.axvline(28.5, color='green', linestyle = '--')
plt.show()  


##### Industry - no outliers?

region_df.boxplot(column = ['pct_industry_employment'],
                 vert = False,
                 meanline = True,
                 showmeans = True)
plt.show()

sns.distplot(region_df['pct_industry_employment'], bins = 15, kde = False)
plt.tight_layout()
plt.show()  

        
##### Services employement

region_df.boxplot(column = ['pct_services_employment'],
                 vert = False,
                 meanline = True,
                 showmeans = True)
plt.axvline(51, color='red', linestyle = '--')
plt.show()

sns.distplot(region_df['pct_services_employment'], bins = 25, kde = False)
plt.tight_layout()
plt.axvline(51, color='green', linestyle = '--')
plt.show()  


############ Gender employment subplot #######################

plt.subplot(2,2,1) #male
sns.distplot(region_df['pct_male_employment'],color='blue')
plt.xlabel('Male Employment')
plt.axvline(4.05, color='green', linestyle = '--')

plt.subplot(2,2,2) #female
sns.distplot(region_df['pct_female_employment'],color='pink')
plt.xlabel('Female Employment')
plt.axvline(7.5, color='green', linestyle = '--')

plt.tight_layout()
plt.savefig('Gender emp.png')
plt.show()  


################ Sectors employment #######################

plt.subplot(2,2,1)
sns.distplot(region_df['pct_agriculture_employment'], bins=20,color='purple')
plt.axvline(33, color='green', linestyle = '--')
plt.tight_layout()
plt.xlabel('% Employment in Agricultural Sector')

plt.subplot(2,2,2)
sns.distplot(region_df['pct_services_employment'], bins=20,color='orange')
plt.axvline(51, color='green', linestyle = '--')
plt.tight_layout()
plt.xlabel('% Employment in Services Sector')

plt.subplot(2,2,3)
sns.distplot(region_df['pct_industry_employment'], bins=15)
plt.tight_layout()
plt.xlabel('% Employment in Industrial Sector')

plt.savefig('Sector employment.png')
plt.show()  

################ Electricity employment #######################

plt.subplot(2,2,1)
sns.distplot(region_df['access_to_electricity_rural'], bins=20,color='purple')
plt.axvline(88, color='green', linestyle = '--', label='Lower outlier')
plt.tight_layout()
plt.xlabel('% Rural Access')

plt.subplot(2,2,2)
sns.distplot(region_df['access_to_electricity_urban'], bins=20,color='orange')
plt.axvline(98, color='green', linestyle = '--', label='Lower outlier')
plt.tight_layout()
plt.xlabel('% Urban Access')

plt.subplot(2,2,3)
sns.distplot(region_df['access_to_electricity_pop'], bins=15)
plt.axvline(89, color='green', linestyle = '--', label='Lower outlier')
plt.tight_layout()
plt.xlabel('% Total Access')

plt.savefig('Access to Electricity.png')
plt.tight_layout()
plt.show()  

############################# GDP #################################

##### GDP exports

region_df.boxplot(column = ['exports_pct_gdp'],
                 vert = False,
                 meanline = True,
                 showmeans = True)
plt.axvline(75, color='red', linestyle = '--')

plt.show()

sns.distplot(region_df['exports_pct_gdp'], bins=30)
plt.tight_layout()
plt.axvline(75, color='green', linestyle = '--')
plt.show()  


##### GDP usd

region_df.boxplot(column = ['gdp_usd'],
                 vert = False,
                 meanline = True,
                 showmeans = True)
plt.axvline(140000000000, color='red', linestyle = '--')
plt.show()

sns.distplot(region_df['gdp_usd'], bins=30)
plt.axvline(140000000000, color='green', linestyle = '--')
plt.tight_layout()
plt.show()  


##### GDP growth

region_df.boxplot(column = ['gdp_growth_pct'],
                 vert = False,
                 meanline = True,
                 showmeans = True)
plt.axvline(7.3, color='red', linestyle = '--')
plt.show()

region_df.hist('gdp_growth_pct',grid=False,alpha=0.3)
plt.axvline(7.3, color='green', linestyle = '--')
plt.show()

sns.distplot(region_df['gdp_growth_pct'], bins=20)
plt.axvline(7.3, color='green', linestyle = '--')
plt.tight_layout()
plt.show()  

##### HIV

region_df.boxplot(column = ['incidence_hiv'],
                 vert = False,
                 meanline = True,
                 showmeans = True)
plt.axvline(6, color='red', linestyle = '--')
plt.show()

region_df.hist('incidence_hiv',grid=False,alpha=0.3)
plt.axvline(0.10, linestyle= '--', color='r')
plt.title('HIV Rate')
plt.savefig('HIV.png')
plt.show()


##### Homicides

region_df.boxplot(column = ['homicides_per_100k'],
                 vert = False,
                 meanline = True,
                 showmeans = True)
plt.axvline(36, color='red', linestyle = '--')
plt.show()

sns.distplot(region_df['homicides_per_100k'], 
             bins = 15)
plt.axvline(36, linestyle= '--', color='r')
plt.title('Homicides per 100k')
plt.savefig('Homi.png')
plt.show()


####### GDP per capita

region_df.boxplot(column = ['GDP_per_capita'],
                 vert = False,
                 meanline = True,
                 showmeans = True)
plt.axvline(53500, color='red',linestyle = '--')
plt.show()

sns.distplot(region_df['GDP_per_capita'], 
             bins = 15)
plt.axvline(53500, linestyle= '--', color='r')
plt.show()


###### Subplots for fdi, exports, GDP usd, GDP %

plt.subplot(2,2,1)
sns.distplot(region_df['fdi_pct_gdp'], 
             bins = 'fd', color = 'purple')
plt.axvline(12.5, linestyle= '--', color='green')
plt.xlabel("% FDI GDP")

plt.subplot(2,2,2)
sns.distplot(region_df['exports_pct_gdp'], 
             bins=30, color = 'red')
plt.axvline(75, color='green', linestyle = '--')
plt.xlabel("% Exports GDP")

plt.subplot(2,2,3)
sns.distplot(region_df['gdp_growth_pct'], 
             bins=20, color = 'orange')
plt.axvline(7.3, color='green', linestyle = '--')
plt.xlabel("% GDP Growth")

plt.subplot(2,2,4)
sns.distplot(region_df['gdp_usd'], 
             bins='fd')
plt.axvline(140000000000, color='green', linestyle = '--')
plt.xlabel("GDP USD")

plt.tight_layout()
plt.savefig("GDP.jpg")
plt.show()


##### Child Mortality

region_df.boxplot(column = ['child_mortality_per_1k'],
                 vert = False,
                 meanline = True,
                 showmeans = True)
plt.axvline(34.5, color='green', linestyle = '--')
plt.show()

sns.distplot(region_df['child_mortality_per_1k'])
plt.axvline(32, color='green', linestyle = '--')
plt.show() 

##### Air Pollution  
   
sns.distplot(region_df['avg_air_pollution'])
plt.axvline(23, color='green', linestyle = '--')
plt.xlabel('avg_air_pollution')
plt.tight_layout()
plt.show()      

      
##### Women in Parliament    

region_df.boxplot(column = ['women_in_parliament'],
                 vert = False,
                 meanline = True,
                 showmeans = True)
plt.xlabel('Women in parliament')
plt.show()

sns.distplot(region_df['women_in_parliament'],
             bins = 20,
             color = 'y')
plt.xlabel('Women in Parliament')
plt.axvline(27, linestyle = '--', color = 'purple')
plt.axvline(2, linestyle = '--', color = 'red')
plt.tight_layout() 
plt.show()


##### Tax Revenue % GDP

region_df.boxplot(column = ['tax_revenue_pct_gdp'],
                 vert = False,
                 meanline = True,
                 showmeans = True)
plt.xlabel('Tax revenue %GDP')
plt.show()

sns.distplot(region_df['tax_revenue_pct_gdp'],
             bins = 20,
             color = 'b')
plt.xlabel('Tax Percentage of GDP')
plt.axvline(14.0, linestyle = '--', color = 'red')
plt.axvline(16.8, linestyle = '--', color = 'red')
plt.tight_layout()
plt.show()


##### Unemployement %

region_df.boxplot(column = ['unemployment_pct'],
                 vert = False,
                 meanline = True,
                 showmeans = True)  
plt.xlabel('Unemployment %')
plt.show()


##### Urban Population Growth %anual

region_df.boxplot(column = ['urban_population_growth_pct'],
                 vert = False,
                 meanline = True,
                 showmeans = True)
plt.xlabel('Urban population growth %')
plt.show()


##### Total population

region_df.boxplot(column = ['total_pop'],
                 vert = False,
                 meanline = True,
                 showmeans = True)
plt.xlabel('Total population')
plt.show()

sns.distplot(region_df['total_pop'],
             bins = 20,
             color = 'g')
plt.xlabel('Total Population')
plt.axvline(16500000, linestyle = '--', color = 'red')
plt.tight_layout()
plt.show()

###############################################################################
##### FLAG OUTLIERS
###############################################################################

##### RESET INDEX TO FURTHER DEFINE FUNCTIONS

region_df = region_df.reset_index() 
region_df = region_df.drop(['index'], axis=1)

def low_out(col,lim):
    region_df['o_'+col] = 0
    for val in enumerate(region_df.loc[ : , col]):   
        if val[1] < lim:
            region_df.loc[val[0], 'o_'+col] = 1

def up_out(col,lim):
    region_df['o_'+col] = 0
    for val in enumerate(region_df.loc[ : , col]):   
        if val[1] > lim:
            region_df.loc[val[0], 'o_'+col] = 1    

    
##### FLAGGIN LOWER OUTLIERS

low_out('access_to_electricity_pop', 89)
low_out('access_to_electricity_pop', 89)
low_out('access_to_electricity_rural', 88)
low_out('access_to_electricity_urban', 97)


##### FLAGGIN UPPER OUTLIERS

up_out('CO2_emissions_per_capita)', 9)
up_out('pct_female_employment', 8)
up_out('pct_male_employment', 4.05)
up_out('pct_agriculture_employment', 33)
up_out('incidence_hiv', 0.10)
up_out('homicides_per_100k', 36)
up_out('child_mortality_per_1k', 34.5)
up_out('avg_air_pollution', 23)
up_out('total_pop', 16500000)
up_out('GDP_per_capita', 53500)


###############################################################################
##### CORRELATION
###############################################################################

corr_region = region_df.corr().round(2)

###############################################################################
##### STRONG CORRELATION GRAPH
###############################################################################

def graph(X,Y):
    sns.lmplot(x = X,
           y = Y,
           data = region_df,
           fit_reg = True,
           scatter_kws= {"marker": "D","s": 30},
           palette = 'plasma')
    plt.grid()
    plt.savefig(f'{X} vs {Y}.jpg')
    plt.show() 


graph('pct_agriculture_employment', 'access_to_electricity_pop')
graph('pct_agriculture_employment', 'child_mortality_per_1k')
graph('pct_agriculture_employment', 'avg_air_pollution')
graph('pct_agriculture_employment', 'internet_usage_pct')
graph('pct_agriculture_employment', 'GDP_per_capita')
graph('pct_services_employment', 'avg_air_pollution')
graph('pct_services_employment', 'homicides_per_100k')
graph('GDP_per_capita', 'internet_usage_pct')
graph('access_to_electricity_pop', 'child_mortality_per_1k')


#############################################################################
# END OF SCRIPT
###############################################################################
