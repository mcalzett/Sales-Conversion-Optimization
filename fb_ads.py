"""The data used in this project is from an anonymous organisation’s social media ad campaign.

The file conversion_data.csv contains 1143 observations in 11 variables. Below are the descriptions of the variables.

1.) ad_id: an unique ID for each ad.

2.) xyz_campaign_id: an ID associated with each ad campaign of XYZ company.

3.) fb_campaign_id: an ID associated with how Facebook tracks each campaign.

4.) age: age of the person to whom the ad is shown.

5.) gender: gender of the person to whim the add is shown

6.) interest: a code specifying the category to which the person’s interest belongs (interests are as mentioned in the person’s Facebook public profile).

7.) Impressions: the number of times the ad was shown.

8.) Clicks: number of clicks on for that ad.

9.) Spent: Amount paid by company xyz to Facebook, to show that ad.

10.) Total conversion: Total number of people who enquired about the product after seeing the ad.

11.) Approved conversion: Total number of people who bought the product after seeing the ad.
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm 



os.chdir("/Users/mattiacalzetta/Desktop")
df=pd.read_csv("data.csv")
df=df.replace({'30-34':32, '35-39':37,'40-44': 42,'45-49':47,'F':0,'M':1})

#We have data for 3 distinct campaigns of XYZ company
list=df.xyz_campaign_id.unique()

#Let's focus on campaign id "936" for example
campaign_936=df[df['xyz_campaign_id']==936]

#Drop uninformative columns
campaign_936=campaign_936.drop(columns=['xyz_campaign_id','ad_id','fb_campaign_id'])

#--------------------------------------------------------------------------------
#Build KPIs

#Create Cost Per Mille - the cost per one thousand impressions
campaign_936['CPM']=(campaign_936.loc[:,'Spent']/campaign_936.loc[:,'Impressions'])*1000

#Create Click Through Rate - the percentage of impressions that become clicks
campaign_936['CTR']=(campaign_936.loc[:,'Clicks']/campaign_936.loc[:,'Impressions'])*100

#Create Conversion Rate - the percentage of clicks that become a conversion
campaign_936['CR_Approved']=(campaign_936.loc[:,'Approved_Conversion']/campaign_936.loc[:,'Clicks'])*100
campaign_936['CR_Total']=(campaign_936.loc[:,'Total_Conversion']/campaign_936.loc[:,'Clicks'])*100

#Create Cost Per Click - the average cost associated with each click
campaign_936['CPC']=(campaign_936.loc[:,'Spent']/campaign_936.loc[:,'Clicks'])

campaign_936=campaign_936.dropna()


#--------------------------------------------------------------------------------
#Create Heatmap

# Compute the correlation matrix
corr = campaign_936.corr()

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)


# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)


#--------------------------------------------------------------------------------
#Quick multiple linear regression

X = campaign_936[['gender','interest','age']]
y = campaign_936['CTR']

model = sm.OLS(y, X).fit()
predictions = model.predict(X)

print("Summary of the multiple linear regression model:")
model.summary()


