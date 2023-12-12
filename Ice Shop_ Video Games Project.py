#!/usr/bin/env python
# coding: utf-8

# ## Project Description
# 
# You work at the online store "Ice," which sells video games from around the world. Data related to user and expert reviews, genres, platforms (such as Xbox or PlayStation), and historical sales data are available from open sources. In front of you is the data from the year 2016. Let's imagine that it is now December 2016, and you are planning a campaign for the year 2017.
# 
# (Currently, the most important thing for you is to gain experience working with data. It doesn't matter whether you are forecasting 2017 sales based on 2016 data or predicting 2027 sales based on 2026 data.).
# 
# This dataset includes abbreviations. ESRB stands for the Entertainment Software Rating Board, an independent regulatory organization that evaluates game content and assigns age ratings such as Teen or Mature.
# 
# **Project Goal: You need to identify patterns that determine whether a game can be considered successful or not. In doing so, you can find the most promising games and plan advertising campaigns accordingly.**

# ## Import Library
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st
import random
from math import factorial
import math


# ### Import Data Information

# In[2]:


df = pd.read_csv('/datasets/games.csv')


# In[3]:


df


# In[4]:


df.info()


# ## Fixing and Initial Data Exploration

# In[5]:


df.columns = df.columns.str.lower()

df.sample(50)


# First, let's change the column names to all lowercase. Once done, we need to modify data types. We should update the "year_of_release" column, but we can't do that yet as some data in the "year_of_release" is missing. There are missing values in the "year_of_release" column, and it seems the quantity is not significant compared to the total data. We can replace these missing values with 0, as well as in the "name" and "genre" columns.

# In[6]:


list_columns = ['name','year_of_release','genre']

print(df[list_columns].isna().sum())


# In[7]:


for data in list_columns:
    df[data] = df[data].fillna(0)


# We need to change "year_of_release" to an integer because we have just filled in the missing values with 0. If we change it to datetime, it will result in an error.

# In[8]:


df['year_of_release'] = df['year_of_release'].astype(int)
df.info()


# Now, let's examine the sales section to see if there's anything that needs fixing.

# In[9]:


df.describe()


# From the data description above, many games did not make any sales at all. If we are selecting those with potential for the future, those with zero sales everywhere do not need further analysis; we will leave them aside since the data for them is already complete with no "NaN" values.
# 
# Now, let's proceed to the "critic_score," "user_score," and "rating" columns.

# In[10]:


list_column2 = ["critic_score","user_score","rating"]
print(df[list_column2].isna().sum())  


# The missing data is quite substantial; we need to try breaking down its content.

# In[11]:


df['critic_score'].value_counts()


# In[12]:


df['user_score'].value_counts()


# In[13]:


df['rating'].value_counts()


# For 'critic_score,' it seems there is no issue with the distribution; a significant amount of data is missing. We don't have references to fill in this missing data, possibly because the game hasn't been reviewed. Even if we were to calculate averages based on genre, it wouldn't be valid as there might be data with no sales but a 'critic_score.'
# 
# Regarding ratings, we need to understand the explanations for each rating:
# 
# - E = Everyone (suitable for all ages)
# - T = Teen (game is suitable for ages 13 and older)
# - M = Mature 17+ (game is suitable for ages 17 and older)
# - E10+ = Everyone 10+ (game is suitable for ages 10 and older)
# - EC = Early Childhood (made for children before school age)
# - K-A = Kids to Adults (essentially the same as E, but for older games, updated to E)
# - RP = Rating Pending (the game's final rating is yet to be determined)
# - AO = Adults Only 18+ (game is specifically for adults, 18+)
# 
# From the 'rating' data, it seems that sellers often choose games with ratings E, T, M, E10+.
# 
# For 'user_score,' many games still have TBD (To Be Determined). We need to review the TBD entries to understand what characterizes them.
# 
# We found a pattern in the data: for games with 'user_score' as "NaN," it can be confirmed that 'rating' and 'critic_score' are also "NaN." This could be because the game has never been played, hence no critic and user scores. We will also examine games with 'user_score' as "NaN."

# In[14]:


df.query('user_score == "tbd"')


# For "user_score," the TBD data needs to be utilized, indicating that this data is still in the process of evaluation.

# In[15]:


df[df['user_score'].isna()].sample(20)


# If we look at the data above, for "user_score," a rating of "NaN" indicates that the game has not been played. However, we should not discard it because there are still sales data available.

# ## Improve Data Quality

# We need to calculate the total sales (sum of sales in all regions) for each game and enter these values into separate columns.

# In[16]:


df['total_sales'] = df['na_sales']+ df['eu_sales'] + df['jp_sales']+df['other_sales']

df


# The data above is the final dataset that we will analyze further.

# ## Deeper Exploratory Data Analysis

# We need to find out how many games were released in different years using `.value_counts()`.

# In[17]:


df['year_of_release'].value_counts()


# Many games were significantly released above the year 2000.
# 
# 

# We need to analyze how sales vary from one platform to another. It's necessary to create a new dataset containing platforms and create its distribution based on yearly data. We can achieve this by creating a pivot table.

# In[18]:


df_platform_year_sales = pd.pivot_table(df,index=["year_of_release"],columns = "platform", values = "total_sales",aggfunc = "sum")
df_platform_year_sales = df_platform_year_sales.reset_index()

df_platform_year_sales


# In[19]:


df_platform_year_sales = df_platform_year_sales.query("year_of_release > 0")

df_platform_year_sales.plot(kind="line", figsize = (25,10), x = "year_of_release")


# Looking at the data above, it represents all sales each year per platform, and there are numerous platforms. We need to identify which platform has the highest sales, necessitating the creation of another pivot table.

# In[20]:


df_platform_year_sales_max = pd.pivot_table(df,index=["platform"],values = "total_sales",aggfunc = "sum")
df_platform_year_sales_max = df_platform_year_sales_max.reset_index()

df_platform_year_sales_max.sort_values('total_sales', ascending= False)


# It turns out that PS2 has the highest sales. We need to examine its distribution from year to year.

# In[21]:


df_ps2 = df.query('platform == "PS2"')
df_ps2


# Now, let's distribute the data per year.

# In[22]:


df_ps2_pivot = pd.pivot_table(df_ps2, index = "year_of_release" , values = "total_sales", aggfunc = 'sum')
df_ps2_pivot


# In[23]:


df_ps2_pivot.plot(kind = 'bar')


# We can see that the platform that was once popular is "PS2," but now it has either no sales or very few. If we look at the data above, the "PS2" platform was present for approximately 10 years, with the first 5 years being very popular and the following 5 years experiencing a decline in sales. We need to know how long it generally takes for a new platform to emerge and an old platform to fade in popularity. To find the average time, we need to create a new dataframe, find the Min and Max year_of_release for each platform, and then calculate the average. Since some platforms have a release year of 0, which could introduce bias, we need to filter them out first.

# In[24]:


df_mean_year_platform = df.query('year_of_release > 0')
df_mean_year_platform


# In[25]:


df_mean_years_platform_pivot = pd.pivot_table(df_mean_year_platform, values ="year_of_release", index = 'platform', aggfunc=["min","max"])

df_mean_years_platform_pivot = df_mean_years_platform_pivot.reset_index()

df_mean_years_platform_pivot

df_mean_years_platform_pivot['mean_year_different'] = (df_mean_years_platform_pivot['max'] - df_mean_years_platform_pivot['min'])/2

df_mean_years_platform_pivot


# In[26]:


print(df_mean_years_platform_pivot['mean_year_different'].mean())


# We can see that the average lifespan of a platform, from emergence to decline, is 4 years. This implies that when we want to make predictions for 2017, we need to create a new dataset and select the data from the last 4 years.

# ## Clean Data

# In[27]:


df_new = df.query("year_of_release > 2012")
df_new.sample(15)


# ### Clean Data Analysis

# In[28]:


df_new_platform_sales = pd.pivot_table(df_new,index = "platform" , values = "total_sales" ,aggfunc="sum").reset_index()

df_new_platform_sales.sort_values('total_sales',ascending=False)


# We can see that the PS4 has the highest sales compared to other platforms. However, we need to examine its year-wise distribution to gain a better understanding.

# In[29]:


df_new_platform_year_sales = pd.pivot_table(df_new,index=["platform"],columns = "year_of_release", values = "total_sales",aggfunc = "sum")
df_new_platform_year_sales = df_new_platform_year_sales.reset_index()

df_new_platform_year_sales


# In[30]:


df_new_platform_year_sales.plot(kind="bar", figsize = (10,5), x = "platform")


# The platform with the highest sales is PS4, and we can observe the growth or decline of various platforms from the graph above. XOne and PS4P experienced growth and then a decline in 2016, while others have been in decline since 2015. If we look at the potential for profit, PS4 and XOne seem promising, despite a decline in 2016, as there is still an opportunity for sales.

# In[31]:


df_new_platform_year_sales_mean = pd.pivot_table(df_new,index=["platform"],columns = "year_of_release", values = "total_sales",aggfunc = "mean")
df_new_platform_year_sales_mean = df_new_platform_year_sales_mean.reset_index()

df_new_platform_year_sales_mean


# In[32]:


df_new_platform_year_sales_boxplot = pd.pivot_table(df_new,index=["year_of_release"],columns = "platform", values = "total_sales",aggfunc = "sum")
df_new_platform_year_sales_boxplot = df_new_platform_year_sales_boxplot.reset_index()

df_new_platform_year_sales_boxplot.boxplot( column = ['3DS','PC','PS3','PS4','PSP','PSV','Wii','WiiU','X360','XOne'], figsize =(10,5))


# The differences in sales vary for each platform. For PSP, it seems there are no more sales, and the significant differences are noticeable for PS3, PS4, XOne, and X360 between the min and max. The average sales for each platform from year to year also tend to decrease, as seen in "df_new_platform_year_sales_mean." For further analysis, we will focus on PS4 as it has the highest sales. We will examine its correlation with ratings.

# In[33]:


df_ps4_scatter = df_new[df_new['platform'] == "PS4"]
df_ps4_scatter


# In[34]:


df_ps4_scatter.plot(kind='scatter',x='critic_score',y='total_sales',grid=True)


# We can see from the table above that for critic_score, the higher the score, the likelihood of sales and revenue increases. This indicates an influence on game/platform sales. Now, let's examine user_score, but we need to check if there are values with "tbd" and how many there are.

# In[35]:


print(len(df_ps4_scatter[df_ps4_scatter['user_score'] ==  'tbd']))


# There are 6 out of 376 data with "tbd," meaning we can still analyze to determine whether it also has an impact.

# In[36]:


df_ps4_scatter_wo_tbd = df_ps4_scatter.query('user_score != "tbd"')
df_ps4_scatter_wo_tbd = df_ps4_scatter_wo_tbd.dropna(subset=['user_score'])
df_ps4_scatter_wo_tbd['user_score'] = df_ps4_scatter_wo_tbd['user_score'].astype(float)
df_ps4_scatter_wo_tbd


# Additionally, we also need to remove "NaN" values in user_score so that correlation can be calculated.

# In[37]:


df_ps4_scatter_wo_tbd.plot(kind='scatter',x='user_score',y='total_sales',grid=True)


# The scatter plot results from "df_ps4_scatter_wo_tbd" also indicate that as user score increases, there is a higher chance of sales for the game/platform. Now, let's substantiate this observation with numerical data.

# In[38]:


print(df_ps4_scatter['critic_score'].corr(df_ps4_scatter['total_sales']))
print(df_ps4_scatter_wo_tbd['user_score'].corr(df_ps4_scatter_wo_tbd['total_sales']))


# From the results above, it is evident that critic_score has a strong positive correlation with sales, while user_score has a negative correlation with sales, although the correlation is not as strong. This implies that the score from professional critics has a significant impact on sales, whereas user scores, in general, do not contribute significantly to sales.

# Sure, you can create scatter plots for other platforms

# In[39]:


listed_columns = ['3DS','PC','PS3','PS4','PSP','PSV','Wii','WiiU','X360','XOne']

for data in listed_columns:
    df_columns_scatter = df_new[df_new['platform'] == data ]
    df_columns_scatter.plot(kind='scatter',x='critic_score',y='total_sales',grid=True, label = data) 
    print(print("korelasi terhadap professional:",data),df_columns_scatter['critic_score'].corr(df_columns_scatter['total_sales']))
    df_columns_scatter_wo_tbd = df_columns_scatter.query('user_score != "tbd"')
    df_columns_scatter_wo_tbd = df_columns_scatter_wo_tbd.dropna(subset=['user_score'])
    df_columns_scatter_wo_tbd['user_score'] = df_columns_scatter_wo_tbd['user_score'].astype(float)
    df_columns_scatter_wo_tbd.plot(kind='scatter',x='user_score',y='total_sales',grid=True, label = data)
    print(print("korelasi user:",data),df_columns_scatter_wo_tbd['user_score'].corr(df_columns_scatter_wo_tbd['total_sales']))

print(data)


# We have obtained scatter plots and correlations for each platform, and we need to dissect:
# 
# * Correlation with professionals: 3DS, 0.3570566142288103
#   Correlation with users: 3DS, 0.24150411773563016
# 
# * Correlation with professionals: PC, 0.19603028294369382
#   Correlation with users: PC, -0.0938418695247674
# 
# * Correlation with professionals: PS3, 0.3342853393371919
#   Correlation with users: PS3, 0.0023944027357566925
# 
# * Correlation with professionals: PS4, 0.40656790206178095
#   Correlation with users: PS4, -0.031957110204556424
# 
# * Correlation with professionals: PSP, None (NaN)
#   Correlation with users: PSP, -0.9999999999999999
# 
# * Correlation with professionals: PSV, None (NaN)
#   Correlation with users: PSV, 0.2547423503068656
# 
# * Correlation with professionals: Wii, None (NaN)
#   Correlation with users: Wii, 0.6829417215362368
# 
# * Correlation with professionals: WiiU, None, 0.3764149065423912
#   Correlation with users: WiiU, 0.4193304819266187
# 
# * Correlation with professionals: X360, None, 0.3503445460228664
#   Correlation with users: X360, -0.011742185147181342
# 
# * Correlation with professionals: XOne, None, 0.4169983280084017
#   Correlation with users: XOne, -0.06892505328279414
# 
# In general, the correlation with professionals tends to have a positive correlation compared to user correlations, with some being negative. Some platforms have strong correlations, while others do not.
# 3DS, PS3, PS4, WiiU, X360, and XOne have a strong correlation for professional assessments. For user assessments, 3DS, Wii, and WiiU have a strong and positive correlation, while the rest have a negative and not strong enough correlation.

# Now we need to dissect the genre section, identifying which genre is the most popular and widely purchased.

# In[40]:


df_new_genre = pd.pivot_table(df_new,index = "genre", values = 'total_sales', aggfunc = 'sum')
df_new_genre.reset_index().sort_values('total_sales', ascending = False)


# For genres, we can observe that Action is the most popular genre among all genres.

# ### Analysis of User Profiling for Each Region

# In[41]:


df_new


# In[42]:


df_new_genre = pd.pivot_table(df_new, index = "genre",values = ["na_sales","eu_sales","jp_sales","other_sales","total_sales"],aggfunc = 'sum')
df_new_genre.reset_index().sort_values('total_sales', ascending = False)


# For the genre that generates the highest sales when broken down by region, in the EU region, Action is the genre that generates the most revenue, followed by Shooter. In the JP region, Role-Playing is the highest revenue-generating genre, followed by Action. In the NA region, Action is the genre that generates the highest revenue, followed by Shooter.

# In[43]:


df_new_genre = pd.pivot_table(df_new, index = "platform",values = ["na_sales","eu_sales","jp_sales","other_sales","total_sales"],aggfunc = 'sum')
df_new_genre.reset_index().sort_values('total_sales', ascending = False)


# For the platform that generates the highest sales when broken down by region, in the EU region, PS4 is the platform that generates the most revenue, followed by PS3. In the JP region, 3DS is the highest revenue-generating platform, followed by PS3. In the NA region, PS4 is the platform that generates the highest revenue, followed by XOne.

# In[44]:


df_new_genre = pd.pivot_table(df_new, index = "rating",values = ["na_sales","eu_sales","jp_sales","other_sales","total_sales"],aggfunc = 'sum')
df_new_genre.reset_index().sort_values('total_sales', ascending = False)


# Certainly, here's a recap of the explanations for each rating:
# 
# - E = Everyone (suitable for all ages)
# - T = Teen (game can be played by teenagers, minimum age 13+)
# - M = Mature 17+ (game can be played for individuals aged 17 and older)
# - E10+ = Everyone 10+ (game can be played for individuals aged 10 and older)

# It turns out that the rating that generates the highest revenue is Rating M, which means games suitable for individuals aged 17 and older, followed by Rating E (suitable for all ages), then T (teenagers), and E10+ (everyone aged 10 and older). When looking at the sales distribution, Rating M and E could be preferred choices.
# 
# For the rating that generates the highest sales when broken down by region:
# - In the EU region, Rating M generates the most revenue, followed by Rating E.
# - In the JP region, Rating T is the highest revenue-generating, followed by Rating E.
# - In the NA region, Rating M generates the highest revenue, followed by Rating E.

# ## Hypothesis Testing

# Here is the hypothesis that needs further investigation:
# 
# * The average user_rating for the Xbox One and PC platforms is the same.

# In[45]:


df_Xbox_One_hip = df_new[df_new['platform'] == "XOne"]
df_Xbox_One_hip = df_Xbox_One_hip.query('user_score != "tbd"')
df_Xbox_One_hip = df_Xbox_One_hip.dropna(subset=['user_score'])
df_Xbox_One_hip = df_Xbox_One_hip['user_score'].astype(float)


# In[46]:


df_PC_hip = df_new[df_new['platform'] == "PC"]
df_PC_hip = df_PC_hip.query('user_score != "tbd"')
df_PC_hip = df_PC_hip.dropna(subset=['user_score'])
df_PC_hip = df_PC_hip['user_score'].astype(float)


# In[47]:


df_Xbox_One_hip = df_Xbox_One_hip
df_PC_hip = df_PC_hip


alpha = 0.05

results = st.ttest_ind(df_Xbox_One_hip,df_PC_hip,equal_var = False)

print('p-value:', results.pvalue)
if results.pvalue < alpha:
    print("Kita menolak hipotesis nol")
else:
    print("Kita tidak dapat menolak hipotesis nol")


# If the p-value is greater than alpha, it means that the average user_rating for the Xbox One and PC platforms is different.

# The average user_rating for users of the Action and Sports genres is different.

# In[48]:


df_action_hip = df_new[df_new['genre'] == "Action"]
df_action_hip = df_action_hip.query('user_score != "tbd"')
df_action_hip = df_action_hip.dropna(subset=['user_score'])
df_action_hip = df_action_hip['user_score'].astype(float)


# In[49]:


df_sport_hip = df_new[df_new['genre'] == "Action"]
df_sport_hip = df_sport_hip.query('user_score != "tbd"')
df_sport_hip = df_sport_hip.dropna(subset=['user_score'])
df_sport_hip = df_sport_hip['user_score'].astype(float)


# In[50]:


df_action_hip = df_action_hip
df_sport_hip = df_sport_hip


alpha = 0.05

results = st.ttest_ind(df_action_hip,df_sport_hip,equal_var = False)

print('p-value:', results.pvalue)
if results.pvalue < alpha:
    print("Kita menolak hipotesis nol")
else:
    print("Kita tidak dapat menolak hipotesis nol")


# If the p-value is greater than alpha, it means that the average user_rating for users of the Action and Sports genres is *different*.

# # Project Summary

# ### Project Summary:
# 
# The project aims to identify patterns determining whether a game can be considered successful, allowing for the discovery of the most promising games and planning advertising campaigns. Based on the analyzed data, several criteria have the potential to achieve sales in 2017, leading to the discovery of the most promising games.
# 
# Considering data from 2013 to 2016, focusing on recent years due to the aging relevance of platforms, the top 5 platforms over the last 4 years are PS4, PS3, XOne, 3DS, and X360, with a declining trend. We select the top 3 platforms for future consideration: PS4, XOne, and 3DS.
# 
# For future planning, it is essential to create ratings for each game, prioritizing professional ratings over user ratings, as they have a more significant impact.
# 
# The top 4 selling genres are Action, Shooter, Sports, and Role-Playing, with exceptional sales compared to other genres.
# 
# Now, let's delve into specific zone areas:
# - **EU Zone:**
#   * Top Genres: Action and Shooter
#   * Top Platforms: PS4, PS3, XOne
#   * Preferred Ratings: All ESRB ratings (M, E, T, E10+)
# 
# - **JP Zone:**
#   * Top Genres: Action and Role-Playing
#   * Top Platforms: 3DS, PS3, PSV
#   * Preferred Ratings: All ESRB ratings (M, E, T)
# 
# - **NA Zone:**
#   * Top Genres: Action and Shooter
#   * Top Platforms: PS4, XOne, X360
#   * Preferred Ratings: All ESRB ratings (M, E, T, E10+)
# 
# These insights will be crucial for designing effective advertising campaigns tailored to each zone, focusing on the most profitable genres, platforms, and rating categories.
