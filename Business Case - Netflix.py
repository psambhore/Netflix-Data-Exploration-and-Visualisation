#!/usr/bin/env python
# coding: utf-8

# ## Netflix Case Study - 

# ## 1. Importing Libraries and Basic Observations

# In[91]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("Netflix.csv")


# In[3]:


data


# In[4]:


data.shape


# In[5]:


# Data type - 

data.info()


# #### The size of the given dataset is Rows = 8807 & Columns = 12

# In[ ]:





# In[6]:


data.nunique()


# #### In the data it seen that show_id column and title column has unique values. Hence it can be concluded that  - Total 8807 movies/TV shows data is available in the dataset.

# In[7]:


data.describe()


# #### It seems that only single column has numerical values, and it shows release year of the content ranges between what timeframe. It means rest all the columns are having the categorical data.   

# In[ ]:





# ## 2. Data Cleaning 

# #### Overall null values in each column of dataset

# In[8]:


data.isna().sum()


# In[9]:


data.isna().sum().sum()


# In[10]:


data[data["duration"].isna()]


# #### It seems that there are 3 missing values in duration column, that values entered in rating column by mistake.

# In[11]:


# Replacing the wrong entries in rating column


a = data[data["duration"].isna()].index
a


# In[12]:


data.loc[a] = data.loc[a].fillna(method = "ffill", axis = 1)


# In[13]:


# Replacing wrong entries in rating column with "Not Available"

data.loc[a, "rating"] = "Not Available"


# In[14]:


data.loc[a]


# In[15]:


# Null values in rating column fill with "Not Available"

data[data["rating"].isna()]


# In[16]:


b = data[data["rating"].isna()].index
b


# In[17]:


data.loc[b, "rating"] = "Not Available"


# In[18]:


data.loc[b]


# In[19]:


data.rating.unique()


# In[ ]:





# In[20]:


data[data["date_added"].isna()]


# In[21]:


data.drop(data.loc[data["date_added"].isna()].index, axis = 0, inplace = True)
data["date_added"].value_counts()


# In[22]:


# changing data type from object to datetime

data["date_added"] = pd.to_datetime(data["date_added"])
data["date_added"]


# #### Dropped the null values from date_added column, and converted data type of date_added column from "object" to "datetime"

# In[ ]:





# #### Adding the new columns "year_added" & "month_added" by extracting the year & month from 'date_added' column.

# In[23]:


# Add year_added column

data["year_added"] = data["date_added"].dt.year

# Add month_added column

data["month_added"] = data["date_added"].dt.month


# In[24]:


data.head()


# In[25]:


data[["date_added", "year_added", "month_added"]].info()


# In[26]:


data.isna().sum()


# ## 3.Non-Graphical Analysis

# #### Types of content  in dataset.

# In[27]:


data["type"].unique()


# In[28]:


movies = data.loc[data["type"] == "Movie"]
movies.duration.value_counts()


# In[29]:


tv_shows = data.loc[data["type"] == "TV Show"]
tv_shows.duration.value_counts()


# #### Movie and TV shows both have different format for duration, we can change duration for movies as minutes & TV shows as seasons

# In[30]:


movies["duration"] = movies["duration"].str[:-3]
movies["duration"] = movies["duration"].astype("float")


# In[31]:


tv_shows["duration"] = tv_shows["duration"].str[:-7].apply(lambda x : x.strip())
tv_shows["duration"] = tv_shows["duration"].astype("float")


# In[32]:


tv_shows.rename({'duration': 'duration_in_seasons'} ,axis = 1 , inplace = True)
movies.rename({'duration': 'duration_in_minutes'} ,axis = 1 , inplace = True)


# In[33]:


tv_shows.head()


# In[34]:


movies.head()


# In[35]:


movies.duration_in_minutes


# In[36]:


tv_shows.duration_in_seasons


# #### The first movie added on Netflix and most recent movie added on Netflix.

# In[37]:


first_movie = pd.Series((data["date_added"].min().strftime("%B %Y")))
first_movie


# In[38]:


recent_movie = pd.Series((data["date_added"].max().strftime("%B %Y")))
recent_movie


# #### In which year the oldest and the most recent movie/TV show relased on the Netflix.

# In[39]:


oldest = data["release_year"].min()
oldest


# In[40]:


recent = data["release_year"].max()
recent


#  #### Different types of ratings available on Netflix  and the number of content released in each type.

# In[41]:


data.groupby(["type", "rating"])["show_id"].count()


#  ####   Country Column

# In[42]:


data["country"].value_counts()


# It seems that many movies are produced in more than 1 country. Hence, the country column has comma separated values of countries.It's difficult to analyse how many movies were produced in each country. We can split the country column into different rows.

# In[43]:


# drop the null values - 

ctry = data[["show_id", "type", "country"]]
ctry.dropna(inplace = True)
ctry


# In[44]:


# split the countries by comma - 

ctry["country"] = ctry["country"].apply(lambda x : x.split(","))
ctry = ctry.explode("country")
ctry


# In[45]:


# Remove the empty strings values
   
ctry["conuntry"] = ctry["country"].str.strip()


# In[46]:


ctry.loc[ctry["country"] == ""]


# In[47]:


ctry = ctry.loc[ctry["country"] != ""]
ctry["country"].nunique()


# #### There are movies from 196 conutries.

# #### Total movies and tv shows in each country-

# In[48]:


total = ctry.groupby(["country", "type"])["show_id"].count().reset_index()
total.pivot(index = ["country"], columns = "type", values = "show_id").sort_values("Movie", ascending = False)


# ### Director Column 

# In[49]:


data["director"].value_counts()


# In[50]:


drt = data[["show_id", "type", "director"]]
drt.dropna(inplace = True)
drt


# In[51]:


drt["director"].value_counts()


# In[52]:


drt["director"].nunique()


# #### Total 4528 directors in the dataset.

# 
# #### Total movies and tv shows directed by each director-

# In[53]:


total = drt.groupby(["director", "type"])["show_id"].count().reset_index()
total.pivot(index = ["director"], columns = "type", values = "show_id").sort_values("Movie", ascending = False)


# #### We can get details about genres from 'listed_in' column.
# 

# In[54]:


genre = data[["show_id", "type", "listed_in"]]
genre["listed_in"] = genre["listed_in"].apply(lambda x : x.split(","))
genre = genre.explode("listed_in")
genre


# In[55]:


genre["listed_in"].unique()


# In[56]:


genre["listed_in"].nunique()


# #### There are total 73 genres in the dataset.

# #### Total movies and TV shows in each genre - 

# In[57]:


total = genre.groupby(["listed_in", "type"])["show_id"].count().reset_index()
total.pivot(index = "listed_in", columns = "type", values = "show_id").sort_index()


# #### Cast Column

# In[58]:


cast = data[["show_id", "type", "cast"]]
cast.dropna(inplace = True)
cast


# In[59]:


cast["cast"] = cast["cast"].apply(lambda x : x.split(","))
cast = cast.explode("cast")
cast


# In[60]:


cast["cast"].nunique()


# #### There are total 39260 actors.

# #### Total movies and TV shows by each actor - 

# In[61]:


total = cast.groupby(["cast","type"])["show_id"].count().reset_index()
total.pivot(index = "cast", columns = "type", values = "show_id").sort_values("Movie", ascending = False)


# In[ ]:





# ## 4. Visual Analysis - Univariate & Bivariate

# #### 1. Distribution of content across differner types

# In[62]:


types = data.type.value_counts()
plt.pie(types, labels = types.index, autopct = "%1.1f%%", colors = ["green", "orange"])
plt.title("Total Movies & TV Shows")
plt.show()


# It seems that in pie chart around 70% content is Movies and around 30% content is TV shows.

# #### 2. Total Movies/TV Shows by each Director.

# In[63]:


top = drt["director"].value_counts().head(10).index
data_new = drt.loc[drt["director"].isin(top)]

plt.figure(figsize = (8, 4))

sns.countplot(data = data_new, y ="director", order = top, orient = "v")

plt.xlabel("Total Movies/TV Shows", fontsize = 10)
plt.ylabel("Directors", fontsize = 10)

plt.xticks(fontsize = 8)
plt.yticks( fontsize = 8)

plt.title("Total Movies/TV Shows by Director", fontsize = 12)
plt.show()


# #### 3. The number of Movies/TV shows added on Netflix per Year.

# In[64]:


total = data.groupby(["year_added", "type"])["show_id"].count().reset_index()

total.rename({"show_id" : "Count of Movies/TV Shows"}, axis = 1, inplace = True)

plt.figure(figsize = (8, 4))

sns.lineplot(data = total, x = "year_added", y = "Count of Movies/TV Shows", hue = "type", marker = "o")

plt.xlabel("year_added", fontsize = 10)
plt.ylabel(" Count of Movies & TV Shows year_added", fontsize = 10)

plt.title("Total Movies &  TV Shows per Year", fontsize = 12)
plt.show()


# Conclusion - 
# 
# 1. After 2015 content added on Netflix surged drastically.
# 2. In the 2020 - 2021 seen that there is drop in the content added.
# 3. Highest Movies and TV shows added on Netlix in 2019.
# 4. As compare to Movies, TV shows not dropped drastically.

# #### 4.Total Movies/TV Shows per Country

# In[65]:


top = ctry.country.value_counts().head(10).index

data_new = ctry.loc[ctry["country"].isin(top)]

x = data_new.groupby(["country", "type"])["show_id"].count().reset_index()
x.pivot( index = "country", columns = "type", values = "show_id").sort_values("Movie", ascending = False)


# In[66]:


plt.figure(figsize = (8,4))

sns.countplot(data = data_new, x = "country", order = top, hue = "type")

plt.xticks(rotation = 45)
plt.ylabel("Total Movies/TV shows")
plt.title("Total Movies/TV Shows per Country")

plt.show()


# #### 5.Total Movies/TV Shows in each Genre.

# In[67]:


top_movies = genre[genre["type"] == "Movie"].listed_in.value_counts().head(10).index
data_movie = genre.loc[genre["listed_in"].isin(top_movies)]
data_movie


# In[68]:


top_tv_shows = genre[genre["type"] == "TV Show"].listed_in.value_counts().head(10).index
data_tv = genre.loc[genre["listed_in"].isin(top_tv_shows)]
data_tv


# In[69]:


plt.figure(figsize = (8,4))
sns.countplot(data = data_movie, x = "listed_in", order = top_movies)
plt.xticks(rotation = 90, fontsize = 8)
plt.yticks(fontsize = 8)
plt.xlabel("Genres", fontsize = 10)
plt.ylabel("Total Movies", fontsize = 10)
plt.title("Total Movies by Genre")
plt.show()


# In[70]:


plt.figure(figsize = (8,4))
sns.countplot(data = data_tv, x = "listed_in", order = top_tv_shows)
plt.xticks(rotation = 90, fontsize = 8)
plt.yticks(fontsize = 8)
plt.xlabel("Genres", fontsize = 10)
plt.ylabel("Total TV Shows", fontsize = 10)
plt.title("Total TV Shows by Genre")
plt.show()


# ## 5.Bivariate Analysis 

# #### 1.Variation in duration of movies by Release year

# In[71]:


plt.figure(figsize = (10,6))
plt.scatter(movies["duration_in_minutes"], movies["release_year"], alpha = 0.5)

plt.xlabel("duration_in_minutes", fontsize = 12)
plt.ylabel("release_year",fontsize = 12)

plt.title("Variation in duration of movies by Release year")

plt.xlim(0,200)
plt.show()


# Conclusion - 
# 1. Movies shorter than 150 minutes duration have increased drastically after 2000 & that are not much popular.
# 2. Short movies have been popular in last 10 years.

# #### 2. Time when maximum content added on the Netflix.

# In[72]:


month_year = data.groupby(["year_added", "month_added"])["show_id"].count().reset_index()


# In[73]:


plt.figure(figsize = (8, 5))

sns.lineplot(data = month_year, x = "year_added", y = "show_id", hue = "month_added")

plt.title("Max shows added on Netflix")
plt.show()


# Conclusion - 
# 
# 1. Shows getting added on Netflix is increasing with each year until 2020.
# 2. Oct-Dec have more shows being added than the other months of the year.

# ####  3. The countries  which has added more number of content over the time.

# In[74]:


country_list = ctry.country.value_counts().head(10).index


# In[75]:


top_10_countries = ctry.loc[ctry["country"].isin(country_list)]
country_year = top_10_countries.merge(data, on = "show_id")[["show_id", "country_x", "type_x", "year_added"]]
country_year.columns = ["show_id", "country", "type", "year_added"]
country_year = country_year.groupby(["country", "year_added"])["show_id"].count().reset_index()


# In[76]:


plt.figure(figsize =(8,4))

sns.lineplot(data = country_year, x = "year_added", y = "show_id", hue = "country", palette = "rainbow")


plt.show()


# Conclusion -
# 1. United Stated have added highset number of movies/TV shows over the time. 
# 2. Since 2016, India has seen spike in popularity of content and added more number of content, followed by United Kingdom at 3rd    position.

# #### 4.Popular genres in top 20 countries
# 

# In[84]:


top_20_country = ctry.country.value_counts().head(20).index
top_20_country = ctry.loc[ctry["country"].isin(top_20_country)]


# In[82]:


x = top_20_country.merge(genre, on = "show_id").drop_duplicates()
country_genre = x.groupby(["country", "listed_in"])["show_id"].count().sort_values(ascending = False).reset_index()
country_genre = country_genre.pivot(index = "listed_in", columns = "country", values = "show_id")


# In[86]:


plt.figure(figsize = (12,10))
sns.heatmap(data = country_genre, annot = True, fmt = ".0f", vmin = 20, vmax = 250)
plt.xlabel('Countries' , fontsize = 10)
plt.ylabel('Genres' , fontsize = 12)
plt.title('Countries V/s Genres' , fontsize = 10)


#  ## 6.Insights based on Non-Graphical and Visual Analysis

# 1. On Netflix around 70 % content is of Movies & and 30 % is of TV Shows.
# 2. Content on Netflix from 122 countries are present, in which United States is the highest contributor with almost 37 % of all content.
# 3. International Movies and TV Shows , Dramas , and Comedies are the top 3 genres on Netflix for both Movies and TV shows.
# 4. Only United States have a good mix of almost all genres.
# 5. Indian Actors acted in maximum movies on netflix. Top 5 actors are in India based on quantity of movies.
# 6. Shorter duration movies have been popular in last 10 years.
# 7. Content uploading on the Netflix started form the year 2008, and it has very less content till 2014.
# 8. Drastic surge in the content uploaded on Netflix marks in 2015.
# 9. Year 2020 and 2021 has seen the drop in content added on Netflix, possibly because of Pandemic.
# 10. From 2018 drop in the movies content is seen, but rise in TV shows is observed. It shows the rise in popularity of TV shows in the recent years.
# 11. On Netflix around 4528 directors have their movies or tv shows on Netflix.
# 12. In the range 2005-2021 max shows.
# 13. 1-3 seasons is the range for TV shows seasons, excluding potential outliers.
# 1

# ## 7. Business Insights

# 1. Netflix is currently serving mostly Mature audiences or Children with parental guidance. It have scope to cater other audiences as well such as familymen , Senior citizen , kids of various age etc.
# 
# 2. The country like India , which is highly populous , has maximum content available only in three rating TV-MA, TV-14 , TV-PG. It is unlikely to serve below 14 age and above 35 year age group .
# 
# 3. Netflix ha need to add demographic content of any country. Netflix can produce higher number of content in the perticular rating as per demographic of the country. 
# 
# 4. Like in Indian Mythological content is highly popular. We can create such more country specific genres and It might also be liked acorss the world just like Japanese Anime.
# 
# 5. Japan have only 3 rating of content largely served - TV-MA, TV-14 , TV-PG.Japan have high population of age above 60, and this can be served by increasing the content suitable for this age group.
# 
# 6. Very limited genres are focussed in most of the countries except US. It seems the current available genres suits best for US and few countries but maximum countries need some more genres which are highly popular in the region.
