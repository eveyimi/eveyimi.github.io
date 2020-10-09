---
layout: post
title:  "Spotify Data Normalization"
date:   2020-10-09
image:  images/HW4/spotify.png
tags:   [study]
---

Hello, welcome to my blog! This post will share the data normalization of **[Spotify Dataset][Spotify Dataset]**.

Please visit my **[GitHub][GitHub]** for more information. 

# Normal Forms

> Every table should not have any duplication or dependencies that are not key or domain constraints
>
> **First Normal Form (1NF):** If a relation contain composite or multi-valued attribute, it violates first normal form or a relation is in first normal form if it does not contain any composite or multi-valued attribute. A relation is in first normal form if every attribute in that relation is singled valued attribute.
> 
> **Second Normal Form (2NF):** To be in second normal form, a relation must be in first normal form and relation must not contain any partial dependency. A relation is in 2NF if it has No Partial Dependency, i.e., no non-prime attribute (attributes which are not part of any candidate key) is dependent on any proper subset of any candidate key of the table. Partial Dependency – If the proper subset of candidate key determines non-prime attribute, it is called partial dependency.
> 
> **Third Normal Form (3NF):** A relation is in third normal form, if there is no transitive dependency for non-prime attributes as well as it is in second normal form. A relation is in 3NF if at least one of the following condition holds in every non-trivial function dependency X –> Y:
>    1. X is a super key.
>    2. Y is a prime attribute (each element of Y is part of some candidate key).
> 
> <cite>GeeksforGeeks</cite>

<br>

# Dataset Overview
I use the Spotify dataset from the source above. The data comes from Spotify via the spotifyr package. Charlie Thompson, Josiah Parry, Donal Phipps, and Tom Wolff authored this package to make it easier to get either your own data or general metadata arounds songs from Spotify's API. We can use the code below to get the data and look into the info. We can see that there are 23 columns and 32833 entries.
{% highlight python %}
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
all_data = pd.read_csv("https://raw.githubusercontent.com/rfordatascience
                        /tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv")
all_data.info()
{% endhighlight %}

        RangeIndex: 32833 entries, 0 to 32832
        Data columns (total 23 columns):
        #   Column                    Non-Null Count  Dtype  
        ---  ------                    --------------  -----  
        0   track_id                  32833 non-null  object 
        1   track_name                32828 non-null  object 
        2   track_artist              32828 non-null  object 
        3   track_popularity          32833 non-null  int64  
        4   track_album_id            32833 non-null  object 
        5   track_album_name          32828 non-null  object 
        6   track_album_release_date  32833 non-null  object 
        7   playlist_name             32833 non-null  object 
        8   playlist_id               32833 non-null  object 
        9   playlist_genre            32833 non-null  object 
        10  playlist_subgenre         32833 non-null  object 
        11  danceability              32833 non-null  float64
        12  energy                    32833 non-null  float64
        13  key                       32833 non-null  int64  
        14  loudness                  32833 non-null  float64
        15  mode                      32833 non-null  int64  
        16  speechiness               32833 non-null  float64
        17  acousticness              32833 non-null  float64
        18  instrumentalness          32833 non-null  float64
        19  liveness                  32833 non-null  float64
        20  valence                   32833 non-null  float64
        21  tempo                     32833 non-null  float64
        22  duration_ms               32833 non-null  int64  
        dtypes: float64(9), int64(4), object(10)
<!-- table or not -->

<br>

# Normalization

## 1. First Normal Form (1NF)
There is no composite or multi-valued attribute, so that it follow first normal form. We don't have to split composite entries. 


## 2. Second Normal Form (2NF)
For 2NF, we need to break partial dependencies by identifing candidate PK for each row. If there is a composite PK, see if other columns have partial dependencies.

### overview
Extracted the data from the `malaria_deaths.csv` file as a dataframe, we can see that there are four columns, including `Entity`, `Code`, `Year`, `Deaths`. `Deaths` indicates deaths per 100,000 people for the corresponding country. `malaria_inc.info()` tells us there are 6156 rows and 4 columns. I then took a look at the situation of deaths in different countries to by comparing the avearge the deaths from 1990 to 2016.

        RangeIndex: 6156 entries, 0 to 6155
        Data columns (total 4 columns):
         #   Column  Non-Null Count  Dtype  
        ---  ------  --------------  -----  
         0   Entity  6156 non-null   object 
         1   Code    5292 non-null   object 
         2   Year    6156 non-null   int64  
         3   Deaths  6156 non-null   float64
        dtypes: float64(1), int64(1), object(2)

<br>
### code
Data Manipulation
{% highlight python %}
malaria_deaths = pd.read_csv("~/work/ym/hw3_malaria/malaria_deaths.csv")
malaria_deaths.columns = ["Entity", "Code", "Year", "Deaths"]
average_deaths = malaria_deaths.groupby("Entity")["Deaths"].mean()
world_deaths = world.merge(average_deaths, left_on = 'name', right_on = 'Entity')
{% endhighlight %}
Data Visualization
{% highlight python %}
ax = world_inc.dropna().plot(column = 'Deaths', cmap = 'Reds', 
                             figsize = (25,15), scheme = 'quantiles', 
                             k = 3, legend = True);
ax.set_title('Average deaths of Malaria among contries from 1990 to 2016', 
              fontdict = {'fontsize':25})
ax.set_axis_off()
ax.get_legend().set_bbox_to_anchor((.12,.12))
{% endhighlight %}


### visualizations
![Deaths of countries]({{site.baseurl}}/images/HW3/death_country.png)
*Average deaths of Malaria among contries from 1990 to 2016*

<br>
For the reason that the max value of deaths data is below 200, I thus splited the data into 3 ranks. The deeper the color, the more serious the Malaria in this country. Just as the previous figure, the Malaria is most serious in Africa area. However, the difference between those two figures is that some rigions like China and Europe seems located inside the medium ranking. I guess it might because the death data is from 1990 rather than from 2000, when the medical level is rather low compare to 21th century. It is possible that almost all countries ware suffring from the Malaria to some extent. However, with the development of technology and medical level, some countries cna better control the Malaria and therefore has fewer cases. It is also the reason that other countries should offer help to those countries who are still suffering from the Malaria.

## 3. Third Normal Form (3NF)

### overview
Based on the `malaria_deaths` dataset and `world` dataset, we can see how the Malaria distributes among different continents, using the `continent` field in `world` dataset. I first merge those two datasets, and then group by `continent` and `Year` since I am also curious how the deaths data change over years.
Next, I unstack the dataframe and save the death data into a dictionary with the `continent` as keys and `Deaths` as values. After doing that, I was able to draw a stackplot as followed. 

### code
Data Manipulation
{% highlight python %}
continent_death = world.merge(malaria_deaths, left_on = 'name', right_on = 'Entity')
x = continent_death.groupby(['continent', 'Year'])["Deaths"].mean()
x = x.unstack(level='continent')
Year = list(x.index)
death_by_continent = {}
for c in list(x.columns):
    death_by_continent[c] = list(x[c])  
{% endhighlight %}

Data Visualization
{% highlight python %}
fig, ax = plt.subplots(figsize = (20, 10))
ax.stackplot(Year, death_by_continent.values(),
             labels=death_by_continent.keys())
ax.legend(loc='lower left')
ax.set_title('Deaths of Malaria among continents through 1990 to 2016', 
              fontdict = {'fontsize':25})
ax.set_xlabel('Year', fontdict = {'fontsize':13})
ax.set_ylabel('Number of people (millions)', fontdict = {'fontsize':13})
plt.show()
{% endhighlight %}

### visualizations
![Deaths of countries]({{site.baseurl}}/images/HW3/death_continent.png)
*Deaths of Malaria among continents through 1990 to 2016*

<br>
As it is shown in the figure, there is a U-curve on deaths of Malaria from 1990 to 2016 and the peek showed in 2002~2003. After that, the deaths ratetall over the world decreased to almost half of before. We can also see that Africa takes the largest position of deaths among all continents and second largest continent is Oceania, and the third is Asia. The reason why Oceania has the second largest death rate is probabally because of the terrain.

## 4. How the age infects deaths of Malaria

### overview
Extracted the data from the `malaria_deaths_age.csv` file as a dataframe, we can see that there are six columns, including `entity`, `code`, `year`, `age_group`, `deaths`, `Unnamed: 0	`. `age_group` has 5 levels, `15-49`, `5-14`, `50-69`, `70 or older`, `Under 5`. `malaria_deaths_age.info()` tells us there are 30780 rows and 6 columns. I then took a look at the deaths rate of different age groups from 1990 to 2016, by merging to `world`, then grouping by `age_group` and `year`, and finally unstacking and drawing plots.

        RangeIndex: 30780 entries, 0 to 30779
        Data columns (total 6 columns):
         #   Column      Non-Null Count  Dtype  
        ---  ------      --------------  -----  
         0   Unnamed: 0  30780 non-null  int64  
         1   entity      30780 non-null  object 
         2   code        26460 non-null  object 
         3   year        30780 non-null  int64  
         4   age_group   30780 non-null  object 
         5   deaths      30780 non-null  float64
        dtypes: float64(1), int64(2), object(3)

<br>       
### code
Data Manipulation
{% highlight python %}
malaria_deaths_age = pd.read_csv("~/work/ym/hw3_malaria/malaria_deaths_age.csv")
world_death_age = world.merge(malaria_deaths_age, left_on = 'name', right_on = 'entity')
y = world_death_age.groupby(['age_group', 'year'])["deaths"].mean()
y = y.unstack(level='age_group')
Year_age = list(y.index)
death_by_age = {}
for c in list(y.columns):
    death_by_age[c] = list(y[c])  
{% endhighlight %}

Data Visualization
{% highlight python %}
fig, ax = plt.subplots(figsize = (20, 10))
ax.plot(y)
ax.legend(y.columns, loc='lower left')
ax.set_title('Deaths of Malaria among age groups through 1990 to 2016', 
              fontdict = {'fontsize':25})
ax.set_xlabel('Year',fontdict = {'fontsize':13})
ax.set_ylabel('Number of people (millions)', fontdict = {'fontsize':13})
ax.set_facecolor('whitesmoke')
plt.show()
{% endhighlight %}

### visualizations
![Deaths of countries]({{site.baseurl}}/images/HW3/death_age.png)
*Deaths of Malaria among age groups through 1990 to 2016*

<br>
As it is shown in the figure, The age group `Under 5` takes the largest portion of the total death cases, which matched the theory that children under the age of five and pregnant women are the two demographics most at risk of severe infection. It also even did not show a significant decreasing pattern comparing to the cases 30 years ago. Vulnerable groups are not only more susceptible to infection, but also more difficult to restore health. This requires the tilt of more social resources.



[Spotify Dataset]: https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-01-21
[GitHub]: https://github.com/eveyimi/eveyimi.github.io


<!-- https://medium.com/using-specialist-business-databases/creating-a-choropleth-map-using-geopandas-and-financial-data-c76419258746 -->