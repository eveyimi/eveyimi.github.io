---
layout: post
title:  "Spotify Data Normalization"
date:   2020-10-09
image:  images/HW4/spotify.png
tags:   [study]
---

Hello, welcome to my blog! This post will share the data visualizations of **[Malaria Dataset][Malaria Dataset]**.

Please visit my **[GitHub][GitHub]** for more information. 

# About Malaria

> Malaria is a disease caused by parasitic single-celled organisms of the genus Plasmodium. These are transmitted between people via bites from Anopheles mosquitoes. Long-term infection leads to damage to internal organs and chronic anaemia arising from the destruction of the blood cells. Children under the age of five and pregnant women are the two demographics most at risk of severe infection.
>
> Malaria has plagued humanity throughout history and remains one of the biggest causes of death and morbidity. The World Health Organization considers nearly half the world’s population at risk, with an estimated 216 million new cases and 445,000 deaths in 2016, mostly in sub-Saharan Africa. Since 2000, a huge international effort has provided interventions such as bed nets, insecticides, and drugs across the malaria-endemic world to tackle the disease. Knowledge of the spatial distribution of malaria through time is key to targeting these resources effectively and to inform future control planning.
> 
> <cite>Malaria Atlas Project</cite>

<br>

# Data visualizations

## 1. Incidence of Malaria in different countries

### overview
Extracted the data from the `malaria_inc.csv` file as a dataframe, we can see that there are four columns, including `Entity`, `Code`, `Year`, `Incidence`. Incidence means that incidence of malaria per 1,000 population at risk. Calling `malaria_inc.info()` we can get the below informtion, telling us there are 508 rows and 4 columns, and also the non-null count and type of each column. The year data is every five year, from 2000 to 2015. It is worthwhile to see how the incidence related to countries and thus we can know which countries need help badly. I first group the `malaria_inc` dataset by `Entity`, which is the name of countries, calculate the average incidence from 2000 to 2015, and mergeit with `world` data from `GeoPandas` package, using the name of countries as foreign key. The `world` dataset has informations such as country name, code, coordinator, population, continent and soon. After merging the datasets, I can use it to draw a choropleth map, a map which depicts the spread or impact of certain phenomena across a geographical area. 

        RangeIndex: 508 entries, 0 to 507
        Data columns (total 4 columns):
         #   Column     Non-Null Count  Dtype  
        ---  ------     --------------  -----  
         0   Entity     508 non-null    object 
         1   Code       400 non-null    object 
         2   Year       508 non-null    int64  
         3   Incidence  508 non-null    float64
        dtypes: float64(1), int64(1), object(2)

<br>
### code
Data Manipulation
{% highlight python %}
malaria_inc = pd.read_csv("~/malaria_inc.csv")
malaria_inc.columns = ["Entity", "Code", "Year", "Incidence"]
average_inc = malaria_inc.groupby("Entity")["Incidence"].mean()
world_inc = world.merge(average_inc, left_on = 'name', right_on = 'Entity')
{% endhighlight %}
Data Visualization
{% highlight python %}
ax = world_inc.dropna().plot(column = 'Incidence', cmap = 'Reds', 
                              figsize = (25,15), scheme = 'quantiles', 
                              k = 10, legend = True);
ax.set_title('Average incidence of Malaria among contries from 2000 to 2015', 
              fontdict = {'fontsize':25})
ax.set_axis_off()
ax.get_legend().set_bbox_to_anchor((.12,.12))
{% endhighlight %}


### visualizations
![Incidence of countries]({{site.baseurl}}/images/HW3/inc.png)
*Average incidence of Malaria among contries from 2000 to 2015*

<br>
I splited the data into 10 ranks. The deeper the color, the more serious the Malaria in this country. As we can see from the figure above, the sub-Saharan Africa has the most urgent situation of Malaria. While some regions are ranged from 0.01 to 2.08 incidence of malaria per 1,000 population, the most serious regions can have more than 400 incidence per 1,000 population, which urges the society to pay more attention to this underrepresented areas and offer help in future control planning.

## 2. Deaths of Malaria in different countries

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

## 3. Deaths of Malaria in all continents

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



[Malaria Dataset]: https://github.com/rfordatascience/tidytuesday/tree/master/data/2018/2018-11-13
[GitHub]: https://github.com/eveyimi/eveyimi.github.io


<!-- https://medium.com/using-specialist-business-databases/creating-a-choropleth-map-using-geopandas-and-financial-data-c76419258746 -->