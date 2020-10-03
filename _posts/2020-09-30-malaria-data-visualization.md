---
layout: post
title:  "Malaria Data Visualization"
date:   2020-09-30
image:  images/HW3/malaria.jpg
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




[Malaria Dataset]: https://github.com/rfordatascience/tidytuesday/tree/master/data/2018/2018-11-13
[GitHub]: https://github.com/eveyimi/eveyimi.github.io


https://medium.com/using-specialist-business-databases/creating-a-choropleth-map-using-geopandas-and-financial-data-c76419258746