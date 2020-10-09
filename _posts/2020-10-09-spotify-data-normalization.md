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
For 2NF, we need to break partial dependencies by identifing candidate PK for each row. If there is a composite PK, see if other columns have partial dependencies. First of all find if there is a composite PK. Though it seems like `track_id` is the PK, however, if we drop the duplicates of column `track_id`, we can find that there are duplicate rows it is actually not the PK. Another two columns whose name include 'id' also can not be the primary key alone for the same reason. Hense, there must exist a composite PK. I then tries to permutate `track_id` and `track_album_id` and `playlist_id` to see if there could be a composite PK. Unfortunately, I still did not find the composite PK. The `playlist_subgenre` column called my attention then, and I found that `track_id`, `track_album_id` and `playlist_subgenre` together become a composite PK. However, it is clear that other exist partial dependecies since there are roughly three fields: track info, album info and genre. We need to split the table into smaller tables. Specifically, for each song, its feature is unique, so we need to make `danceability`, `energy` and etc belong to track table. Then we shall find that genre table actually has a composite PK which includes `track_id` and `playlist_subgenre`, and `playlist_subgenre` is the PK of genre_sub table. Thus, for 2NF, I decided to split the table as followed:
- `track`: 'track_id' (single PK), 'track_name', 'track_artist', 'track_popularity', 
           'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness','acousticness',
           'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms';    
- `album`: 'track_album_id' (single PK), 'track_album_name', 'track_album_release_date';
- `genre`: ['track_id', 'playlist_subgenre'] (composite PK), 'playlist_name', 'playlist_id';
- `genre_sub`: 'playlist_subgenre' (PK), 'playlist_genre';
<br>
And there also should be a table that connects `track`, `album` and `genre`:
- `composite`: 'track_id', 'track_album_id', 'playlist_subgenre';


## 3. Third Normal Form (3NF)
For 2NF, we need to remove transitive dependencies. Based on 2NF results, we should find that there are transitive dependencies in table genre, where `playlist_name` depends on `playlist_id`. I then split them to follow the 3NF:
- `track`: 'track_id' (single PK), 'track_name', 'track_artist', 'track_popularity', 
           'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness','acousticness',
           'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms';    
- `album`: 'track_album_id' (single PK), 'track_album_name', 'track_album_release_date';
- `genre`: ['track_id', 'playlist_subgenre'] (composite PK), 'playlist_id';
- `playlist`: 'playlist_id' (PK), 'playlist_name';
- `genre_sub`: 'playlist_subgenre' (PK), 'playlist_genre';
<br>
And there also should be a table that connects `track`, `album` and `genre`:
- `composite`: 'track_id', 'track_album_id', 'playlist_subgenre';


## Populate tables
### track
{% highlight python %}
df_track = df.iloc[:, [0,1,2,3,11,12,13,14,15,16,17,18,19,20,21,22]].drop_duplicates()
df_track
{% endhighlight %}
![1]({{site.baseurl}}/images/HW4/1.png)
*Track*

### album
{% highlight python %}
df_album = df.iloc[:, 4:8].drop_duplicates()
df_album
{% endhighlight %}
![2]({{site.baseurl}}/images/HW4/2.png)
*Album*

### genre
{% highlight python %}
df_genre = df.iloc[:, [0,10,8]].drop_duplicates()
df_genre
{% endhighlight %}
![3]({{site.baseurl}}/images/HW4/3.png)
*Genre*

### playlist
{% highlight python %}
df_playlist = df.iloc[:, [8,7]].drop_duplicates()
df_playlist
{% endhighlight %}
![4]({{site.baseurl}}/images/HW4/4.png)
*Playlist*

### genre_sub
{% highlight python %}
df_genre_sub = df.iloc[:, [10,9]].drop_duplicates()
df_genre_sub
{% endhighlight %}
![5]({{site.baseurl}}/images/HW4/5.png)
*Genre Sub*

### composite
{% highlight python %}
df_composite = df.iloc[:, [0,4,10]].drop_duplicates()
df_composite
{% endhighlight %}
![6]({{site.baseurl}}/images/HW4/6.png)
*Composite*


# Dataset Overview



[Spotify Dataset]: https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-01-21
[GitHub]: https://github.com/eveyimi/eveyimi.github.io


<!-- https://medium.com/using-specialist-business-databases/creating-a-choropleth-map-using-geopandas-and-financial-data-c76419258746 -->