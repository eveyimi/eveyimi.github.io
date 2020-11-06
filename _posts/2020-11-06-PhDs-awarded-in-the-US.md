---
layout: post
title:  "PhDs awarded in the US"
date:   2020-11-06
image:  images/HW6/logo2.jpg
tags:   [study]
---

Hello, welcome to my blog! This post will share the data manipulation of **[PhDs awarded in the US][PhDs awarded in the US]**.

Please visit my **[GitHub][GitHub]** for more information. 

# Introduction
The Star Wars API is the world's first quantified and programmatically-formatted set of Star Wars data.

# Consume data
Using the below code to consume data using Star Wars API, we find that there are 82 people in total. 

{% highlight python %}
base_url = 'https://swapi.dev/api/people'
resp = requests.get(base_url)
data = resp.json()
{% endhighlight %}

        {'count': 82,
        'next': 'http://swapi.dev/api/people/?page=2',
        'previous': None,
        'results': [{'name': 'Luke Skywalker',
        'height': '172',
        'mass': '77',
        'hair_color': 'blond',
        'skin_color': 'fair',
        'eye_color': 'blue',
        'birth_year': '19BBY',
        'gender': 'male',
        'homeworld': 'http://swapi.dev/api/planets/1/',
        'films': ['http://swapi.dev/api/films/1/',
        'http://swapi.dev/api/films/2/',
        'http://swapi.dev/api/films/3/',
        'http://swapi.dev/api/films/6/'],
        'species': [],
        'vehicles': ['http://swapi.dev/api/vehicles/14/',
        'http://swapi.dev/api/vehicles/30/'],
        'starships': ['http://swapi.dev/api/starships/12/',
        'http://swapi.dev/api/starships/22/'],
        'created': '2014-12-09T13:50:51.644000Z',
        'edited': '2014-12-20T21:17:56.891000Z',
        'url': 'http://swapi.dev/api/people/1/'},
        {'name': 'C-3PO',
        'height': '167',
        'mass': '75',
        'hair_color': 'n/a',
        'skin_color': 'gold',
        'eye_color': 'yellow',
        'birth_year': '112BBY',
        'gender': 'n/a',
        'homeworld': 'http://swapi.dev/api/planets/1/',
        'films': ['http://swapi.dev/api/films/1/',
        'http://swapi.dev/api/films/2/',
        'http://swapi.dev/api/films/3/',
        'http://swapi.dev/api/films/4/',
        'http://swapi.dev/api/films/5/',
        'http://swapi.dev/api/films/6/'],
        'species': ['http://swapi.dev/api/species/2/'],
        'vehicles': [],
        'starships': [],
        'created': '2014-12-10T15:10:51.357000Z',
        'edited': '2014-12-20T21:17:50.309000Z',
        'url': 'http://swapi.dev/api/people/2/'},
        ... ...


<br>
Then, try to consume them one by one until we get all 82 people data and use a list to store all the JSON data.
{% highlight python %}
people = []
count = 0 # we will stop consuming until the count is 82
i = 1
while True:
    r = requests.get(os.path.join(base_url, str(i))).json()
    if r == {'detail': 'Not found'}:
        i += 1
        continue
    people.append(r)
    count += 1
    i += 1
    if count == data['count']:
        break 
{% endhighlight %}

        [{'name': 'Luke Skywalker',
        'height': '172',
        'mass': '77',
        'hair_color': 'blond',
        'skin_color': 'fair',
        'eye_color': 'blue',
        'birth_year': '19BBY',
        'gender': 'male',
        'homeworld': 'http://swapi.dev/api/planets/1/',
        'films': ['http://swapi.dev/api/films/1/',
        'http://swapi.dev/api/films/2/',
        'http://swapi.dev/api/films/3/',
        'http://swapi.dev/api/films/6/'],
        'species': [],
        'vehicles': ['http://swapi.dev/api/vehicles/14/',
        'http://swapi.dev/api/vehicles/30/'],
        'starships': ['http://swapi.dev/api/starships/12/',
        'http://swapi.dev/api/starships/22/'],
        'created': '2014-12-09T13:50:51.644000Z',
        'edited': '2014-12-20T21:17:56.891000Z',
        'url': 'http://swapi.dev/api/people/1/'},
        {'name': 'C-3PO',
        'height': '167',
        'mass': '75',
        'hair_color': 'n/a',
        'skin_color': 'gold',
        'eye_color': 'yellow',
        'birth_year': '112BBY',
        'gender': 'n/a',
        'homeworld': 'http://swapi.dev/api/planets/1/',
        'films': ['http://swapi.dev/api/films/1/',
        'http://swapi.dev/api/films/2/',
        'http://swapi.dev/api/films/3/',
        'http://swapi.dev/api/films/4/',
        'http://swapi.dev/api/films/5/',
        'http://swapi.dev/api/films/6/'],
        'species': ['http://swapi.dev/api/species/2/'],
        'vehicles': [],
        'starships': [],
        'created': '2014-12-10T15:10:51.357000Z',
        'edited': '2014-12-20T21:17:50.309000Z',
        'url': 'http://swapi.dev/api/people/2/'},
        {'name': 'R2-D2',
        'height': '96',
        'mass': '32',
        'hair_color': 'n/a',
        'skin_color': 'white, blue',
        'eye_color': 'red',
        'birth_year': '33BBY',
        'gender': 'n/a',
        'homeworld': 'http://swapi.dev/api/planets/8/',
        'films': ['http://swapi.dev/api/films/1/',
        'http://swapi.dev/api/films/2/',
        'http://swapi.dev/api/films/3/',
        'http://swapi.dev/api/films/4/',
        'http://swapi.dev/api/films/5/',
        'http://swapi.dev/api/films/6/'],
        'species': ['http://swapi.dev/api/species/2/'],
        'vehicles': [],
        'starships': [],
        'created': '2014-12-10T15:11:50.376000Z',
        'edited': '2014-12-20T21:17:50.311000Z',
        'url': 'http://swapi.dev/api/people/3/'},
        ... ...


<br>
We are required to provide the name of films each people appeared in. The raw people data only contains the URL of the films as below.
{% highlight python %}
people[0]['films']
{% endhighlight %}
        ['http://swapi.dev/api/films/1/',
        'http://swapi.dev/api/films/2/',
        'http://swapi.dev/api/films/3/',
        'http://swapi.dev/api/films/6/']


<br>
And each film API contains the below information, taking the first people as an example.
{% highlight python %}
requests.get(people[0]['films'][0]).json()
{% endhighlight %}
        {'title': 'A New Hope',
        'episode_id': 4,
        'opening_crawl': "It is a period of civil war.\r\nRebel spaceships, striking\r\nfrom a hidden base, have won\r\ntheir first victory against\r\nthe evil Galactic Empire.\r\n\r\nDuring the battle, Rebel\r\nspies managed to steal secret\r\nplans to the Empire's\r\nultimate weapon, the DEATH\r\nSTAR, an armored space\r\nstation with enough power\r\nto destroy an entire planet.\r\n\r\nPursued by the Empire's\r\nsinister agents, Princess\r\nLeia races home aboard her\r\nstarship, custodian of the\r\nstolen plans that can save her\r\npeople and restore\r\nfreedom to the galaxy....",
        'director': 'George Lucas',
        'producer': 'Gary Kurtz, Rick McCallum',
        'release_date': '1977-05-25',
        'characters': ['http://swapi.dev/api/people/1/',
        'http://swapi.dev/api/people/2/',
        'http://swapi.dev/api/people/3/',
        'http://swapi.dev/api/people/4/',
        'http://swapi.dev/api/people/5/',


<br>
We need to use the request library again to get all the names.
{% highlight python %}
for i in range(data['count']):
    people[i]['film_name'] = []
    for item in people[i]['films']:
        film = requests.get(item).json()
        people[i]['film_name'].append(film['title'])     
{% endhighlight %}


<br>
Then we transfrom the JSON data into a dataframe.
{% highlight python %}
df = pd.json_normalize(people)
{% endhighlight %}
![1]({{site.baseurl}}/images/HW5/1.png)
*Dataframe - 1*
![1]({{site.baseurl}}/images/HW5/2.png)
*Dataframe - 2*

# The oldest person
BBY means Before the Battle of Yavin. If we want to find out the oldest person, we need to find the person with the biggest number before BBY. First we need to remove the `BBY` from the birth yeas. Then we can find the index of the oldest person.
{% highlight python %}
df_birth = df[df['birth_year'].str.contains("BBY")]
idx = df_birth[(df_birth['birth_year']==max(df_birth['birth_year']))].index
{% endhighlight %}
We then use the index to find the people's name, which is `Yoda`.
{% highlight python %}
df_birth.loc[idx]['name'].to_list()[0]
{% endhighlight %}
        'Yoda'

<br>
Then we can figure out the titles of all the films she appeared in.
{% highlight python %}
df_birth.loc[idx]['film_name'].to_list()
{% endhighlight %}
        [['The Empire Strikes Back',
        'Return of the Jedi',
        'The Phantom Menace',
        'Attack of the Clones',
        'Revenge of the Sith']]


<br>

[Star Wars Dataset]: https://swapi.dev/documentation
[GitHub]: https://github.com/eveyimi/eveyimi.github.io


<!-- https://medium.com/using-specialist-business-databases/creating-a-choropleth-map-using-geopandas-and-financial-data-c76419258746 -->