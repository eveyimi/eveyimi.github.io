---
layout: post
title:  "Deep Learning"
date:   2020-11-09
image:  images/HW6/logo2.jpg
tags:   [study]
---

Hello, welcome to my blog! This post will share the data manipulation of **[PhDs awarded in the US][PhDs awarded in the US]**.

Please visit my **[GitHub][GitHub]** for more information. 

# Introduction

From website Science & Engineering Doctorates we can find the Doctorate Recipients data from U.S. Universities before 2017. These tables present detailed data on the demographic characteristics, educational history, sources of financial support, and postgraduation plans of doctorate recipients. Explore the Survey of Earned Doctorates data further via NCSES's interactive data tool. By anlyzing those datasets we can get some interesting findings.

# Preparations
We should first import the necessary packages. You should register first in chart studio to get your username and api_key.
{% highlight python %}
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import matplotlib.pyplot as plt
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import chart_studio.plotly as py
import plotly.figure_factory as ff
import plotly.graph_objects as go
import chart_studio
from plotly.subplots import make_subplots

pd.options.plotting.backend = 'plotly'
chart_studio.tools.set_credentials_file(username='', api_key='')
{% endhighlight %}

# Visualizations

## Table 2
This table describes the Doctorate-granting institutions and doctorate recipients per institution: 1973–2017.
{% highlight python %}
recipient = pd.read_excel("Doctorate recipients from U.S. colleges and universities 1958–2017.xlsx",skiprows=[0,1,2])
recipient.head(10)
{% endhighlight %}

![1]({{site.baseurl}}/images/HW6/1.png)

### Data visualization
We first use two bar charts to describe the trends of recipient numbers and institution numbers seperately. We can see that both numbers increase a lot through 1973 to 2017. 
{% highlight python %}
fig = make_subplots(rows=1, cols=2)

fig.add_trace(go.Bar(x=institution['Year'],
                y=institution['Total'],
                name='Total PhD recipients',
                marker_color='rgb(55, 83, 109)'                     
                ),row=1, col=1)
fig.add_trace(go.Bar(x=institution['Year'],
                y=institution['Doctorate-granting institutions'],
                name='Doctorate-granting institutions',
                marker_color='rgb(26, 118, 255)'
                ), row=1, col=2)

fig.update_layout(height=400, width=1000, title_text="Doctorate-granting institutions")
py.plot(fig, filename='instituion_bar', auto_open=True)
fig.show()
py.plot(fig, filename='instituion_bar', auto_open=True)
# 'https://plotly.com/~Yi_/17/'
{% endhighlight %}
<iframe width="900" height="500" frameborder="0" scrolling="no" src="//plotly.com/~Yi_/17/"></iframe>

We can also make a 3D visualization to explain the relationship between Year, Number of PhD recipients and Number of institutions.
{% highlight python %}
fig = px.scatter_3d(institution, x="Year", y="Doctorate-granting institutions", 
                    z="Total", size="Mean (per institution)",
                    title = "Doctorate-granting institutions and doctorate recipients per institution: 1973–2017")
fig.show()
py.plot(fig, filename='instituion', auto_open=True)
# 'https://plotly.com/~Yi_/7/'
{% endhighlight %}
<iframe width="750" height="500" frameborder="0" scrolling="no" src="//plotly.com/~Yi_/7/"></iframe>


## Table 12
This table describes the Doctorate recipients, by major field of study: Selected years, 1987–2017.
{% highlight python %}
major = pd.read_excel("Doctorate recipients, by major field of study 1987–2017.xlsx",skiprows=[0,1,2,3])
major = major.iloc[:,[0,1,3,5,7,9,11,13]]
major.columns = ["Field of study" ,"1987","1992","1997","2002","2007","2012","2017"]
major_list = ["Life sciences","Physical sciences and earth sciences",
              "Mathematics and computer sciences", "Psychology and social sciences",
              "Engineering","Education","Humanities and arts","Othera"]
major["Major"] = 'NA'
test = major
for i in range(test.shape[0]):
    if test.iloc[i,0] in major_list:
        print(test.iloc[i,0])
        j = i+1
        while j<major.shape[0] and test.iloc[j,0] not in major_list:
            test.iloc[j,8] = test.iloc[i,0]
            j+=1
df = test[~test['Major'].str.contains("NA")]
df1 = pd.melt(df, id_vars=['Major', 'Field of study'], var_name='Year', value_name='Num')
df1.head(10)
{% endhighlight %}

![2]({{site.baseurl}}/images/HW6/2.png)

### Data visualization
We visualization the data by grouping majors and showing with the same color of sub-majors. From the line plot we can see that the Biological and biomedical science sub-major in Life sciences	major has the most number of doctorate recipients and increases the most. And the sub-major who ranks second is Psychology in major Psychology and social sciences.
{% highlight python %}
fig = px.line(df1, x="Year", y="Num", color="Major",
              line_group="Field of study", hover_name="Field of study",
              title='Doctorate recipients, by major field of study: 1987–2017')
fig.show()
py.plot(fig, filename='major_field_of_study', auto_open=True)
# 'https://plotly.com/~Yi_/9/'
{% endhighlight %}
<iframe width="900" height="500" frameborder="0" scrolling="no" src="//plotly.com/~Yi_/9/"></iframe>


## Table 34
This table describes the Highest educational attainment of either parent of doctorate recipients: Selected years, 1987–2017.
{% highlight python %}
highest_edu = pd.read_excel("Highest educational attainment of either parent of doctorate recipients.xlsx", skiprows=[0,1,2])
highest_edu = pd.melt(highest_edu, id_vars=['Year'], var_name='Education', value_name='Num')
highest_edu.head(10)
{% endhighlight %}

![3]({{site.baseurl}}/images/HW6/3.png)

### Data visualization
I drew a sctter plot to visulize the change of highest educational attainment of either parent of doctorate recipients with year changing. The size is based on number and color is based on groups. We can see that the top number of highest education transfer from High school or less to advanced degree, which is a result of the increase of education level of the whole society. Meanwhile, Bachelor's degree and Some college are roughly the same or have small fluctuation.
{% highlight python %}
fig = px.scatter(highest_edu, x="Year", y="Num", size="Num", color="Education",
           hover_name="Education", log_x=False, size_max=30)
fig.show()
py.plot(fig, filename='education_multiple', auto_open=True)
# 'https://plotly.com/~Yi_/20/'
{% endhighlight %}
<iframe width="900" height="500" frameborder="0" scrolling="no" src="//plotly.com/~Yi_/20/"></iframe>

I also created a dashboard as followed which will display the data table and also a line plot of the change of highest educational attainment with years changing. We can select different Highest educational attainment of either parent of doctorate recipients.
{% highlight python %}
import pandas as pd
import matplotlib.pyplot as plt #for plotting
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from jupyter_dash import JupyterDash
from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = JupyterDash(__name__, external_stylesheets=external_stylesheets)

highest_edu_0 = pd.read_excel("Highest educational attainment of either parent of doctorate recipients.xlsx", skiprows=[0,1,2])
highest_edu = pd.melt(highest_edu_0, id_vars=['Year'], var_name='Education', value_name='Num')
edu_list = highest_edu['Education'].unique()

def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

app.layout = html.Div([
    html.Div([
        html.H4('Highest Education options'),
        dcc.Dropdown(
            id='education',
            options=[{'label': i, 'value': i} for i in edu_list],
            value='Advanced degree'
        ),    
    ],style={'width': '30%', 'display': 'inline-block'}),

    html.Div([
        html.Div([
        html.H4('Highest Education dataframe'),
        generate_table(highest_edu_0)
        ], style={ 'width': '40%', 'display': 'inline-block','padding': '0 20'}),

        html.Div([
            dcc.Graph(id='edu_graph'),
        ], style={'width': '55%', 'display': 'inline-block','vertical-align': 'top' }),
    ], className="row"),
])

@app.callback(
    Output('edu_graph', 'figure'),
    [Input('education', 'value')])

def update_graph(education):

    highest_edu2 = highest_edu[highest_edu['Education'] == education]
    
    fig = px.line(highest_edu2, x="Year", y="Num")
    return fig

if __name__ == '__main__':
    app.run_server(mode='inline',debug=True)

{% endhighlight %}

First select an educational attainment.
![10]({{site.baseurl}}/images/HW6/10.png)

Then have interactive operations.
![11]({{site.baseurl}}/images/HW6/11.png)

## Table 35
This table describes the Doctorate recipients' primary source of financial support, by broad field of study, sex, citizenship status, ethnicity, and race: 2017
{% highlight python %}
finanial = pd.read_excel("Financial support 2017.xlsx", skiprows=[0,1,2,3,4])
finanial = finanial.rename(columns={"Unnamed: 0": "Field of study", "Unnamed: 1": "Total", 
                                    "Unnamed: 6":"Hispanic or Latino", "Unnamed: 13":"Ethnicity not reported"})
test = finanial.iloc[[0,1,2,3,4,5,6],:]
for i in range(1, test.shape[0]):
    for j in range(1, test.shape[1]):
        test.iloc[i, j] = int((test.iloc[0, j] * test.iloc[i, j]) / 100)
test = test.drop([0])
test.head(10)
{% endhighlight %}

![4]({{site.baseurl}}/images/HW6/4.png)

I then divided it into three seperate datasets to see the relationship of sex, race, citizen with doctorate recipients' primary source of financial support seperarely. <br>
The first is between primary source of financial support and gender.
{% highlight python %}
financial_major_sex = test.iloc[:,[0,2,3]]
financial_major_sex_1 = pd.melt(financial_major_sex, id_vars=['Field of study'], 
                                var_name='Sex', value_name='Num')
financial_major_sex_1.head(10)
{% endhighlight %}

![5]({{site.baseurl}}/images/HW6/5.png)

The second is between primary source of financial support and race.
{% highlight python %}
financial_major_race= test.iloc[:,[0, 6,7,8,9,10,11,12,13]]
financial_major_race_1 = pd.melt(financial_major_race, 
                                 id_vars=['Field of study'], 
                                 var_name='Race', value_name='Num')
financial_major_race_1.head(10)
{% endhighlight %}

![6]({{site.baseurl}}/images/HW6/6.png)

The first is between primary source of financial support and citizen situation.
{% highlight python %}
financial_major_citizen= test.iloc[:,[0,4,5]]
financial_major_citizen_1 = pd.melt(financial_major_citizen, 
                                    id_vars=['Field of study'], 
                                    var_name='Citizen', value_name='Num')
financial_major_citizen_1.head(10)
{% endhighlight %}

![7]({{site.baseurl}}/images/HW6/7.png)

### Data visualization
I then draw sunburst plot to describe their relationships. From the fisrt plot we can see that male takes more financial support than female in every source, except Own resources.
{% highlight python %}
fig = px.sunburst(financial_major_sex_1, path=['Field of study','Sex'], values='Num',
                  color='Num', hover_data=['Num'],
                  color_continuous_scale='RdBu')
fig.show()
py.plot(fig, filename='financial_major_sex', auto_open=True)
# 'https://plotly.com/~Yi_/11/'
{% endhighlight %}
<iframe width="750" height="500" frameborder="0" scrolling="no" src="//plotly.com/~Yi_/11/"></iframe>
From the second plot we can see that white people takes most part of financial support in every source.
{% highlight python %}
fig = px.sunburst(financial_major_race_1, path=['Field of study','Race'], values='Num')
fig.show()
py.plot(fig, filename='financial_major_race', auto_open=True)
# 'https://plotly.com/~Yi_/13/'
{% endhighlight %}
<iframe width="750" height="500" frameborder="0" scrolling="no" src="//plotly.com/~Yi_/13/"></iframe>
From the third plot we can see that U.S. citizen or permanent resident take most part of financial support in every source.
{% highlight python %}
fig = px.sunburst(financial_major_citizen_1, path=['Field of study','Citizen'], values='Num')
fig.show()
py.plot(fig, filename='financial_major_citizen', auto_open=True)
# 'https://plotly.com/~Yi_/15/'
{% endhighlight %}
<iframe width="750" height="500" frameborder="0" scrolling="no" src="//plotly.com/~Yi_/15/"></iframe>

I also created a dashboard as followed and we can selection different doctorate recipients primary source of financial support and see three pie charts at once.
{% highlight python %}
import pandas as pd
import matplotlib.pyplot as plt #for plotting
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from jupyter_dash import JupyterDash
from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = JupyterDash(__name__, external_stylesheets=external_stylesheets)

finanial = pd.read_excel("Financial support 2017.xlsx", skiprows=[0,1,2,3,4])
finanial = finanial.rename(columns={"Unnamed: 0": "Field of study", "Unnamed: 1": "Total", 
                                    "Unnamed: 6":"Hispanic or Latino", "Unnamed: 13":"Ethnicity not reported"})
test = finanial.iloc[[0,1,2,3,4,5,6],:]
for i in range(1, test.shape[0]):
    for j in range(1, test.shape[1]):
        test.iloc[i, j] = int((test.iloc[0, j] * test.iloc[i, j]) / 100)
test = test.drop([0])

financial_major_sex = test.iloc[:,[0,2,3]]
financial_major_sex_1 = pd.melt(financial_major_sex, id_vars=['Field of study'], var_name='Sex', value_name='Num')
financial_major_race= test.iloc[:,[0, 6,7,8,9,10,11,12,13]]
financial_major_race_1 = pd.melt(financial_major_race, id_vars=['Field of study'], var_name='Race', value_name='Num')
financial_major_citizen= test.iloc[:,[0,4,5]]
financial_major_citizen_1 = pd.melt(financial_major_citizen, id_vars=['Field of study'], var_name='Citizen', value_name='Num')

field_list = list(test["Field of study"])

app.layout = html.Div([
    html.H4("Doctorate recipients primary source of financial support"),
    html.H6("by broad field of study, sex, citizenship status, ethnicity, and race: 2017"),
    
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='field',
                options=[{'label': i, 'value': i} for i in field_list],
                value='Teaching assistantships'
            ),
        ],style={'width': '50%', 'display': 'inline-block'}),
    ]),

    html.Div([
        html.Label("By gender"),
        dcc.Graph(id='g1'),
    ], style={'width': '50%','display': 'inline-block', 'padding': '0 20'}),

    html.Div([
        html.Label("By race"),
        dcc.Graph(id='g2'),
        html.Label("By citizen"),
        dcc.Graph(id='g3'),
    ], style={'display': 'inline-block', 'width': '50%','vertical-align': 'top'}),
])

@app.callback(
    Output('g1', 'figure'),
    [Input('field', 'value')])
def update_graph1(option):
    sex = financial_major_sex_1[financial_major_sex_1['Field of study'] == option]
    fig = px.pie(sex, names='Sex', values='Num')
    return fig

@app.callback(
    Output('g2', 'figure'),
    [Input('field', 'value')])
def update_graph2(option):
    race = financial_major_race_1[financial_major_race_1['Field of study'] == option]
    fig = px.pie(race, names='Race', values='Num')
    return fig

@app.callback(
    Output('g3', 'figure'),
    [Input('field', 'value')])
def update_graph3(option):
    citizen = financial_major_citizen_1[financial_major_citizen_1['Field of study'] == option]
    fig = px.pie(citizen, names='Citizen', values='Num')
    return fig

if __name__ == '__main__':
    app.run_server(mode='inline',debug=True)

{% endhighlight %}

First select a source.
![8]({{site.baseurl}}/images/HW6/8.png)

Then have interactive operations.
![9]({{site.baseurl}}/images/HW6/9.png)

[PhDs awarded in the US]: https://ncses.nsf.gov/pubs/nsf19301/data
[GitHub]: https://github.com/eveyimi/eveyimi.github.io

