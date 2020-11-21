---
layout: post
title:  "Exploring COVID-19 related clinical trials"
date:   2020-11-17
image:  images/final/logo.jpeg
tags:   [study]
---

Hello, welcome to my blog! This post will explore COVID-19 related clinical trials. If you are interested, please visit my **[GitHub][GitHub]** for more information. 

# Introduction

### 1. Basic information
- **Project name**: Exploring COVID-19 related clinical trials in the U.S. and Beyond
- **Team members**: Oana Enache, Yi Mi, Yue Han
- **Repository link**: https://github.com/oena/bios823_final_project
- **Dashboard link**: https://share.streamlit.io/oena/bios823_final_project/dashboard/app_main.py

### 2. Product and Purpose
COVID-19 is ravaging the world. At this critical moment, clinical trials are crucial to overcome the virus. It is the basis for treatment of patients and vaccine development. We developed a dashboard to display the information related to COVID-19 related clinical trials, including data visualization of world trials and U.S. trials, clustering tirals by similarity and predicting trials' opening status.The purpose of our project is to give people a basic understanding of COVID-19 related clinical research and to increase confidence for everyone to defeat the virus. Our audience of visualization parts could be any people who are curious about COVID-19 related clinical trials even without professional knowledge since all plots are intuitive and easy to understand. But for the clustering part and predicting part, our audience needs to have some knowledge of statistics. 

### 3. Dataset
The dataset is from ClinicalTrials.gov, a database of privately and publicly funded clinical studies conducted around the world. The dataset includes ongoing and completed COVID-19 studies listed on the World Health Organization's International Clinical Trials Registry Platform (WHO ICTRP), in order to give researchers and the public rapid access to studies on COVID-19 studies. Information in the WHO ICTRP comes from clinical trial databases maintained by other countries or regions of the world. 

### 4. My contributions and skillsets
I mainly cleaned data, did exploratory data analyses, set up SQL server, built models and visualizations and set up my page on streamlit dashboard.

The skillsets I demonstrate in this project include Data Science (`pandas`, `numpy`, `pandas_profiling`, `missingno`), SQL (`sqlite3`), Machine Learning (`sklearn`, `pycaret`, `yellowbrick`, `skopt`), Visualization (`plotly`, `matplotlib`), Dashboard (`streamlit`), several classifiers and functional programming in Python.

In this post, I will elaborate on my contribution to this project and the technology I used. All code can be found in our repository.

# Data Cleaning
Based on my teammates' basic data cleaning process, I further cleaned the data to make it comply with SQLite3 schema.

### 1. Get data
The first step I did was to get the data and take a deep look of the data. Basically, my teammates did the following steps.
- selected columns of interests
- cleaned date columns into standard format with month and year
- cleaned locations columns, including country, city of state, instituions
- transferred all letters to upper cases
- replaced "nan" and "NaN" to `np.nan`

Based on their effort, I found that there are still fields cleaning steps could be optimized and thus further cleaned the data to make them efficient and easy to use.

### 2. Clean Study Design and Intervention
- **Study Design**: explains the investigative methods or strategies used in the clinical study. It also provides other Information, such as study type, estimated or actual enrollment, study start and completion dates, and official study title.
- **Intervention**: for interventional studies, this section explains the type of intervention/treatment participants receive, what the dosage is for drugs, and how long the participants receive the intervention; for observational studies, this section explains the participant groups that are observed and any treatments or exposures that are of interest in the study.

First, I found that there are two fields are JSON-like, which is defined by myself. Strictly speaking, they are not JSON data, but they looked like JSON data. In each cell of them, several pairs of key and value exist, which are mapped by colon and separated by vertical bars. I wrote two funtions which were applied to each row of `Study Design` and `Intervention` to get the pair of key and value, save the data into real JSON format and then transfer the JSON data into dataframe. Below are examples of `Study Design` and `Intervention`. We can see that there are ALLOCATION, INTERVENTION MODEL, MASKING and PRIMARY PURPOSE keys inside this `Study Design` cell. There are also DRUG and PROCEDURE keys inside this `Intervention` cell.
            
        # Study Design
        'ALLOCATION: RANDOMIZED|INTERVENTION MODEL: SINGLE GROUP ASSIGNMENT|MASKING: TRIPLE (PARTICIPANT, CARE PROVIDER, INVESTIGATOR)|PRIMARY PURPOSE: TREATMENT'
        # Intervention
        'DRUG: DUVELISIB|PROCEDURE: PERIPHERAL BLOOD DRAW

<br>

Actually, there are more keys inside `Study Design` and `Intervention`, which is a difficulty here to not miss any of them. My strategy is to first extract all possibles keys and then for each cell find out if any key exists. Eventually, I successfully expand those them and get dataframes. Below is the `Study Design` dataframe. We can find that it is super sparse.

![]({{site.baseurl}}/images/final/study-design-df.jpg)

### 3. Create a SQLite3 schema following 3NF
As introduced in previous post of spotify data normalization, third normal form (3NF) is a database schema design approach for relational databases which uses normalizing principles to reduce the duplication of data, avoid data anomalies, ensure referential integrity, and simplify data management. Therefore, to make our data easy to manipulate, I created a SQLite3 schema following 3NF.

First, I checked and confirmed that there are no duplication and the `NCT Number` is the primary key of the data. I have handled JSON-like fields (`Study Design` and `Intervention`). Those two table can be store individually in SQL with `NCT Number` as PK. However, for the following four fields, `Outcome measures`, `Sponsor collaborators`, `Funded bys`, `Study type`, there are multiple values inside one cell, separated by vertical bars. To keep columns atomic, we have to further process them.

#### Handle multi-valued attributes
- **Outcome measures**: describes the measurements that are used to determine the effects of intervention or treatment on participants. Types of outcome measures include primary outcome measures, secondary outcome measures, and other pre-specified measures. For observational studies, this section explains the participant groups that are observed and any treatments or exposures that are of interest in the study.
- **Sponsor collaborators**: the study sponsors, which are concrete institution names in this dataset
- **Funded bys**: clinical studies can be funded, by pharmaceutical companies, academic medical centers, voluntary groups, and other organizations, in addition to Federal agencies such as the National Institutes of Health, the U.S. Department of Defense, and the U.S. Department of Veterans Affairs.
- **Study type**: describes the nature of a clinical study. Study types include interventional studies (also called clinical trials), observational studies (including patient registries), and expanded access.

For example, we could have a 'OTHER\|NIH' in `Funded bys` field. 
![]({{site.baseurl}}/images/final/before.png)
*Before*

What I did was to extract those four columns with `NCT Number` separately (i.e. have four tables) and split the columns by vertical bars (i.e. long term). Then, for each of those four tables, we cannot have the `NCT Number` as the primary key anymore, since we will have duplicate PK. Under this situation, the `NCT Number` and `Funded bys` together will be a composite PK and others three tables are similar. We can easily use SQL quries to extrat data.

![]({{site.baseurl}}/images/final/after.png)
*After*

Other useful fields are store together in one table. 3NF is satisfied without partial dependency or transitive dependency. All tables are saved in SQLite3.
{% highlight sql %}
import sqlite3
conn = sqlite3.connect('covid_trials.db')
trial_info.to_sql('trial_info', conn, if_exists='replace', index=False)
study_designs.to_sql('study_designs', conn, if_exists='replace', index=False)
interventions.to_sql('interventions', conn, if_exists='replace', index=False)
outcome_measures.to_sql('outcome_measures', conn, if_exists='replace', index=False)
sponsor_collaborators.to_sql('sponsor_collaborators', conn, if_exists='replace', index=False)
funded_bys.to_sql('funded_bys', conn, if_exists='replace', index=False)
study_type.to_sql('study_type', conn, if_exists='replace', index=False)
{% endhighlight %}

# Openning Status classification
In the beginning, I planned to predict if the clinical trial will succeed or not based on variable `Status`. It would be really meaningful to have an idea of this most important sense. But when I looked into the data, I found that the data is highly imbalanced--it indicates that only one trial is successful, whose status is 'APPROVED FOR MARKETING'.

- **Status**:

        ACTIVE, NOT RECRUITING      262
        APPROVED FOR MARKETING        1
        AVAILABLE                    22
        COMPLETED                   460
        ENROLLING BY INVITATION     125
        NO LONGER AVAILABLE           4
        NOT YET RECRUITING          870
        RECRUITING                 1992
        SUSPENDED                    23
        TERMINATED                   30
        WITHDRAWN                    60

<br>

## 1. Data preprocessing

#### Further cleaning
I listed further data clearning steps as followed:
1. **Age**: extracted the age phases in paranthesis and replaced the missing data as "OTHERS"; made the type as "category"
2. **Gender**: replaced the missing data as "All"; made the type as "category"
3. **Phases**: the original phases data may include two phases in one cell, I extracted the highest phase of each cell; made the type as "category"
4. **Study Type**: it describes the nature of a clinical study, including interventional studies (also called clinical trials), observational studies (including patient registries), and expanded access. Some "expanded access" are followed by redundent details, and thus I removed the details and made the type as "category"
5. **Results**: it indicates if the trail has result or not; made the type as "category"
6. **Funded bys**: seperated the founded bys into four columns ('INDUDTRY', 'NIH', 'OTHER', 'U.S. FED') and each is a binary variable, indicating if the trial is funded by each of them or not
7. **Locations**: replaced the missing data as "OTHER"; made the type as "category"
8. **Duration**: calculated the duration of the trials by months
9. **Status**: classified the status into active and not active two categories; made the type as "category"

#### Imputation and Handling outliers
There are some missing data in quantative variables **Enrollment** and **Duration**. I first fixed the missing data by imputing the mean and then removed the outliers which exceeded three times of the standard deviation.

#### Handling imbalance data
After finishing all the steps above, I applied `get_dummies` on X and splited the data into train and test datasets. Next I handled imbalanced data. Even if I decided to classify the active status rather than predicting the success and failure, the data is still imbalanced. After comparing the performance of over-sample, de-sample and the combination of them, I found that over-sample outputs the best performance, which was then used to handle our imbalanced data.

## 2. Modelling
I first tried some traditional classifiers, such as Random Forest classifier and Logistic Regression classifier. When I tried to tune hyperparameters, I carefully read through the **[notebook][notebook]** of BIOSTAT 823 and found a really fancy tool, which is `pycaret`. `pycaret` provides nice and easy to use APIs for modelling.

#### Comparing and tuning models
I used pycaret to compare models and tuned models. After setting up, we can call `compare_models` and provide the metric we want to sort based on. I Here I sort the models by Accuracy.

{% highlight python %}
best_model = compare_models(sort = 'Accuracy')
{% endhighlight %}
![]({{site.baseurl}}/images/final/model1.png)
*Comparing models*

We can see that Extreme Gradient Boosting classifier has the best performance on Accuracy, AUC and Kappa. We can create and tune the xgboost by the code below.

{% highlight python %}
clf = create_model('xgboost')
tuned_clf = tune_model(clf)
{% endhighlight %}
![]({{site.baseurl}}/images/final/model2.png)
*Creating and tuning xgboost model*

I got the idea here to display model comparison on our dashboard.

#### Performance Plots
In order to get more intuitive of model performance, I draw three different plots, ROC Curve, Precision-Recall Curve and Confusion Matrix. You can see the plots in next section.

An ROC curve (receiver operating characteristic curve) is a graph showing the performance of a classification model at all classification thresholds. This curve plots two parameters: True Positive Rate. False Positive Rate. An ROC curve plots TPR vs. FPR at different classification thresholds. Lowering the classification threshold classifies more items as positive, thus increasing both False Positives and True Positives. 
<cite>Google Machine Learning Crash Course.</cite>

The Precision-Recall curve shows the tradeoff between precision and recall for different threshold. A high area under the curve represents both high recall and high precision, where high precision relates to a low false positive rate, and high recall relates to a low false negative rate. High scores for both show that the classifier is returning accurate results (high precision), as well as returning a majority of all positive results (high recall).
<cite>scikit-learn.</cite>

Confusion Matrix is an NxN table that summarizes how successful a classification model's predictions were; that is, the correlation between the label and the model's classification. One axis of a confusion matrix is the label that the model predicted, and the other axis is the actual label. N represents the number of classes. In a binary classification problem, N=2. <cite>Google Machine Learning Crash Course.</cite>

## 3. Insights
Below is a table of top important features I got from Extreme Gradient Boosting classifier. We can see that the most influential feature is observational study type. And if the trial has no result yet, it will be highly possible that it is still openning and active. 
![]({{site.baseurl}}/images/final/feature.png)
*Top important features*
It might not have much practical use to predict the openning status of clinical trials, which only matters to the clinical volunteers. However, I believe that after we have more data, we can predict whether one trail will succeed or not. Meanwhile, I also went through the Machine Learning process completely in this project, which improved my ability to process data and build models, allowing me to apply more useful and interesting tools.

# Building dashboard
Each of us built our own streamlit dashboard page. I found it is really a code-light tool to build dashboard, compared to Dash. Below is the overview of my page. All plots are drawn with plotly and support interaction. You can have a try on the dashboard we deployed.
![]({{site.baseurl}}/images/final/d1.png)
*Predicting trials' activity status page*

In my Predicting trials' activity status page, I have the first visualization part, which displays the comparsion amoung all classifiers. We allow users to select one metric to see the performance difference between models. The metrics include Accuracy, AUC, Recall, Precision, F1, Kappa, MCC and Time. Compared to the previous image, we can see that though Extreme Gradient Boosting classifier has best Accuracy, its Recall is not the highest.
![]({{site.baseurl}}/images/final/d2.png)
*Select a metric*

In the second part, I display the Performance Plots mentioned before, including ROC Curve, Precision-Recall Curve and Confusion Matrix. 
![]({{site.baseurl}}/images/final/d3.png)
*Plots*

We also allow users to select a classifier and display the corresponding plots. For example, here we select Extreme Gradient Boosting classifier which has the best performance in Accuracy. Compared to default Random Forrest classifier, it does have a better performance.
![]({{site.baseurl}}/images/final/d4.png)
*Select a classifier*

# Reflection and Conclusion

enviroment imcompatible
github usage







[GitHub]: https://github.com/eveyimi/eveyimi.github.io
[notebook]: http://people.duke.edu/~ccc14/bios-823-2020/index.html
