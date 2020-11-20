---
layout: post
title:  "Exploring COVID-19 related clinical trials"
date:   2020-11-17
image:  images/final/logo.jpeg
tags:   [study]
---

Hello, welcome to my blog! This post will explore COVID-19 related clinical trials. If you are interested, please visit my **[GitHub][GitHub]** for more information. 

# Introduction

The following are ongoing and completed COVID-19 studies listed on the World Health Organization's International Clinical Trials Registry Platform (WHO ICTRP). Information in the WHO ICTRP comes from clinical trial databases maintained by other countries or regions of the world. COVID-19 studies listed on ClinicalTrials.gov are not included in the list below, but can be found using our search for COVID-19.
To give researchers and the public rapid access to studies on COVID-19 in other countries, ClinicalTrials.gov will update this list weekly. You can also access the information directly from the WHO ICTRP.

We developed a dashboard to display the information related to COVID-19 related clinical trials, including data visualization of world trials and U.S. trials, clustering tirals by similarity and predicting trials' opening status. Our audience of visualization parts could be any people who are curious about COVID-19 related clinical trials even without professional knowledge since all plots are intuitive and easy to understand. But for the clustering part and predicting part, our audience needs to have some knowledge of statistics. We all know that clinical research is a crucial step to overcome the virus. The purpose of our project is to give people a basic understanding of COVID-19 related clinical research and to increase confidence for everyone to defeat the virus.

# Data Cleaning
Based on my teammates' basic data cleaning process, I further cleaned the data to make it comply with SQLite3 schema.

## 1. Get data
The first step I did was to get the data and take a deep look of the data. Basically, my teammates did the following steps.
- selected columns of interests
- cleaned date columns into standard format with month and year
- cleaned locations columns, including country, city of state, instituions
- transferred all letters to upper cases
- replaced "nan" and "NaN" to `np.nan`

Based on their effort, I found that there are still fields cleaning could be optimized and thus further cleaned the data to make them efficient and easy to use.

## 2. Clean Study Design and Intervention
First, I found that there are two field are JSON-like, which is defined by myself Strictly speaking, they are not JSON data, but they looked like JSON data. In each cell of them, several pairs of key and value exist, which are separated by vertical bars. Here are examples of `Study Design` and `Intervention`.
    # Study Design
    'ALLOCATION: RANDOMIZED|INTERVENTION MODEL: SINGLE GROUP ASSIGNMENT|MASKING: TRIPLE (PARTICIPANT, CARE PROVIDER, INVESTIGATOR)|PRIMARY PURPOSE: TREATMENT'
    # Intervention
    'DRUG: DUVELISIB|PROCEDURE: PERIPHERAL BLOOD DRAW|DRUG: PLACEBO'


## 3. Create a SQLite3 schema following 3NF

#### Handle one field with multiple information
1. outcome_measures, 
2. sponsor_collaborators, 
3. funded_bys, 
4. study_type

#### Handle JSON-like field
1. Study Design
2. Intervention

#### Handle Other fields
replace NA

## 3. Set up SQL server

<!-- code -->


# Exploratory Data Analysis
Initially, we all did EDA seperately to explore data and get basic sense ourselves.
1. Study design
2. duration

# Openning Status classification
In the beginning, I plan to predict if the clinical trial will succeed or not based on variable `status`. I thought it would be really meaningful to have an idea of this most important sense. But when I looked into the data, I found that the data is highly imbalanced--it indicates that only one trial is successful, whose status is 'APPROVED FOR MARKETING'.
## 1. Data preprocessing

#### Further cleaning
1. age
2. gender
3. phases
4. Study Type
5. results
6. funded bys
7. locations
8. duration

#### Imputation
1. enrollment
2. duration

#### Handling imbalance data

## 2. Modelling
1. Comparing models
I used pycaret to compare models and tuned models.
<!-- plot -->
2. ROC / confusion matrix / 
difficulty: cannot fit in our env

## 3. Insights
Most important variables
future plan in predicting probability of success


# Building dashboard
Each of us built our own dashboard page. 
## 1. Drawing Plots

## 2. Combining to Streamlit


# Reflection and Conclusion
enviroment imcompatible
github usage







[GitHub]: https://github.com/eveyimi/eveyimi.github.io

