# Bertelsmann / Arvato Capstone Project

Udacity Data Scientist Nanodegree Capstone Project

All the code is in a public repository at the link below:

https://github.com/bergr7/Bertelsmann_Arvato_Project

A blog post on Medium summarising the work can be found at the link below:

https://medium.com/@bernardogarciadelrio/customer-segmentation-bertelsmann-arvato-capstone-project-f66360186392


## Table of Contents
1. Installation
2. Project Motivation
3. File Descriptions
4. Results
5. Licensing, Authors, Acknowledgements
6. MIT License

## Installation

- Libraries included in Anaconda distribution of Python 3.8.
- Packages versions:

    - pandas Version: 1.0.1
    - numpy Version: 1.19.2
    - matplotlib Version: 3.3.2
    - imblearn Version: 0.0
    - xgboost Version: 1.2.1
    - scikit-learn Version: 0.22.1
    - pickle5 Version: 0.0.11

## Project Motivation

In this capstone project, I was asked to analyse demographics data for customers of a mail-order sales company in
Germany and compare it against demographics information for the general population.

The datasets used were provided by Bertelsmann Arvato Analytics, and the project represents a real-life data science task.
This data has been strictly used for this project only and will be removed from my laptop after completion of the project
in accordance with Bertelsmann Arvato terms and conditions.

The project is divided in 3 parts:

**Part 1: Customer Segmentation Report**
In this part, I used unsupervised learning techniques to analyse features of established customers and the general
population in order to create customer segments with the aim of targeting them in marketing campaigns.

**Part 2: Supervised Learning Model**
Next step was to build a supervised learning machine learning model that predicted whether or not each individual will
respond to the campaign.

**Part 3: Kaggle Competition**
Finally, after spot-checking several models, choosing the most promising one and fine-tuning its hyperparemeters, I used
the model to make predictions on the campaign data as part of a Kaggle Competition.

## File Descriptions

The data provided is not publicly available according to Bertelsmann Arvato terms and conditions.

The code is contained in following files:

  - **Arvato_EDA.ipynb** - Jupyter Notebook that contains the EDA of the general population and customers data. Here the
  data preprocessing steps are identified.
  - **data_preprocessing.py** - Python script that performs data preprocessing on general population and customers data.
  - **Arvato_Customer_Segmentation.ipynb** - Jupyter Notebook with the Customer Segmentation Report.
  - **supervised_model_data_preprocessing.py** - Python script that performs data preprocessing on dataset for individuals
  that were targets of a marketing campaign.
  - **Arvato_Supervised_Model.ipynb** - Jupyter Notebook that contained a supervised learning model trained on dataset for individuals
  that were targets of a marketing campaign and tested on the test set.
  - **CRISP-DM.pdf** - A pdf with notes documenting the CRISP-DM process for this project.

## Results

**Customer Segmentation Report**

Due to the lack of meaning for some of the features utilised in the analysis, it is hard to describe some of the clusters.
I would probably drop features without feature meaning if I had to carry out the cluster analysis again or I would come
back to people with business domain knowledge if possible.

However, I have been able to identify the following groups of potential customers:

- Individuals that own luxurious and powerful cars and, therefore, these are likely to be in a good financial position and
have high incomes.
- Individuals that live in good neighborhood areas and municipalities. Also there is more than one car in their household.
- Relatively young individuals with great interest in environmental sustainability, that are somewhat dissapointed with
their current social status and interested in low interest rates financial services.

If the company were to launch a marketing campaign for the current services, it would be sensible to target these groups
of people.

I have also identified some groups of the German population that are under-represented in the customers base:

- Young people with average cars and income.
- Money savers with low interest in financial services.
- Young people with low-midclass cars and low income.

If the company were looking to reach a larger part of the German population, it might be worth designing some type of
financial service that aligns with the needs of these groups and target them with a different marketing campaign. The
company would need to consider the profitability of these services though before launching them as it is probably that
customers of these profile will not tend to spend a lot of money often.

**Supervised Learning Model**

I trained a XGBClassifier with a performance of ROC AUC score of 0.73226.

All the code is in a public repository at the link below:

https://github.com/bergr7/Bertelsmann_Arvato_Project

A blog post on Medium summarising the work can be found at the link below:

https://medium.com/@bernardogarciadelrio/customer-segmentation-bertelsmann-arvato-capstone-project-f66360186392

## Licensing, Authors and Acknowledgements

I would like to give credit to:

- Bertelsmann Arvato Analytics for providing the datasets.
- Stackoverflow community for making code questions and answers publicly available
- Jason Brownlee from Machine Learning Mastery for the post '8 Tactics to Combat
  Imbalanced Classes in Your Machine Learning Dataset'
- Udacity Data Scientist Nanodegree Colleagues for their previous work on this project.
- The DataDrivenInvestor for putting together an useful function to run PCA analysis.

## MIT License

Copyright 2020 Bernardo Garcia

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
