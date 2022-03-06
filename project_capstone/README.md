# Udacity-Data-Scientist
# Capstone Project

In this capstone project, we will analyse the user data from Sparkify between 2018-10-01 to 2018-12-03 to try to predict the users who may leave the platform.


We define  the users who choose to cancel the subscriptionthose as churn users.


The blog link:https://medium.com/@sxio067/sparkify-churn-prediction-with-pyspark-48f3b92875c

# Other Info:
1. This project now is using a sub-dataset (128Mb) of the full dataset (12GB).
2. Requirements: 
- python==3.6.x
- numpy==1.21.5
- pandas==1.3.5
- seaborn==0.8.1
- scikit-learn==0.21
3. Spark = 2.4.5

# Summary
From the prediction result above, we can see that the models are obviously overfitted. They performances on testset are poor. This is because the data for training model is samll. After feature engineering, there are only hunderds of records can be used for model training.

It is expected to use the large dataset(12G) with comparatively complete information that the models could learn.

From the EDA, we found some valueable information that helps to save the users.


# Acknowledgements

Udacity provides the best practices in data processing including data cleaning, data processing process and so on, which guides me in this completing this project.