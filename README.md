# Credit-Card-Default-Prediction
Building a Machine Learning Classification model to predict whether a customer will default in credit card payment or not.

Abstract:

Credit risk plays a major role in the banking industry business. Banks' main activities involve granting loan, credit card, investment, mortgage, and others. Credit cards have been one of the most booming financial services by banks over the past years. However, with the growing number of credit card users, banks have been facing an escalating credit card default rate. For this, data analytics can provide solutions to tackle the current phenomenon and manage credit risks. 
This project provides a performance evaluation of credit card default prediction. Thus, Logistic Regression, XGB Classifier, SVM Classifier and Random Forest Classifier are used to test the variable in predicting credit default and Random Forest proved to have the highest precision, recall and area under the curve. This result shows that Random Forest best describes which factors should be considered with recall of 83 % and an AUC ROC score of 0.93 while assessing the credit risk of credit card customers.

1. Problem Statement:

Credit risk management problem researches have been around credit scoring only. It would go a long way to research how machine learning can be applied to qualitative areas for better computations of credit risk exposure by predicting probabilities of default.
The purpose of this project is to conduct qualitative analysis on credit card default risk by using interpretable machine learning models with accessible customer data, instead of credit score or credit history, with the goal of assisting and speeding up the human decision making process.

2. Introduction:

The government vigorously promotes the economic construction of large- and medium-sized cities, which not only improves people’s living standards but also changes people’s consumption concept and consumption mode. People are more and more inclined to spend ahead of time and mortgage their “credit” to the bank to enjoy certain things in advance. However, when consuming, people often lack rational thinking and overestimate their ability to repay loans to banks in time. On the one hand, it increases the loan risk of banks; on the other hand, it increases the credit crisis of consumers themselves. With a large number of banks selling credit cards, the phenomenon of credit card default emerges one after another. It is very important for banks to effectively identify high-risk credit card default users. Generally speaking, compared with the credit card customers who have not paid their loans overdue, there are fewer overdue repayments. This variable feature of overdue and overdue loan repayment is called “two classifications” in machine learning prediction. In the prediction of “two classifications,” a few categories are called positive examples (default), and most categories are called counterexamples (non-default)


3. Data:

This dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005. It includes 30,000 rows and 25 columns.There is a clear explanation on these 25 variables. Overall, the cleanliness and usability of this data are high. 

The data generated from these variables can be analyzed to draw inferences regarding Credit Card default prediction.

4. Steps Involved:

The steps involved in this exploratory data analysis are:

Understanding the Data:

The first step in any exploratory data analysis and modeling is to understand what we are looking at, but without going into the details. We try to understand the problem we want to solve, thinking about the entire dataset and the meaning of the variables.
This phase can be slow and boring but it will give us the opportunity to make an opinion of our dataset.


Data Preparation and Cleaning:

In this stage, I tried to do basic cleaning of the data in order to continue the analysis like:

Renamed a few variables:-

I renamed a few variables like ‘default.payment.next.month’ to ‘default’.

Treated the null values:-

There were no null values to take care of in our dataset.

Handling discrepancies in the data :-

There were some discrepancies in the classes of some variables like in ‘Education’ column, values like 4,5, and 6 were present which were not explained in the data description. So I combined 4,5 and 6 to 0.

Data type conversion:-

Then I converted the data type of all the columns from object to integer type.

Outlier Treatment: There were many outliers in numerical continuous features. I have used the capping method to take care of outliers and skewness.                       

                      
Exploratory Analysis:

In this section, I explored the data to get insights about it. Some of the key information that I could collect from the data are:-

Distribution of Classes in the target variable

There was class imbalance in the target variable classes. I used the ‘SMOTE’ oversampling method to take care of this imbalance.
 
Categorical Variable Analysis:

Gender : Although there are more female credit card holders, the default proportion among men is higher.

Education : The default proportion decreases with increase in education level. Also, customers with higher education levels get higher credit limits.

Married : Married people have higher proportions of default.

Age : Default proportion is lowest for people in their 30s and then steadily rises with age.


Numerical Variable Analysis:

Credit Limit : I checked for the distribution of credit limit with respect to default and found with increasing credit limit, default proportion decreases.

PAY_1 to PAY_6 : There was a huge jump from May(pay_5) to June(pay_4) when delayed payment increased significantly, then it peaked at July(pay_3).Things started to get better in August(pay_2) and September(pay_1)

BILL_1 to BILL_6 : There were some negative bills in every month’s bill statement. An informed guess would be, these were refunds from last billing                    cycle.

Feature Engineering & Scaling:

Then I did some feature engineering using the ‘get_dummies’ function for features like Education, Gender, Married etc. 
The values of the numerical continuous features were at different scales. I used the Standard scaler for feature scaling.


Modeling:

After declaring the independent and target variables and splitting them into test and train data, I used a number of algorithms to train from the train data and then predict on the test data. Then I compared these predicted classes with the actual classes with the help of some evaluation metrics.

Algorithms : The algorithms that I used are : 

Logistic Regression:

In statistics, the Logistic model is a statistical model that models the probability of an event taking place by having the log-odds for the event to be a linear combination of one or more independent variables. In regression analysis, Logistic Regression is estimating the parameters of a logistic model. Logistic regression predicts the output of a categorical dependent variable. Therefore, the outcome must be a categorical or a discrete value like yes or no and 0 or 1. But instead of giving the exact value as 0 and 1, it gives the probabilistic value which lie between 0 and 1. 

Random Forest Classifier:

Random Forest is a popular machine learning algorithm that belongs to the supervised learning technique. It is based on ensemble learning, which is a process of combining multiple classifiers to solve a complex problem and to improve the performance of the model. Instead of relying on one decision tree, the random forest takes the prediction from each tree and based on the majority votes of predictions, it predicts the output.

XGBoost Classifier:

XGBoost, which stands for Extreme Gradient Boosting, is a scalable, distributed gradient boosted decision tree machine learning library. It provides parallel tree boosting and is the leading machine learning library for regression, classification and ranking problems.

Support Vector Machines(SVM):

Support Vector Machines is one of the most popular Supervised Learning algorithms, which is used for Classification as well as Regression problems. The goal of SVM is to create the best fit decision boundary  that can segregate n-dimensional space into classes so that we can easily put the new data point in the correct category in the future. The best decision boundary is called a hyperplane. 
	
Evaluation Metrics : The evaluation metrics that I used are : 

ROC_AUC score : 

AUC stands for “Area under the ROC curve. AUC measures the entire two dimensional area underneath the entire ROC curve. AUC provides an aggregate measure of performance across all possible classification thresholds. 

Precision : 

Precision is a metric that quantifies the number of correct positive predictions made. Precision, therefore, calculates the accuracy for the minority class. It is calculated as the ratio of correctly predicted positive examples divided by the total number of predicted positive examples.

Recall : 

Recall is a metric that quantifies the number of correct positive predictions made out of all positive predictions that could have been made. Unlike precision that only comments on the correct positive predictions out of all positive predictions, recall provides an indication of missed positive predictions.

F1-Score : 

The F1 score combines the precision and recall of a classifier into a single metric by taking their harmonic mean. It is  primarily used to compare the performance of two classifiers.
	

5. References:

Researchgate.net

wikipedia

GeeksforGeeks

