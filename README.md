# House-Price-Prediction-using-machine-Learning

This Project was done under guidance of my mentor Praveen Kumar Vasari...

# Problem :
## Given features of The House, Predict SalePrice of the House
## It is a SUPERVISED LEARNING PROBLEM :
       Given X's predict y
    
# Approach :
  * Given a Dataset which contains 80 Features...
  * Deleted The Unnecessary Features like Id Column...
  * Considered The Required variables From the Dataset..
  * CONDUCTED EDA to get insights into data...
  * Splited the data using train_test_split
  * Fixed missing values using imputation techniques
  * Fed the X_train to the Different models such as :
  *       1. Linear Regression, 2. Decision Trees,3. Random Forest,4. Stochastic Gradient Boosting,5. Lasso,6. Linear Support vector Regressor,K-NN
  * Cross Validation Done...
  * Then Calculated METRICS Such as :
           R2_Score (R-Square score) and RMSE (Root mean Square Error) of Train Data scores, Cross validation score,and Test Data Scores
          
  * Then Selected the Best Model using METRICS Comparison and Found Lasso Regressor as Best Model and Given New Data, it is Predicing with Minimum Error
