from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.linear_model import Lasso

from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


metrics = {
    'Model :':[],
    'Train RMSE' :[],
    'CV RMSE':[],
    'TEST RMSE':[],
    'Train R2':[],
    'CV R2':[],
    'Test R2':[]}


models=[
    ("LR ",LinearRegression()),
    ("DT ",DecisionTreeRegressor()),
    ("SGD    ",SGDRegressor()),
    ("RF     ",RandomForestRegressor()),
    ("GB     ",GradientBoostingRegressor()),
    ("Lasso ",Lasso()),
    ("Linear SVR ",LinearSVR())]
                            
                            

def add(df,model,train_rmse,train_r2,cv_rmse,cv_r2,test_rmse,test_r2):
    a=df.loc[len(df.index)] = [model,train_rmse,train_r2,cv_rmse,cv_r2,test_rmse,test_r2]
    return a



def build_model_regression_(Xtr,ytr,Xt,yt):
    
    metrics_comparison = pd.DataFrame(metrics)  #---------------- CHANGE

    for name,model in models:
        model.fit(Xtr,ytr)
        
        pred_train = model.predict(Xtr) #make the predictions over train set
        mse_train = mean_squared_error(ytr, pred_train)# y (actual sale price) and y_hat (predicted saleprice) 
        r2_train = r2_score(ytr, pred_train)

        pred_test = model.predict(Xt) #make the predictions over train set
        mse_test = mean_squared_error(yt, pred_test)# y (actual sale price) and y_hat (predicted saleprice) 
        r2_test = r2_score(yt, pred_test)
                
        train_rmse,train_r2 = np.sqrt(mse_train),r2_train   # Train
        
        folds = KFold(n_splits = 10, shuffle = True, random_state = 42)   #CV
        scores = cross_val_score(model, Xtr, ytr, scoring='r2', cv=folds)
        
        r2_scores = cross_val_score(model, Xtr, ytr, scoring='r2', cv=folds)
        rmse_scores = cross_val_score(model, Xtr, ytr, scoring='neg_mean_squared_error', cv=folds)

        test_rmse,test_r2 = np.sqrt(mse_test),r2_test     # Test
        
        add(metrics_comparison,name,train_rmse,np.sqrt(abs(rmse_scores)).mean(),test_rmse,\
            train_r2,r2_scores.mean(),test_r2)

    a = pd.DataFrame(metrics_comparison)
    a = a.set_index("Model :").T
    return a
