import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def get_var_wise_null_percentage(df, features, threshold=0.50):
    '''
    DESCRIPTION
    -----------
    This function is used to get the features that has null value percentage
    greater than the threshold    
    
    Parameters
    ----------
    df-DataFrame
    features-list of features
    threshold (between 0-1)- default is 0.50(i.e 50%)

    Returns
    -------
    DataFrame with features, count of null values and the percentages
        

    '''
    try:
        op_df =pd.DataFrame(columns=['Feature','count of null values','percentage of null values'])
        for feature in features:
            null_count=len(df[df[feature].isnull()])
            null_percentage=null_count/len(df)
            if null_percentage>threshold:
                op_df_length = len(op_df)
                op_df.loc[op_df_length] = [feature,null_count,null_percentage]
        return op_df
    except Exception as e:
        print(repr(e))



def get_var_wise_zero_percentage(df, features, threshold=0.50): 
    
    '''
    DESCRIPTION
    -----------
    This function is used to get the features that has zero values percentage
    greater than the threshold    
    
    Parameters
    ----------
    df-DataFrame
    features-list of features
    threshold (between 0-1)- default is 0.50(i.e 50%)

    Returns
    -------
    DataFrame with features, count of zero values and the percentages
        

    '''
    
    
    try:
        op_df=pd.DataFrame(columns=['Feature','count of zero values','percentage of zero values'])
        for feature in features:
            zero_count=(df[feature] == 0).sum(axis=0)
            zero_percentage=zero_count/len(df)
            if zero_percentage>threshold:
                op_df_length = len(op_df)
                op_df.loc[op_df_length] = [feature,zero_count,zero_percentage]
        return op_df
    except Exception as e:
        print(repr(e))
        
        
def drop_constats_vars(df,features):
    '''
    DESCRIPTION
    -----------
    This function is used to drop the features which have a single constant value   
    
    Parameters
    ----------
    df-DataFrame
    features-list of features

    Returns
    -------
    Modified DataFrame,List of the variables that are dropped
        

    '''
    
    
    try:
        features_list= df[features].select_dtypes(include=np.number).columns.tolist()
        variables_dropped=[]
        for feature in features_list:
            if df[feature].nunique()<=1:
                df.drop(feature,axis=1,inplace=True)
                variables_dropped.append(feature)
        if len(variables_dropped)==0:
            return 'No variables with constant values'
        else:
            return df,variables_dropped  #op:variable name or no variables with constant values
    except Exception as e:
        print(repr(e))
        
        
def drop_vars_with_high_nulls(df, features, threshold=0.50):
    
    '''
    DESCRIPTION
    -----------
    This function is used to drop the features which have null percentage greater than threshold   
    
    Parameters
    ----------
    df-DataFrame
    features-list of features
    threshold (between 0-1)- default is 0.50(i.e 50%)

    Returns
    -------
    Modified DataFrame, List of the variables that are dropped along with nulls percentage
        

    '''
    
    try:
        variables_dropped=[]
        for feature in features:
            null_count=len(df[df[feature].isnull()])
            null_percentage=null_count*100/len(df)
            if null_percentage>threshold*100:
                df.drop(feature,inplace=True,axis=1)
                variables_dropped.append([feature,round(null_percentage,5)])
        if len(variables_dropped)==0:
            return 'No variables with high null values'
        else:
            return df,variables_dropped
    except Exception as e:
        print(repr(e))
        
        


def drop_selected_vars(df, features):
    
    '''
    DESCRIPTION
    -----------
    This function is used to drop the selected features 
    
    Parameters
    ----------
    df-DataFrame
    features-list of features
    

    Returns
    -------
    Modified DataFrame,List of the features that are dropped 
        

    '''
  
    try:
        variables_dropped=[]
        for feature in features:
            df.drop(feature,inplace=True,axis=1)
            variables_dropped.append(feature)

        return df,variables_dropped 
    except Exception as e:
        print(repr(e))
        
        
        
        
def get_var_wise_outliers(df, features):
    
    '''
    DESCRIPTION
    -----------
    This function is used to get the variable wise outliers 
    
    Parameters
    ----------
    df-DataFrame
    features-list of features
    

    Returns
    -------
    List of List with the features, count of outliers and list of unique outliers
        

    '''
    
    
    
    try:
        outlier_output=[]
        features_list= df[features].select_dtypes(include=np.number).columns.tolist()
        for feature in features_list:
            outliers=[]
            feature_output=[]
            Q3 =df[feature].quantile(0.75) 
            Q1 =df[feature].quantile(0.25) 
            IQR= Q3-Q1
            low_lim = Q1 - 1.5 * IQR
            up_lim = Q3 + 1.5 * IQR
            outliers = [x for x in df[feature] if x < low_lim or x > up_lim]
            feature_output.extend([feature,len(outliers),set(outliers)])
            outlier_output.append(feature_output)
        return outlier_output 
    except Exception as e:
        print(repr(e))



def get_qualitative_vars(df, features):
    
    '''
    DESCRIPTION
    -----------
    This function is used to get the all the categorical features 
    
    Parameters
    ----------
    df-DataFrame
    features-list of features
    

    Returns
    -------
    List categorical features
        

    '''
    
    try:
        features_list= df[features].select_dtypes(include=object).columns.tolist()
        if len(features_list)==0:
            return "No categorical features"
        else:
            return features_list 
        
    except Exception as e:
        print(repr(e))
        
        
        
        
def get_quantitative_vars(df, features):
    
    '''
    DESCRIPTION
    -----------
    This function is used to get the all the numerical features 
    
    Parameters
    ----------
    df-DataFrame
    features-list of features
    

    Returns
    -------
    List numerical features
        

    '''
    
    try:
        features_list= df[features].select_dtypes(include=np.number).columns.tolist()
        if len(features_list)==0:
            return "No Numerical features"
        else:
            return features_list 
        
    except Exception as e:
        print(repr(e))
        
        
        
def get_datetime_vars(df, features):
    
    
    '''
    DESCRIPTION
    -----------
    This function is used to get the all the DateTime features 
    
    Parameters
    ----------
    df-DataFrame
    features-list of features
    

    Returns
    -------
    List DateTime features
        
    '''
    
    try:
        features_list=[]
        for feature in features:
            if df[feature].dtype=='datetime64[ns]':
                features_list.append(feature)
        if len(features_list)==0:
            return "No variables with datetime datatype"
        else:
            return features_list #return list
    except Exception as e:
        print(repr(e))
        
        
        

def impute_outliers(df,features,mechanism="median"):
    
    
    '''
    DESCRIPTION
    -----------
    This function is used to impute outliers with either mean or median
    
    Parameters
    ----------
    df-DataFrame
    features-list of features
    mechanism(mean or median)- default is median

    Returns
    -------
    Imputes DataFrame,List of imputed features
        
    '''
    
    
    
    try:
        features_list= df[features].select_dtypes(include=np.number).columns.tolist() 
        imputed_features=[]
        for feature in features_list:
            Q3 =df[feature].quantile(0.75) 
            Q1 =df[feature].quantile(0.25) 
            IQR= Q3-Q1
            low_lim = Q1 - 1.5 * IQR
            up_lim = Q3 + 1.5 * IQR
            if mechanism=='median':
                df[feature] = np.where(df[feature] >up_lim, df[feature].median(),df[feature])
                df[feature] = np.where(df[feature] <low_lim, df[feature].median(),df[feature])
                imputed_features.append(feature)
            else:
                df[feature] = np.where(df[feature] >up_lim, df[feature].mean(),df[feature])
                df[feature] = np.where(df[feature] <low_lim, df[feature].mean(),df[feature])
                imputed_features.append(feature)
        if len(imputed_features)==0:
            return "No feature is imputed"
        else:
            return df,imputed_features
    except Exception as e:
        print(repr(e))

        

def encode_categorical_features(df,features,encoding_type='label encoding'):
    
    
    '''
    DESCRIPTION
    -----------
    This function is used to encode the categorical features
    
    Parameters
    ----------
    df-DataFrame
    features-list of features
    encoding_type(label encoding or one-hot encoding)- default is label encoding

    Returns
    -------
    DataFrame with encoded categorical features
        
    '''
    
    
    
    try:
        cat_list = df[features].select_dtypes(include=np.object).columns.tolist()
        if encoding_type=='label encoding':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            for i in cat_list:
                df[i] = le.fit_transform(df[i])
        else:
            for i in cat_list:
                df=pd.concat([df,pd.get_dummies(df[i],prefix=i)],axis=1).drop([i],axis=1)

        return df
    except Exception as e:
        print(repr(e))


def get_variable_variation_info(df, threshold=0.50):
    
    '''
    DESCRIPTION
    -----------
    This function is used to get the variables that has variation more than the threshold  
    
    Parameters
    ----------
    df-DataFrame
    features-list of features
    threshold (between 0-1)- default is 0.50(i.e 50%)

    Returns
    -------
    Two lists good_variation_variables,poor_variation_variables 
        

    '''
    
 
    try:
        from sklearn.feature_selection import VarianceThreshold
        num_list=df.select_dtypes(include=np.number).columns.tolist()
        selector= VarianceThreshold(threshold)
        selector.fit(df[num_list])
        good_variation_variables=df.select_dtypes(include=np.number).columns[selector.get_support(indices=True)].tolist()
        poor_variation_variables =[x for x in df.select_dtypes(include=np.number).columns if x not in good_variation_variables] 
        return good_variation_variables,poor_variation_variables 
    except Exception as e:
        print(repr(e))


def drop_low_variation_variabls(df,features):
    
    '''
    DESCRIPTION
    -----------
    This function is used to get the features that has variation more than the threshold  
    
    Parameters
    ----------
    df-DataFrame
    features-list of features


    Returns
    -------
    DataFrame with dropped low variation features
        
    '''
    
    try:
        good_variation_variables,poor_variation_variables = get_variable_variation_info(df)
        for feature in features:
            if feature in poor_variation_variables:
                df.drop(feature,inplace=True,axis=1)
        return df
    except Exception as e:
        print(repr(e))
        
        
def save_box_plots(df,features,file_name): 
    
    '''
    DESCRIPTION
    -----------
    This function is used to save the boxplots of the features into a pdf that are numeric. 
    
    Parameters
    ----------
    df-DataFrame
    features-list of features
    

    
    
    '''

    
    import datetime
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    try:
        today_date=datetime.datetime.now().strftime("%B")+datetime.datetime.now().strftime("%d")
        pp=PdfPages('{}_{}.pdf'.format(file_name,today_date))
        
        for i in df[features].select_dtypes(exclude='object'):
            fig,ax = plt.subplots(1,1)
            boxplot=pd.DataFrame(df[i]).boxplot()
            pp.savefig(fig)
        print ("Saved Plots to ",file_name+"_"+today_date+".pdf")
        pp.close()
    except Exception as e:
        print(repr(e))
        
        
def save_scatter_plots(df,features_name, file_name):
    
    
    '''
    DESCRIPTION
    -----------
    This function is used to save the scatterplots of the numerical features into a pdf . 
    
    Parameters
    ----------
    df-DataFrame
    features-list of features
    

    
    '''
    
    
    import datetime
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    try:
        today_date=datetime.datetime.now().strftime("%B")+datetime.datetime.now().strftime("%d")
        pp=PdfPages('{}_{}.pdf'.format(file_name,today_date))

        for i in df.select_dtypes(exclude='object'):
            for j in df.select_dtypes(exclude='object'):
                if (i!=j):
                    fig,ax = plt.subplots(1,1)
                    plt.scatter(x=df[i],y=df[j])
                    plt.xlabel(i)
                    plt.ylabel(j)
        pp.savefig(fig)
        print ("Saved Plots to ",file_name+"_"+today_date+".pdf")
        pp.close()
    except Exception as e:
        print(repr(e))




def save_scatter_plots_individaul(df,feature1,feature2, file_name):
    
    '''
    DESCRIPTION
    -----------
    This function is used to save the scatterplots between two numerical features into a pdf . 
    
    Parameters
    ----------
    df-DataFrame
    features-list of features

    
    '''

    import datetime
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    try:
        today_date=datetime.datetime.now().strftime("%B")+datetime.datetime.now().strftime("%d")
        pp=PdfPages('{}_{}.pdf'.format(file_name,today_date))
        fig,ax = plt.subplots(1,1)
        plt.scatter(x=feature1,y=feature2)
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        pp.savefig(fig)
        print ("Saved Plots to ",file_name+"_"+today_date+".pdf")
        pp.close()
    except Exception as e:
        print(repr(e))
        
        
        
def save_scatter_plots_wrt_traget(df,features_name,target, file_name):
    
    
    '''
    DESCRIPTION
    -----------
    This function is used to save the scatterplots with respect to target numerical variable into a pdf . 
    
    Parameters
    ----------
    df-DataFrame
    features-list of features

    
    '''
    
    import datetime
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    try:
        today_date=datetime.datetime.now().strftime("%B")+datetime.datetime.now().strftime("%d")
        pp=PdfPages('{}_{}.pdf'.format(file_name,today_date))
        for i in df.select_dtypes(exclude='object'):
            fig,ax = plt.subplots(1,1)
            plt.scatter(x=df[i],y=df[target])
            plt.xlabel(i)
            plt.ylabel(target)
        pp.savefig(fig)
        print ("Saved Plots to ",file_name+"_"+today_date+".pdf")
        pp.close()
    except Exception as e:
        print(repr(e))
        
        
def show_highly_correlated_independent_vars(df,feature_list,threshold=0.90): 
    
    '''
    DESCRIPTION
    -----------
    This function is used to get the highly correlated features 
    
    Parameters
    ----------
    df-DataFrame
    features-list of features
    threshold (between 0-1)- default is 0.50(i.e 50%)

    Returns
    -------
    DataFrame with the features and the correlation value of the features

    '''
    
    try:
        corr_matrix = df[feature_list].corr()
        corr_table= corr_matrix.where(np. triu (np.ones (corr_matrix.shape), k=1).astype (np.bool)).unstack().reset_index().dropna()
        high_corr_table=corr_table.where((corr_table[0]<-threshold)|(corr_table[0]>threshold)).dropna()
        high_corr_table.columns = ['var1', 'var2','corr_value']
        return high_corr_table
    except Exception as e:
        print(repr(e))
        

def save_correlation_heatmap(df, features,file_name):
   
    '''
    DESCRIPTION
    -----------
    This function is used to save the correlation heatmap to a pdf file 
    
    Parameters
    ----------
    df-DataFrame
    features-list of features
    file_name- name of the file to be saved

    '''
   
    import datetime
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    try:
        today_date=datetime.datetime.now().strftime("%B")+datetime.datetime.now().strftime("%d")
        fig, ax = plt.subplots(1,1)
        sns.heatmap(df.corr(method='pearson'), annot=True, fmt='.4f', 
                cmap=plt.get_cmap('coolwarm'), cbar=False, ax=ax)
        ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")
        pp=PdfPages('{}_{}.pdf'.format(file_name,today_date))
        pp.savefig(fig)
        pp.close()
        return print("correlation map saved !")
    except Exception as e:
        print(repr(e))
        
        
def drop_poorly_correlated_features(df,features,target,threshold=0.50):
    '''
    DESCRIPTION
    -----------
    This function is used to drop the poorly correlated features with respect to target which is less than threshold value
    
    Parameters
    ----------
    df-DataFrame
    features-list of features
    threshold (between 0-1)- default is 0.50(i.e 50%)

    Returns
    -------
    DataFrame with dropped features 

    '''    
    
    try:
        corr_table=df.corrwith(df[target])

        corr_table=corr_table.reset_index()
        corr_table.columns=['features','corr_value']
        low_corr_features1=corr_table.where(((corr_table['corr_value']<threshold)&(corr_table['corr_value']>0))|((corr_table['corr_value']<(-threshold))&(corr_table['corr_value']<0))).dropna()
        df.drop(low_corr_features1['features'],inplace=True,axis=1)
        return df
    except Exception as e:
        print(repr(e))
        
        
def get_model_based_feature_ranking(df,output_variable,model):
    
    '''
    DESCRIPTION
    -----------
    This function is used ranking for different models and also the aggregated rank of all the models and also overall rank
    
    Parameters
    ----------
    df-DataFrame
    output_variable-target variable
    model- Regression or Classification

    Returns
    -------
    DataFrame with model ranks and overall ranks 

    '''    
    
    
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
    from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression,LogisticRegression
    from sklearn.linear_model import Lasso,LassoCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import MinMaxScaler

    numer_list=df.select_dtypes(include=np.number).columns.tolist()
    cat_list = df.select_dtypes(include=np.object).columns.tolist()
    le = LabelEncoder() # label encoding the categorical variables
    for i in cat_list:
        df[i] = le.fit_transform(df[i])
    X=df.drop(output_variable,axis=1)
    Y=df[output_variable]
    mm_scaler = MinMaxScaler()
    X_scaled = mm_scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled,columns=X.columns)
    no_of_cols = X_scaled.shape[1]
    if model=="Classification":
        #Boosting
        gbm_class = GradientBoostingClassifier(random_state=0)
        gbm_class = gbm_class.fit(X,Y)
        gbm_class_sc = gbm_class.fit(X_scaled,Y)
        gbm_model = SelectFromModel(gbm_class, prefit=True)
        gbm_model_sc = SelectFromModel(gbm_class_sc, prefit=True)
        #Random Forest
        rf_class = RandomForestClassifier(n_estimators=100,random_state=0)
        rf_class = rf_class.fit(X,Y)
        rf_class_sc = rf_class.fit(X_scaled,Y)
        rf_model = SelectFromModel(rf_class, prefit=True)
        RF_tree_featuresTrain=X.loc[:, rf_model.get_support()]
        rf_model_sc = SelectFromModel(rf_class_sc, prefit=True)
        RF_tree_featuresTrain_sc=X_scaled.loc[:, rf_model_sc.get_support()]
        
        #Logistic Regression with l2
        logsel = SelectFromModel(LogisticRegression(C=0.5, penalty='l2',random_state=0))
        sel_ = logsel.fit(X, Y)
        sel_sc =  logsel.fit(X_scaled, Y)
        #Ranks
        RF_cls_Ranks = pd.DataFrame(zip(X.columns, rf_class.feature_importances_),\
                                columns=["rf_class_features","rf_class_scores"])\
                                    .sort_values(by=["rf_class_scores"],ascending=False)
        RF_cls_Ranks["rf_class_ranks"]= RF_cls_Ranks["rf_class_scores"].rank(ascending=False)
        GBM_cls_Ranks = pd.DataFrame(zip(X.columns, gbm_class.feature_importances_),\
                            columns=["gbm_cls_features","gbm_cls_scores"])\
                                .sort_values(by=["gbm_cls_scores"],ascending=False)
        GBM_cls_Ranks["gbm_cls_ranks"]= GBM_cls_Ranks["gbm_cls_scores"].rank(ascending=False)
        ranks_df = pd.concat([ RF_cls_Ranks,GBM_cls_Ranks], axis=1)
        ranks_df["mean_rank_score"] = ranks_df[['rf_class_ranks','gbm_cls_ranks']].mean(axis=1)
        ranks_df[['rf_class_features','mean_rank_score']].sort_values(by=['mean_rank_score'])
        ranks_df['overall_rank']=ranks_df['mean_rank_score'].rank()
    elif model=="Regression":
        #Boosting
        gbmr = GradientBoostingRegressor(random_state=0)
        gbm_reg = gbmr.fit(X,Y)
        gbm_reg_model = SelectFromModel(gbm_reg, prefit=True)#for manual understanding - has been using it
        gbm_reg_sc = gbmr.fit(X_scaled,Y)
        gbm_reg_model_sc = SelectFromModel(gbm_reg_sc, prefit=True)
        #Random Forest
        rfr = RandomForestRegressor(n_estimators=100,random_state=0)
        rf_reg = rfr.fit(X,Y)
        rf_reg_sc = rfr.fit(X_scaled,Y)
        rf_reg_model = SelectFromModel(rf_reg, prefit=True)
        RFR_tree_featuresTrain=X.loc[:, rf_reg_model.get_support()]
        rf_reg_model_sc = SelectFromModel(rf_reg_sc, prefit=True)
        RFR_tree_featuresTrain_sc=X_scaled.loc[:, rf_reg_model_sc.get_support()]
        # Lasso
        las = LassoCV(random_state=0)
        las_reg = las.fit(X,Y)
        las_reg_sc = las.fit(X_scaled,Y)
        #Ranks
        RF_reg_Ranks = pd.DataFrame(zip(X.columns, rf_reg.feature_importances_),\
                            columns=["rf_reg_features","rf_reg_scores"])\
                                .sort_values(by=["rf_reg_scores"],ascending=False)
        RF_reg_Ranks["rf_reg_ranks"]= RF_reg_Ranks["rf_reg_scores"].rank(ascending=False)

        las_reg_ranks = pd.DataFrame(zip(X.columns,las_reg.coef_),\
                                columns=["las_reg_features","las_reg_scores"]).\
                                sort_values(by=["las_reg_scores"],ascending=False)
        las_reg_ranks["las_reg_ranks"]= las_reg_ranks["las_reg_scores"].drop(las_reg_ranks[las_reg_ranks['las_reg_scores']==0].index).rank(ascending=False)
        las_reg_ranks["las_reg_ranks"].fillna(value=0,inplace=True)
        GBM_reg_Ranks = pd.DataFrame(zip(X.columns, gbm_reg.feature_importances_),\
                            columns=["gbm_reg_features","gbm_reg_scores"])\
                                .sort_values(by=["gbm_reg_scores"],ascending=False)
        GBM_reg_Ranks["gbm_reg_ranks"]= GBM_reg_Ranks["gbm_reg_scores"].rank(ascending=False)
        ranks_df = pd.concat([RF_reg_Ranks, las_reg_ranks,GBM_reg_Ranks], axis=1)
        ranks_df["mean_rank_score"] = ranks_df[['rf_reg_ranks','las_reg_ranks','gbm_reg_ranks']].mean(axis=1)
        ranks_df[['rf_reg_features','mean_rank_score']].sort_values(by=['mean_rank_score'])
        ranks_df['overall_rank']=ranks_df['mean_rank_score'].rank()
    return ranks_df
    
        

