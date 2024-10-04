########################################################################################################################
# Last update: 2024-03-12 16:13 MDT
########################################################################################################################
import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LassoCV, Lasso
from captum.attr import KernelShap, ShapleyValueSampling
np.random.seed(42)
import torch
########################################################################################################################


def feature_preprocessing(df_X, df_y, indicator_col_name, target_col_name, lasso_threshold=0.0, classification=True,
                          null_threshold=0.35, var_threshold=0.05,
                          alpha_range_LASSO=None, n_alphas_LASSO=100, max_iter_LASSO=1000, cv_LASSO=5,
                          selection_LASSO='random', max_depth_forest=7, bootstrap_forest=True, alpha_Boruta=0.05,
                          max_iter_Boruta=100, perc_Boruta=100, two_step_Boruta=True, lassoBorutaBoth = 'Both', verbose=1):
    """
    This preprocessing pipeline involves four procedures:
    Step 1: Nan-percentage-removal: features with more than a pre-specified percentage of Nan will be removed
    Step 2: Low-variance-removal: features with variance lower than a pre-specified threshold will be removed
    Step 3: LASSOCV-obtain: obtain linear features (using sklearn.linear_model.LASSOCV)
    Step 4: Boruta-obtain: obtain non-linear features (using
    :param df_X: the feature dataset (as a Pandas 2-dimensional DataFrame)
    :param df_y: the true labels of the target as (a Pandas 2-dimensional DataFrame)
    :param indicator_col_name: the name of the indicator column expected to exist in both X and y (as a string)
    :param target_col_name: the name of the target column (as a string)
    :param classification: a classification task if True and a regression task if False (default = True)
    :param null_threshold: the percentage of samples is required to preserve a column (as a float in (0, 1))
    :param var_threshold: variance (lower) threshold for features (Default = 0.05)
    :param alpha_range_LASSO: range of alphas when performing LASSOCV (Default = None, computed automatically)
    :param n_alphas_LASSO: number of alphas when performing LASSOCV (Default = 100)
    :param max_iter_LASSO: maximum iterations when fitting a LASSOCV model (Default = 1000)
    :param cv_LASSO: number of folds for cross-validation when performing LASSOCV (Default = 10)
    :param selection_LASSO: selector of LASSOCV to update coefficients (Default = 'random')
    :param max_depth_forest: maximum depth of the tress in random forest (Default = 7)
    :param bootstrap_forest: random bootstrapping in random forest if True (Default = True)
    :param alpha_Boruta: significance level in Boruta (Default = 0.05)
    :param max_iter_Boruta: maximum iterations in Boruta (Default = 100)
    :param perc_Boruta: the percentile in Boruta (Default = 95)
    :param two_step_Boruta: harsh Bonferroni correction will be used if False in Boruta (Default = True)
    :param lassoBorutaBoth: needs to be one of 'lasso', 'boruta', 'both'. Which sections of pipeline to run.
    :param verbose: fulling logging if 1 and partial logging if 0  (Default = 1)
    :return:
    """
    assert indicator_col_name in df_X.columns
    assert indicator_col_name in df_y.columns
    assert target_col_name in df_y.columns
    assert df_X[indicator_col_name].equals(df_y[indicator_col_name])
    assert selection_LASSO in ['cyclic', 'random']
    assert classification in [True, False]
    assert 0 < null_threshold < 1
    assert var_threshold >= 0
    assert alpha_range_LASSO is None or all([0 < alpha < 1 for alpha in alpha_range_LASSO])
    assert max_iter_LASSO > 0
    assert cv_LASSO > 0
    assert selection_LASSO in ['cyclic', 'random']
    assert max_depth_forest > 0
    assert bootstrap_forest in [True, False]
    assert 0 < alpha_Boruta < 1
    assert max_iter_Boruta > 0
    assert two_step_Boruta in [True, False]
    assert verbose in [0, 1]
    assert lassoBorutaBoth in ['lasso', 'boruta', 'both']
    ####################################################################################################################

    indicator_col = df_X[indicator_col_name]
    X = df_X.drop(indicator_col_name, axis=1)
    y = df_y[target_col_name]
    if verbose == 1:
        print(f"Dimension of the dataset: {X.shape}", flush=True)

    ####################################################################################################################
    # Step 1 Nan-percentage-removal: Remove features with > null_threshold of missing values
    ####################################################################################################################
    del_col_list = [col for col in X.columns if X[col].isna().sum() > null_threshold * X.shape[0]]
    X.drop(del_col_list, axis=1, inplace=True)
    if verbose == 1:
        print(f"Deleting {len(del_col_list)} features with >{null_threshold * 100}% null", flush=True)

    ####################################################################################################################
    # Step 2 Low-variance-removal: Remove features with variance lower than var_threshold
    ####################################################################################################################
    X_shape_old = X.shape
    var_selector = VarianceThreshold(threshold=var_threshold)
    try: 
       X = var_selector.fit_transform(X)
    except ValueError:
       print('No features passed variance threshold. Exiting')
       return None, None
    X = pd.DataFrame(X, columns=var_selector.get_feature_names_out())
    X_shape_new = X.shape
    if verbose == 1:
        print(f"Deleting {X_shape_old[1]-X_shape_new[1]} features with variance at most {var_threshold}", flush=True)

####################################################################################################################
    # Prepare output file
    ####################################################################################################################
    feature_names = X.columns
    df_coef = pd.DataFrame(feature_names, columns=['Feature'])
    ####################################################################################################################
    # Defining a model class to compute Shapley values later
    ####################################################################################################################
    class model_torch:
        def __init__(self, model):
            self.model = model

        def forward(self, X_):
            return torch.Tensor(self.model.predict(X_))
            
    ####################################################################################################################
    # Step 3 LASSOCV-obtain: Obtain important linear features from LASSOCV
    ####################################################################################################################
    lasso_feat_set = set()
    B_feat_set = set()
    if lassoBorutaBoth == 'lasso' or lassoBorutaBoth == 'both':
       if selection_LASSO == 'random':
           lasso_selector = LassoCV(n_alphas=n_alphas_LASSO, alphas=alpha_range_LASSO, max_iter=max_iter_LASSO,
                                    cv=cv_LASSO, selection='random', random_state=42)
       else:
           lasso_selector = LassoCV(n_alphas=n_alphas_LASSO, alphas=alpha_range_LASSO, max_iter=max_iter_LASSO,
                                    cv=cv_LASSO, selection='cyclic')

       lasso_selector.fit(X.to_numpy(), y.to_numpy())
       #df_lasso = pd.DataFrame(zip(lasso_selector.feature_names_in_, lasso_selector.coef_, ), columns=['Feature', 'Lasso_coef'])
       df_coef['Lasso_Coeff'] = lasso_selector.coef_
       df_coef['Lasso_Coeff_Abs'] = np.abs(lasso_selector.coef_)
       gs = ShapleyValueSampling(model_torch(lasso_selector).forward)   #(best_lasso_model.predict) 
       attributions = gs.attribute(torch.tensor(np.array(X.values))).numpy()
       attributions = np.abs(attributions)
       lasso_shapvals = np.mean(attributions, axis=0)
       df_coef['Lasso_Shap_Val'] = lasso_shapvals
    ####################################################################################################################
    # Step 4 Boruta-obtain: Obtain important non-linear features from Boruta
    ####################################################################################################################
    if lassoBorutaBoth == 'boruta' or lassoBorutaBoth == 'both':
       bootstrap = True if bootstrap_forest else False
       random_state_Boruta = 42 if bootstrap_forest else None
       if classification:
           randomForestModel = RandomForestClassifier(class_weight='balanced', max_depth=max_depth_forest,
                                                      bootstrap=bootstrap, random_state=random_state_Boruta,
                                                      n_jobs=-1)
       else:
           randomForestModel = RandomForestRegressor(max_depth=max_depth_forest,
                                                     bootstrap=bootstrap, random_state=random_state_Boruta,
                                                     n_jobs=-1)

       perc_union_dict, X_feats = {}, list(X.columns)

       def boruta_perc_union_features(perc_value):
           if perc_value in perc_union_dict.keys():
               return perc_union_dict[perc_value]
           B_selector = BorutaPy(randomForestModel, n_estimators='auto', verbose=0, alpha=alpha_Boruta,
                                 perc=perc_value, max_iter=max_iter_Boruta, two_step=two_step_Boruta, random_state=42)
           B_selector.fit(X.to_numpy(), y.to_numpy())
           boruta_feature_importances = B_selector._get_imp(X.to_numpy(), y.to_numpy())
           B_model_ = B_selector._get_model()
           return B_model_, boruta_feature_importances, B_selector.support_

       B_model, B_importances, B_support = boruta_perc_union_features(perc_Boruta)
       df_coef['Boruta_Importances'] = B_importances
       Boruta_SHAP_model = ShapleyValueSampling(model_torch(B_model).forward)
       Boruta_attr = Boruta_SHAP_model.attribute(torch.Tensor(X.values)).cpu().numpy()
       MASHAP_Boruta = np.mean(np.abs(Boruta_attr), axis=0)
       df_coef['Boruta_Shap_Val'] = MASHAP_Boruta
       B_feat_set = set(feature_names[i] for i in range(len(feature_names)) if list(B_support)[i])
       #Note: B_feat_set != num nonzero Boruta importances.
       #Boruta uses an internal threshold. We set these to 0 for features not meeting the threshold
       #even though they really have an importance to show Boruta did not select these features
       for i, row in df_coef.iterrows():
          if row['Boruta_Importances'] != 0 and row['Feature'] not in B_feat_set: #need to set to 0
             df_coef.at[i, 'Boruta_Importances'] = 0
           
    #return dataset that contains all non-zero cols of lasso unioned with non-zeros of Boruta
    lassocollist = []
    borutacollist = []
    if lassoBorutaBoth == 'lasso' or lassoBorutaBoth == 'both':
       lassocollist = df_coef[df_coef['Lasso_Coeff'].abs() > lasso_threshold]
       lassocollist = lassocollist['Feature'].tolist()
    union_LASSO_Boruta = np.unique(lassocollist + list(B_feat_set))
    #del_row_list is used to remove features from db_coef where importance == 0
    del_row_list = []
    for i, row in df_coef.iterrows():
       if row['Feature'] not in union_LASSO_Boruta:
          del_row_list.append(i)
    df_coef = df_coef.drop(del_row_list, axis=0)
    return pd.concat([indicator_col, X[list(union_LASSO_Boruta)]], axis=1), df_coef

########################################################################################################################


