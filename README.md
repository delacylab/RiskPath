# RiskPath
RiskPath: a multistep predictive pipeline for temporally-sensitive biomedical risk stratification that achieves very strong performance and is tailored to the constraints and demands of biomedical practice. The core algorithm is a LSTM network adapted to data with the characteristics common in clinical practice (tabular; non-square; collected annually; ≤10 timepoints) and rendered translationally explainable by extending the Shapley method of computing feature importances for timeseries data and embedding this into the algorithm. RiskPath also provides data-driven approaches for streamlining features in timeseries data before and during model training and analyzing performance-complexity trade-offs in model construction.

Full paper available at https://www.medrxiv.org/content/10.1101/2024.09.19.24313909v1

# Data Preparation (General)

The following steps should be used to prepare tabular datasets for RiskPath.

The set of independent variables (feature set) should have their observations represented by a series of two-dimensional tabular datasets, each saved in .csv format, one per time period and per learning partition (e.g., training partition and test partition, respectively). Call them the feature datasets. RiskPath assumes that observations are represented by rows and features by columns for each feature dataset. 

The variable of interest to be predicted (target) should be a separate two-dimensional dataset saved in .csv format. Call it the target dataset. To ensure that rows in the target dataset and feature datasets are associated to the same observations, RiskPath requires an identifier column in both the target dataset and each feature dataset and each row in the feature set should have a matching identifier in the target dataset. This identifier is used as the runtime parameter IDKey when running scripts. The target must be represented as a column in the target dataset. Below we term the column name of the target as target name.

The current implementation of RiskPath requires consistent feature labels across time periods for each file in the feature dataset. The naming pattern of 

# (target name)^(time period identifier)_(partition label).csv 

must be used where (target name) encodes the column name of the target, (time period identifier) encodes the time period, and (partition label) encodes the partition in the full dataset. For example, a filename might look like hypertension^timeperiod3_test.csv.

# Feature Filtering

RiskPath offers feature filtering to obtain a selected subset of important features from your dataset to promote parsimonious solutions and reduced runtime. Coarse feature selection is available with several parameters that user can specify.

    • null_threshold: Remove features with more than a specified percentage of missing data (e.g., 0.3).
    • var_threshold: Remove features with a variance lower than the specified value (e.g., 0.05).

Finer feature filtering is also offered using different methods:

[LASSOCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html) from scikit-learn<br>
[Boruta](https://github.com/scikit-learn-contrib/boruta_py) used for nonlinear feature filtering<br>
Union of LASSOCV and Boruta. <br>

While each method allows various runtime parameters, we only highlight those that are more commonly used to control the strictness of feature selection in practice.

By assuming a linear dependency between the feature set and the target, LASSOCV aims to capture the linearly important feature subset. It performs a grid search over a range of regularization terms alphas (computed automatically in scikit-learn), and identifies the best alpha using cross-validation (i.e., alpha used in the best-fitting model with performance measured by mean squared errors averaged across k-fold validation sets). The lasso_threshold parameter is used to control the strictness of the selection process is lasso_threshold: minimum absolute value of the coefficient from the best-fitting model that a feature must have to be considered important. 

To account for non-linear relationships between the features and the target, we adopt Boruta as a complementary method to capture these important features. Boruta is a feature selection method that extends random forest models by embedding statistical techniques including multiple hypothesis testing and sample shuffling. By comparing the importance score (e.g., Gini impurity) of a feature with those of the shadow features (i.e., original features with their observations shuffled) in a random forest, Boruta employs iterative hypothesis tests to determine if a feature is important or unimportant. Parameters that can be controlled for the Boruta subroutine include:

    • perc_Boruta: the percentile of the importance score of the shadow features. A high percentile (e.g., 100) refers to a strict selection process (i.e., minimizing Type I error) and a low percentile (e.g., 50) refers to a lenient selection process (i.e., minimizing Type II error)

Users can fine-tune the filtering process with the parameter lassoBorutaBoth that decides which method(s) to be used (e.g., ‘None’, ‘lasso’, ‘boruta’, or ‘both’). The lasso_threshold and perc_Boruta runtime parameters are required but should be set to -1 if a user is not using that particular method.

RiskPath uses two scripts to accomplish this: Feature_Filtering.py and LASSOCV_boruta_perc.py. Feature_Filtering.py sets various input/output options for LASSOCV_Boruta_perc.py and does batch feature filtering. The user is welcome to use either script they would like depending on their use case but will have to manually handle their own input/output concerns if they use LASSOCV_Boruta_perc.py directly.

Typically, feature filtering is performed in the training partition with the resultant selected features conformed in the test partition. The full set of runtime parameters for Feature_Filtering.py follows:

    • input_dir – input directory to your input feature files.
    • output_dir – prefix to output directories. Three output directories will be created using this parameter as a root name. One will have the filtering method, lasso_threshold, and perc_Boruta appended to output_dir and contain the original input files subsetted with only the features flagged as important. The second output directory will have _Importances appended to the first output directory name and contain a spreadsheet of features with nonzero importances. The third will have a suffix of _AllTPs appended and will contain subsetted files from both partitions and should serve as input to the next step of this pipeline.
    • target_file – path to your target matrix file.
    • target_list – comma separated list of targets. Columns with these names must appear in your target file.
    • Classification_or_regression – True for classification, else regression
    • lassoBorutaBoth – Feature filtering method – Must be one of LASSO, boruta, both, None. Use None to bypass feature filtering entirely. Even if None, this script must still be run to ensure naming convention compliance with next steps.
    • IDKey – the primary key of your identifier e.g. ‘subjectkey’
    • TrainLabel – label used for your training data in filenames e.g. tt in our examples. 
    • TestLabel - label used in your test set filenames e.g. rep in examples here.
    • lasso_threshold - A float value that sets the threshold for determining lasso importance.
    • perc_Boruta - An integer value that determines Boruta strictness (values range from 1-100).
    • Run with:
    • python Feature_Filtering.py path/to/input/features directory/to/save/results path/to/target/file target True|False LASSO|boruta|both primary_key trainLabel testLabel lasso_threshold perc_Boruta

# Data Preparation for Deep Learning

The script Matrix_Assembly.py is required to prepare the feature datasets, with or without features filtered, for subsequent deep learning processes. It concatenates all the feature datasets (in different time periods) into two different matrices, one in 2-dimensional (2D) and one in 3-dimensional (3D). The 2D case is a straightforward horizontal concatenation of the features across different feature datasets. In contrast, the 3D case makes use of the assumption that if a feature is important in any time period, it is important in all time periods. Thus, the resultant 3D matrix contains features that are selected in at least one time period. But if it was not present in another time period, RiskPath will fill the missing value with 1.
Runtime parameters:<br>

    • input_dir – This should be a path to the AllTPs directory created in the previous step.
    • Targetfileprefix - The prefix of your target filename as specified in the prerequisites.
    • Targets – comma separated list of targets 
    • IDKey – primary key of your data.
    • Tps – comma separated list of time periods you want assembled. This allows creating matrices for specific time periods rather than all of them at once.
    • TrainLabel – label used for your training data in filenames.
    • TestLabel - label used in your test set filenames
    • output_dir – Output directory prefix. Similar to the last step, this parameter is a root directory name that will have the selected time periods and dimension appended to the end of the directory name. Inside that directory, subdirectories will be created for each target.
    • Run with:
    • python Matrix_Assembly.py path/to/_AllTPs/directory/from/last/step path/with/prefix/to/target_csv target primary_key comma_separated_list_of_numerical_time_periods trainLabel testLabel
    • As an example, our running of this program looks like python Matrix_Assembly.py Filter_Output_boruta_AllTPs targets_after_balancing_ hypertension IDNO 0,1,2,3,4,5,6,7,8,9 train test Final_Matrices

# Deep Learning
The script torch_class_gridsearch.py executes the main prediction pipeline in RiskPath. RiskPath allows optimization of the topology of the embedded Long Short Term Memory deep learning algorithm. It does this via a grid search approach to examine the effect of varying with unit size (network width) of each layer where the minimum and maximum are runtime parameters.  For 3D matrices, we use a LSTM with 2 layers (if the bidirectional runtime parameter == True, then those layers will be bidirectional for a total of 4 layers). For 2D matrices, we use a simple deep learning feedforward network with 3 layers. Both are implemented in Pytorch with the AdamW optimizer. kfold validation is implemented for training with k specified as a runtime parameter and early stopping based on validation loss inside the kfold with maximum epochs = 150 as the default behavior. 

RiskPath uses the GradientShap method from the captum library (https://captum.ai/) to compute feature importances on a per feature per time period level on the unseen test data using the training data as background. This helps determine which of your features are important predictors. These scores are also averaged across time periods for a single importance score per feature. The single importance score per feature results per model are recorded in target_importances.csv. The importance score per feature per time period results are in target_importancesTPs.csv. Riskpath also generates ROC curves  and Shapley value plots (with feature importances and the data averaged over time periods due to plot functions not supporting three dimensions) as zip files containing the per model res

Runtime parameters:<br>

    • input_dir - path to files created by Matrix_Assembly
    • target - target to predict
    • output_dir - directory to save results
    • numOfGPUs - Number of GPUs to use. -1 for CPU only
    • bidir – Use bidirectional LSTM, True or False. Ignored if using 2D matrices but must still be present.
    • K - Number of cross-validation folds
    • layer_size_min – minimum units per layer for grid search
    • layer_size_max - maximum units per layer for grid search
    • preset_feature_list – Set to true to use a specific feature list (must be placed in code). False to use all features.
    • TrainLabel - label used for your training data in filenames.
    • TestLabel - label used for your test data in filenames.
    • Run with:
    • python torch_class_gridsearch.py path/to/features/from/MatrixAssembly target output_directory numberOfGPUs bidir k layer_size_min layer_size_max False trainLabel testLabel
    • An example from our work is python torch_class_gridsearch.py ../CHS_10year_Cohort1/Final_lasso_0-1-2-3-4-5-6-7-8-9_3D HYPER_balanced ../CHS_10year_Cohort1/Results_3D_lasso 2 True 10 5 1200 False train test
    
Outputs
RiskPath generates the following output:

    • File target_AllModels.csv - Deep Learning performance metrics including:
    • unitsize - unit size of the model
    • elapsed_epochs - This is averaged over the  kfolds
    • number of features - number of features in the data
    • model_parameters - number of parameters in the model
    • The following parameters are reported for the training, validation and unseen test data (these results are obtained by taking the best model as determined by validation auc and then running that model on the unseen test data):
    • fitness (BIC fitness score), accuracy, precision, recall, spec, f1, auc,_loss (Cross Entropy Loss)
    • File Statistics.csv - summary statistics including number of rows in your training and test data files, total number of features, number of kfolds, matrix fill percentage (non-null percentage), running time in hours, minutes, seconds.
