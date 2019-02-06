% Clear prior data
clc; clear;

%Set random number seed for reproducability
rng(34);

% Load the data in a dataframe / table
train_df = readtable('Credit_Train.csv');
test_df = readtable('Credit_Test.csv');

% Financial Assumptions - NPV assumptions for calculating financial error
principal = 1000;
interest_rate = 0.10;
discount_rate = 0.06;
term = 5;

recovery_rate = 0.50;

NPV_typical = -principal + pvfix(discount_rate,term,interest_rate*principal,principal,0);
NPV_defaulted = -principal + (recovery_rate * principal);

% Model Assumptions - modify model parameters including train size,
% oversampling level, use of engineered features, etc.
% Set parameter true / false given required run
train_data_size = 25000;
test_data_size = 15000;
oversample = true;
oversample_target_ratio = 0.25;

add_engineered_feats = false;
feature_selection = false;

if add_engineered_feats
    feature_subset = {'RevolvingUtilizationOfUnsecuredLines','age'...
        ,'DebtRatio','NumberOfOpenCreditLinesAndLoans','WeightedLate', 'PNIpD'...
        'NumberOfDependents','SeriousDlqin2yrs'};
else
    feature_subset = {'RevolvingUtilizationOfUnsecuredLines','age'...
        ,'NumberOfTime60_89DaysPastDueNotWorse','DebtRatio','MonthlyIncome','NumberOfOpenCreditLinesAndLoans',...
        'NumberOfDependents','SeriousDlqin2yrs'};
end

cap_outliers = true;

f_measure = 2;

% Cap outlier values - outlier values are capped at values determined by
% exploratory data analysis in order to improve NB performance
if cap_outliers
    % NumberOfTime30_59DaysPastDueNotWorse
    train_df.NumberOfTime30_59DaysPastDueNotWorse = min(train_df.NumberOfTime30_59DaysPastDueNotWorse,15);
    test_df.NumberOfTime30_59DaysPastDueNotWorse = min(test_df.NumberOfTime30_59DaysPastDueNotWorse,15);

    % NumberOfTime60_89DaysPastDueNotWorse
    train_df.NumberOfTime60_89DaysPastDueNotWorse = min(train_df.NumberOfTime60_89DaysPastDueNotWorse,10);
    test_df.NumberOfTime60_89DaysPastDueNotWorse = min(test_df.NumberOfTime60_89DaysPastDueNotWorse,10);

    % NumberOfTimes90DaysLate
    train_df.NumberOfTimes90DaysLate = min(train_df.NumberOfTimes90DaysLate,10);
    test_df.NumberOfTimes90DaysLate = min(test_df.NumberOfTimes90DaysLate,10);
    
    % Monthly Income
    train_df.MonthlyIncome = min(train_df.MonthlyIncome,25000);
    test_df.MonthlyIncome = min(test_df.MonthlyIncome,25000);
end

% Plotting charts - Can toggle whether charts are created to increase
% performance speed of runs
% Random forest feature importance
plot_rf_fi = false;
% Correlation matrix
plot_covm = false;
% ROC curves
plot_ROC_curves = false;

% Select size of train & test split
train_df = train_df(1:train_data_size,:);
test_df = test_df(1:test_data_size,:);

% Feature engineering - created new features based on Kaggle interviews
if add_engineered_feats 
    % F11= Net Income=[Monthly Income*(1- Debt Ratio)]
    train_df.NetIncome = train_df.MonthlyIncome.*(1- train_df.DebtRatio); 
    test_df.NetIncome = test_df.MonthlyIncome.*(1- test_df.DebtRatio);

    % F12= Positive net Income per dependent = max(0, [Monthly Income*(1- Debt Ratio)/(No of dependents+1)]
    train_df.PNIpD = max(0,(train_df.MonthlyIncome.*(1- train_df.DebtRatio))./(train_df.NumberOfDependents+1));
    test_df.PNIpD = max(0,(test_df.MonthlyIncome.*(1- test_df.DebtRatio))./(test_df.NumberOfDependents+1));

    % F13=Weighted No of late Payments = 1*(Times 30 to 59 days)  +3*(Times 60 to 89 days)   +5*(Times>90 days)
    train_df.WeightedLate = (train_df.NumberOfTime30_59DaysPastDueNotWorse)+3*(train_df.NumberOfTime60_89DaysPastDueNotWorse)+5*(train_df.NumberOfTimes90DaysLate);
    test_df.WeightedLate = (test_df.NumberOfTime30_59DaysPastDueNotWorse)+3*(test_df.NumberOfTime60_89DaysPastDueNotWorse)+5*(test_df.NumberOfTimes90DaysLate);
    
    % Creates maximum weighting for F13 to avoid substantial outliers
    wl_max_threshold = 30;
    train_df.WeightedLate = min(train_df.WeightedLate,wl_max_threshold);
    test_df.WeightedLate = min(test_df.WeightedLate,wl_max_threshold);
end

% Oversample the minority class in order to change class imbalance to
% target rate specified in model assumptions section (e.g. 0.20)
if oversample
    % Create a dataset of only defaulted samples
    minority_df = train_df(train_df.SeriousDlqin2yrs == 1,:);
    % Count the number of non-defaulted instances
    n_majority = sum(train_df.SeriousDlqin2yrs == 0);
    % Create new minority samples based on random sampling of existing
    n_new_minority_samples = round(((oversample_target_ratio * n_majority) / (1-oversample_target_ratio)) - size(minority_df,1));
    % Append newly created minority samples to main dataframe / table
    train_df = [train_df; minority_df(randsample(size(minority_df,1),n_new_minority_samples,true),:)];
end

% Perform feature selection if selected based on identified features above
if feature_selection
    train_df = train_df(:,feature_subset);
    test_df = test_df(:,feature_subset);
end
    
% Ensure there are no missing values
assert(sum(ismissing(train_df),'all') == 0,'There are missing values')

% Set target and remove from X_train and X_test
target = 'SeriousDlqin2yrs';
y_train = train_df(:,{target});
X_train = removevars(train_df,{target});

y_test = test_df(:,{target});
X_test = removevars(test_df,{target});

% Convert tables to arrays for model training
X_train_labels = X_train.Properties.VariableNames;
X_train = table2array(X_train);
y_train = table2array(y_train);
X_test = table2array(X_test);
y_test = table2array(y_test);

% Set weight scores for weighted recall/precision output as desired
recall_w = 0.80;
precision_w = 0.20;

% Hyperparameter tuning inputs using K-fold gridsearch
n_folds = 5;
cv = cvpartition(size(X_train,1),'KFold', n_folds);

% Random Forest Model -------------------------------------------------
% Select possible model hyperparameters
rf_possible_parameters = [4; 6; 8; 10; 12];

% Instantiate parameter gridsearch list
rf_parameters = zeros(n_folds * numel(rf_possible_parameters),3); 
idx = 1;

% Check performance at each level of tree depth by retraining model on
% cross validation train set and evaluating on cross validation test set
% for each hyperparameter
for i = 1:n_folds
    for j = 1:numel(rf_possible_parameters)
        rf_mdl = TreeBagger(100, X_train(cv.training(i),:),...
            y_train(cv.training(i),:),'MaxNumSplits',rf_possible_parameters(j),...
            'NumPredictorsToSample',round(sqrt(size(X_train,2))));
        [~, rf_pred_probs] = predict(rf_mdl, X_train(cv.test(i),:));
        rf_pred_probs = rf_pred_probs(:,2);
        [~, ~,~, rf_AUC] = perfcurve(y_train(cv.test(i),:),rf_pred_probs,1);
        rf_parameters(idx,:) = [i rf_possible_parameters(j) rf_AUC];
        idx = idx + 1;
    end
end

% Instantiate the mean rf_parameters 
cv_rf_parameters = zeros(numel(rf_possible_parameters),2);

% Average each hyperparameter across folds to determine best performance
for i = 1:numel(rf_possible_parameters)
cv_rf_parameters(i,:) = [rf_possible_parameters(i) mean(...
    rf_parameters((rf_parameters(:,2) == rf_possible_parameters(i)),3))];
end

% Select hyperparameter with maximum mean performance
[~,optimal_rf_hp_index] = max(cv_rf_parameters(:,2));
optimal_rf_hp = cv_rf_parameters(optimal_rf_hp_index,1);

% Retrain model on full training set with optimized hyperparameter and
% return ROC inputs and AUC
rf_mdl = TreeBagger(500, X_train, y_train, 'MaxNumSplits', optimal_rf_hp,...
    'NumPredictorsToSample',round(sqrt(size(X_train,2))),'OOBPredictorImportance','on');
[~, rf_pred_probs] = predict(rf_mdl, X_test);
rf_pred_probs = rf_pred_probs(:,2);
[rf_FPR, rf_TPR, ~, rf_AUC] = perfcurve(y_test,rf_pred_probs,1);

% Return false positive and false negative rates for financial error metric
% Multiply error rates by appropirate NPV cost and select ROC threshold which
% minimizes the financial error
[rf_FP, rf_FN, rf_threshold] = perfcurve(y_test,rf_pred_probs,1,'XCrit','fp','YCrit','fn');
rf_FP_NPV = abs(rf_FP .* NPV_typical);
rf_FN_NPV = abs(rf_FN .* NPV_defaulted);
rf_NPV_error = rf_FP_NPV + rf_FN_NPV;
[rf_min_NPV_error, rf_min_NPV_error_idx] = min(rf_NPV_error);
rf_threshold = rf_threshold(rf_min_NPV_error_idx);
rf_predictions = double(rf_pred_probs>=rf_threshold);

% Create confusion matrix and model evaluation outputs based on predictions
% which are above threshold determined by financial error metric
rf_cm = confusionmat(y_test, rf_predictions);
rf_precision = rf_cm(2,2) / (rf_cm(2,2)+rf_cm(1,2));
rf_recall = rf_cm(2,2) / (rf_cm(2,2)+rf_cm(2,1));
rf_fmeasure = ((f_measure^2+1) .* rf_precision .* rf_recall) ./ ((f_measure^2*rf_precision) + rf_recall);
rf_weighted_score = recall_w * rf_recall + precision_w * rf_precision;

rf_MaxPossNPV = NPV_typical.*(rf_cm(1,1)+rf_cm(1,2));
rf_ActualNPV = (NPV_typical.*rf_cm(1,1))+(NPV_defaulted.*rf_cm(2,1)); 
rf_prop_EconomicValue = rf_ActualNPV / rf_MaxPossNPV;

% Display model outputs
disp('Random Forest Hyperparameter Performance:')
disp(cv_rf_parameters)

disp(['Random Forest AUC: ', num2str(rf_AUC)])
disp('Random Forest Confusion Matrix:')
disp(rf_cm)

disp(['Mean Economic NPV Error: ', num2str(rf_min_NPV_error / size(y_test,1))])
disp(['Expected NPV: ', num2str(rf_ActualNPV / size(y_test,1))])
disp(['Proportion of Economic Value: ', num2str(rf_prop_EconomicValue)])
disp(' ')

disp('Random Forest Confusion Matrix Metrics:')
disp(['Precision: ', num2str(rf_precision)])
disp(['Recall: ', num2str(rf_recall)])
disp(['F-Measure B=',num2str(f_measure),': ', num2str(rf_fmeasure)])
disp(['Weighted: ', num2str(rf_weighted_score)])
disp(' ')


% Naive Bayes Model -------------------------------------------------
% Select possible model hyperparameters
nb_possible_parameters = {'normal'; 'kernel'};

% Instantiate parameter gridsearch list
nb_parameters = zeros(n_folds * numel(nb_possible_parameters),3); 
idx = 1;

% Check performance for normal and kernal distribution by retraining model on
% cross validation train set and evaluating on cross validation test set
% for each hyperparameter
for i = 1:n_folds
    for j = 1:numel(nb_possible_parameters)
        nb_mdl = fitcnb(X_train(cv.training(i),:),...
            y_train(cv.training(i),:),'Distribution', nb_possible_parameters{j});
        [~, nb_pred_probs] = predict(nb_mdl, X_train(cv.test(i),:));
        nb_pred_probs = nb_pred_probs(:,2);
        [~, ~,~, nb_AUC] = perfcurve(y_train(cv.test(i),:),nb_pred_probs,1);
        nb_parameters(idx,:) = [i j nb_AUC];
        idx = idx + 1;
    end
end

% Instantiate the mean rf_parameters 
cv_nb_parameters = zeros(numel(nb_possible_parameters),2);

% Average each hyperparameter across folds to determine best performance
for i = 1:numel(nb_possible_parameters)
cv_nb_parameters(i,:) = [i mean(nb_parameters((nb_parameters(:,2) == i),3))];
end

% Select hyperparameter with maximum mean performance
[~,optimal_nb_hp_index] = max(cv_nb_parameters(:,2));
optimal_nb_hp = cv_nb_parameters(optimal_nb_hp_index,1);

% Retrain model on full training set with optimized hyperparameter and
% return ROC inputs and AUC
nb_mdl = fitcnb(X_train, y_train, 'Distribution', nb_possible_parameters{optimal_nb_hp});
[~, nb_pred_probs] = predict(nb_mdl, X_test);
nb_pred_probs = nb_pred_probs(:,2);
[nb_FPR, nb_TPR, ~, nb_AUC] = perfcurve(y_test,nb_pred_probs,1);

% Return false positive and false negative rates for financial error metric
% Multiply error rates by appropirate NPV cost and select ROC threshold which
% minimizes the financial error
[nb_FP, nb_FN, nb_threshold] = perfcurve(y_test,nb_pred_probs,1,'XCrit','fp','YCrit','fn');
nb_FP_NPV = abs(nb_FP .* NPV_typical);
nb_FN_NPV = abs(nb_FN .* NPV_defaulted);
nb_NPV_error = nb_FP_NPV + nb_FN_NPV;
[nb_min_NPV_error, nb_min_NPV_error_idx] = min(nb_NPV_error);
nb_threshold = nb_threshold(nb_min_NPV_error_idx);
nb_predictions = double(nb_pred_probs>=nb_threshold);

% Create confusion matrix and model evaluation outputs based on predictions
% which are above threshold determined by financial error metric
nb_cm = confusionmat(y_test, nb_predictions);
nb_precision = nb_cm(2,2) / (nb_cm(2,2)+nb_cm(1,2));
nb_recall = nb_cm(2,2) / (nb_cm(2,2)+nb_cm(2,1));
nb_fmeasure = ((f_measure^2+1) .* nb_precision .* nb_recall) ./ ((f_measure^2 * nb_precision) + nb_recall);
nb_weighted_score = recall_w * nb_recall + precision_w * nb_precision;

nb_MaxPossNPV = NPV_typical.*(nb_cm(1,1)+nb_cm(1,2));
nb_ActualNPV =(NPV_typical.*nb_cm(1,1))+(NPV_defaulted.*nb_cm(2,1));
nb_prop_EconomicValue = nb_ActualNPV / nb_MaxPossNPV;

% Display model outputs
disp('Naive Bayes Hyperparameter Performance:')
disp(cv_nb_parameters)

disp(['Naive Bayes AUC: ', num2str(nb_AUC)])
disp('Naive Bayes Confusion Matrix:')
disp(nb_cm)

disp(['Mean Economic NPV Error: ', num2str(nb_min_NPV_error / size(y_test,1))])
disp(['Expected NPV: ', num2str(nb_ActualNPV / size(y_test,1))])
disp(['Proportion of Economic Value: ', num2str(nb_prop_EconomicValue)])
disp(' ')

disp('Naive Bayes Confusion Matrix Metrics:')
disp(['Precision: ', num2str(nb_precision)])
disp(['Recall: ', num2str(nb_recall)])
disp(['F-Measure B=',num2str(f_measure),': ', num2str(nb_fmeasure)])
disp(['Weighted: ', num2str(nb_weighted_score)])

% Plot feature importance from random forests
if plot_rf_fi
    figure();
    bar(rf_mdl.OOBPermutedPredictorDeltaError);
    title('Random Forests Feature Importance');
    set(gca,'xticklabel',{'Line Util.','Age','Times 30-59 Days Late','Debt to Income','Income Per Month',...
        'Unsecured Lines','Times 90 Days Late','Secured Lines','Times 60-89 Days Late','Dependents'});
    xtickangle(45);
end

% Plot covariance matrix
if plot_covm
    figure()
    covmat = corrcoef(double([X_train y_train]));
    x = size(covmat, 2);
    imagesc(covmat);
    set(gca,'XTick',1:x);
    set(gca,'YTick',1:x);
    set(gca,'YTickLabel',[X_train_labels 'SeriousDlqin2yrs']);
    axis([0 x+1 0 x+1]);
    grid;
    colorbar;
end

% Plot ROC curves
if plot_ROC_curves
    figure()
    plot(rf_FPR,rf_TPR)
    hold on 

    plot(nb_FPR,nb_TPR)
    xlabel('False Positive Rate'); ylabel('True Positive Rate');
    title('Receiver Operating Characteristic Curves')
    %legend('Random Forests','Naive Bayes','Location','Best')
    hold off
end


% Appendix: Naive Bayes manual hyperparameter testing
%nb_mdl = fitcnb(X_train, y_train, 'Distribution', 'kernel','kernel',...
    %{'triangle','triangle','triangle','triangle','triangle','triangle',...
    %'triangle','triangle','triangle','triangle','triangle','triangle','triangle'});
%nb_mdl = fitcnb(X_train, y_train, 'Distribution', 'kernel','kernel',...
    %{'epanechnikov','epanechnikov','epanechnikov','epanechnikov', ...
    %'epanechnikov','epanechnikov','epanechnikov','epanechnikov','epanechnikov',...
    %'epanechnikov','epanechnikov','epanechnikov','epanechnikov'});
%nb_mdl = fitcnb(X_train, y_train, 'Distribution', 'kernel','kernel',...
    %{'normal','normal','normal','normal','normal','normal','normal','normal',...
    %'normal','normal','normal','normal','normal'});