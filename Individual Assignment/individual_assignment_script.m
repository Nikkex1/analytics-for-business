%% Data Import & Initialization

data_table = readtable("weather_dataset.xlsx");
rng('default');

%% Task 1

% Identify groups in OBSERVED:
[OBS_groups,OBS_vals] = findgroups(data_table.OBSERVED);
% Calculate the maximum value for both groups:
max_Td_mu_byGroup = splitapply(@max,data_table.Td_mu,OBS_groups)

mean_P_var_byGroup = splitapply(@mean,data_table.P_var,OBS_groups)

% Create three columns for year, month and day:
[data_table.YY,data_table.MM,data_table.DD] = ymd(data_table.datetime);
% Remove the original datetime column:
data_table.datetime = [];

% Target variable OBSERVED (binary):
Y1 = data_table.OBSERVED;
% Target variable U_mu (continuous):
Y2 = data_table.U_mu;
% Predictor matrix:
X = data_table;
X(:,17:18) = [];
% Convert the table to array format:
X = table2array(X);

%% Task 2

% Reduce dimensionality with PCA:
[coeff,score,latent,~,explained] = pca(X);
% Plot the cumulative explained variance:
figure;
pareto(explained)
xlabel('Principal Component')
ylabel('Variance Explained (%)')
title('Variance Explained by Principal Components')
% Number of components:
enough_explained = cumsum(explained)/sum(explained) >= 94/100;
number_of_components = find(enough_explained,1)
% PCA-transformed dataset:
PCA_data = score(:,1:2);
pc1 = PCA_data(:,1);
pc2 = PCA_data(:,2);

% K-means clustering with Squared Euclidian distance
% with 2, 3 and 5 clusters:
kList = [2, 3, 5];
E = evalclusters(score,'kmeans','silhouette','KList',kList);
% Silhouette scores and optimal number of clusters:
silhouette_scores = E.CriterionValues
optimal_clusters = E.OptimalK
% Scatter plot of the optimal number of clusters:
figure;
gscatter(pc1,pc2, E.OptimalY)
xlabel('First Principal Component')
ylabel('Second Principal Component')
title('Optimal K-means Clustering')

% Hierarchical clustering with cosine distance
% and average linkage:
cos_dist = pdist(PCA_data, "cosine");
cos_tree = linkage(cos_dist,'average');
% Define and visualize the clusters (cut-off = 0.6):
cos_tree_clusters = cluster(cos_tree,'criterion','distance',...
    'cutoff',0.6);
optimal_clusters2 = length(unique(cos_tree_clusters))
dendrogram(cos_tree,'ColorThreshold',0.6)
% Silhouette score:
[silh_cos,h] = silhouette(PCA_data,cos_tree_clusters,"cosine");
cos_tree_silhouette = mean(silh_cos)

% Gaussian Mixture Model (GMM) with 3 clusters and
% 1000 iterations:
gmm_clusters = 3;
gmm_options = statset('MaxIter',1000);
gmm_pc_model = fitgmdist(PCA_data,gmm_clusters,'Options',...
    gmm_options)
% Probability of the 123rd observation belonging to
% each of the three clusters:
obs123 = PCA_data(123,:);
P = posterior(gmm_pc_model,obs123)

%% Task 3

% Split the original data into 60% training and
% 40% testing:
cv1 = cvpartition(length(Y2),'holdout',0.4);
X_train = X(training(cv1),:);
X_test = X(test(cv1),:);
Y2_train = Y2(training(cv1),:);
Y2_test = Y2(test(cv1),:);

% Linear regression using the entire training set:
linear_model = fitlm(X_train,Y2_train,'linear');

% Divide the data further:
cv2 = cvpartition(length(Y2_train),'holdout',0.2);
% Model training:
X_train2 = X_train(training(cv2),:);
Y2_train2 = Y2_train(training(cv2),:);
% Model validation:
X_val = X_train(test(cv2),:);
Y2_val = Y2_train(test(cv2),:);

% Generate 100 lambda values (from 10^(-4) to 10^1):
lambda_values_ridge = logspace(-4,1,100);
immse_ridge = zeros(length(lambda_values_ridge),1);
% Train 100 ridge regressions:
for i=1:length(lambda_values_ridge)
    % Train ridge
    b=ridge(Y2_train2,X_train2,lambda_values_ridge(i),0);
    % Predict on validation set
    ypred_ridge=b(1)+X_val*b(2:end);
    % Compute MSE
    immse_ridge(i) = immse(Y2_val,ypred_ridge);
end
% Optimal lambda
[~,best_ridge_idx]=min(immse_ridge);
best_ridge_lambda=lambda_values_ridge(best_ridge_idx)
% Plot lambda (x) and validation MSE (y):
figure;
plot(lambda_values_ridge,immse_ridge)
xlabel('Lambda');
ylabel('Validation MSE');
title('Ridge Regression: Validation MSE vs. Lambda');
% Highlight optimal lambda
hold on;
plot(best_ridge_lambda, immse_ridge(best_ridge_idx), 'ro',...
'MarkerSize', 10,'LineWidth', 2);
hold off;
% Retrain the model on full training
% set using optimal lambda:
b_final_ridge=ridge(Y2_train,X_train,best_ridge_lambda,0);
ypred_final_ridge=b_final_ridge(1)+X_test*b_final_ridge(2:end);
MSE_final_ridge=immse(Y2_test,ypred_final_ridge);

% Generate 100 lambda values (from 10^(-10) to 10^1):
lambda_values_lasso = logspace(-10,1,100);
immse_lasso = zeros(length(lambda_values_lasso),1);
% Train 100 lasso regressions:
for i=1:length(lambda_values_lasso)
    % Train ridge
    [B,fitinfo]=lasso(X_train2,Y2_train2,'Lambda',lambda_values_ridge(i));
    % Predict on validation set
    ypred_lasso=fitinfo.Intercept+X_val*B;
    % Compute MSE
    immse_lasso(i) = immse(Y2_val,ypred_lasso);
end
% Optimal lambda
[~,best_lasso_idx]=min(immse_lasso);
best_lasso_lambda=lambda_values_lasso(best_lasso_idx)
% Plot lambda (x) and validation MSE (y):
figure;
plot(lambda_values_lasso,immse_lasso)
xlabel('Lambda');
ylabel('Validation MSE');
title('Lasso Regression: Validation MSE vs. Lambda');
% Highlight optimal lambda
hold on;
plot(best_lasso_lambda, immse_lasso(best_lasso_idx), 'ro',...
'MarkerSize', 10,'LineWidth', 2);
hold off;
% Retrain the model on full training set using optimal lambda:
[B_final_lasso, fitinfo_final_lasso] = lasso(X_train,Y2_train,...
'Lambda',best_lasso_lambda);
ypred_final_lasso = fitinfo_final_lasso.Intercept + X_test * B_final_lasso;
MSE_final_lasso=immse(Y2_test,ypred_final_lasso);

% Train six linear regressions PCR
[coeff2,score2,~,~,explained2,mu2] = pca(X_train2);
PCR_6_components = score2(:,1:6);
immse_pcr = zeros(6,1);

for i=1:6
    % Train PCR
    Xtrain_pca = PCR_6_components(:,1:i);
    PCR_model = fitlm(Xtrain_pca,Y2_train2);
    % Predict on validation set
    val_score = (X_val-mu2)*coeff2(:,1:i);
    ypred_PCR = predict(PCR_model,val_score);
    % Compute MSE
    immse_pcr(i) = immse(Y2_val,ypred_PCR);  
end
% Optimal number of components
[~,best_n_components] = min(immse_pcr)
% Plot the number of components (x) and validation MSE (y)
figure;
plot(1:6,immse_pcr)
xlabel('Number of Components');
ylabel('Validation MSE');
title('Principal Component Regression: Validation MSE vs. Number of Components');
% Perform PCA on full training set and retrain a linear regression model
[coeff_final,score_final,~,~,explained_final,mu_final]=pca(X_train);
score_train_final = score_final(:,1:best_n_components);
PCR_model_final = fitlm(score_train_final,Y2_train)
% Transform
score_test_final = (X_test - mu_final)*coeff_final(:,1:best_n_components);
ypred_final_pcr = predict(PCR_model_final,score_test_final);
MSE_final_pcr = immse(Y2_test,ypred_final_pcr);

% Model performance based on RMSE
RMSE_linear = linear_model.RMSE
RMSE_ridge = sqrt(MSE_final_ridge)
RMSE_lasso = sqrt(MSE_final_lasso)
RMSE_pcr = sqrt(MSE_final_pcr)
% Best model:
best_RMSE = min([RMSE_linear,RMSE_ridge,RMSE_lasso,RMSE_pcr])
% Coefficient comparison:
reg_names = {'Linear','Ridge','Lasso'}
reg_table = table(...
    [linear_model.Coefficients(2:end,"Estimate")],...
    [b_final_ridge(2:end)],...
    [B_final_lasso],...
    'VariableNames',reg_names)

%% Task 4

% Split target variable Y1:
Y1_train = Y1(training(cv1),:);
Y1_test = Y1(test(cv1),:);

% Apply K-fold cross-validation on the training data (K=4):
cv_kfold = cvpartition(length(Y1_train),'KFold',4);

% Train and cross-validate a model with 50 lambda values:
[B4,fitinfo4] = lassoglm(X_train,Y1_train,'binomial','CV',cv_kfold,'MaxIter',1e5,'NumLambda',50);
% Plot the number of selected predictors as a function of lambda:
lassoPlot(B4,fitinfo4,'PlotType','Lambda','XScale','log');
% Plot the cross-validated deviance as a function of lambda:
lassoPlot(B4,fitinfo4,'plottype','CV');
legend('show')
% Number of selected features using LambdaMinDeviance:
num_coeffs_lambdaMinDev = fitinfo4.DF(fitinfo4.IndexMinDeviance)
% Number of selected features using Lambda1SE:
num_coeffs_lambda1SE = fitinfo4.DF(fitinfo4.Index1SE)
% Retrain the Lasso Model using Lambda1SE:
[B4_final,fitinfo4_final]=lassoglm(X_train,Y1_train,"binomial","Lambda",fitinfo4.Lambda1SE);
% Full coefficient vector:
coef_final_1SE = [fitinfo4_final.Intercept;B4_final];
ypred_lasso_1SE = glmval(coef_final_1SE,X_test,'logit');

% Model 1 with RBF kernel, KernelScale 'auto' and standardized data:
SVM1 = fitcsvm(X_train,Y1_train,'Standardize',true,'KernelFunction',...
    'rbf','KernelScale','auto','CVPartition',cv_kfold);
% Model 2 with polynomial kernel and standardized data:
SVM2 = fitcsvm(X_train,Y1_train,'Standardize',true,'KernelFunction',...
    'polynomial','CVPartition',cv_kfold);
% Accuracy of the third fold for the SVM with RBF kernel:
accuracy3_SVM1 = 1 - kfoldLoss(SVM1,"Mode","individual");
disp(accuracy3_SVM1(3))
% Best-performing SVM model based on cross-validation accuracy:
accuracy_SVM1 = mean(accuracy3_SVM1)
accuracy_SVM2 = 1 - kfoldLoss(SVM2)
% Retrain the SVM model on the full training set using RBF kernel:
final_SVM = fitcsvm(X_train,Y1_train,'Standardize',true,'KernelFunction',...
    'rbf','KernelScale','auto');
ypred_SVM = predict(final_SVM,X_test);

% Generate 10 Min Leaf Size values from 10^1 to 10^2:
leafs = logspace(1,2,10);
% Train a classification tree model for each value of Min Leaf Size:
accuracy_ctree = zeros(10,1);
for n=1:10
    ctree_model = fitctree(X_train,Y1_train,'MinLeafSize',leafs(n),...
        'CVPartition',cv_kfold);
    % Cross-validated accuracy of each model:
    accuracy_ctree(n) = 1 - kfoldLoss(ctree_model);
end
% Optimal Min Leaf Size:
[~,best_ctree_idx] = max(accuracy_ctree);
optimal_leafs = leafs(best_ctree_idx)
% Plot Min Leaf Size (x) vs. cross-validated accuracy (y):
figure;
semilogx(leafs,accuracy_ctree, '-o');
xlabel('Min Leaf Size');
ylabel('Cross-Validates Accuracy');
title('Optimizing Leaf Size');
grid on;
% Retrain the classification tree on the full training set using optimal
% Min Leaf Size:
final_ctree = fitctree(X_train,Y1_train,'MinLeafSize',optimal_leafs);
ypred_ctree = predict(final_ctree,X_test);

% Lasso Logistic Regression:
[xx1SE,yy1SE,thr1SE,auctest_lasso_1SE,optROCpt] = perfcurve(Y1_test,ypred_lasso_1SE,'1')
% SVM:
[~,~,~,auctest_SVM] = perfcurve(Y1_test,ypred_SVM,'1')
% Classification tree:
[~,~,~,auctest_ctree] = perfcurve(Y1_test,ypred_ctree,'1')

% Lasso Logistic Regression:
thr_opt1SE = thr1SE((xx1SE==optROCpt(1))&(yy1SE==optROCpt(2)));
ypred_lasso_1SE_binary = double(ypred_lasso_1SE >= thr_opt1SE);
[confMat_lasso_1SE,~] = confusionmat(Y1_test,ypred_lasso_1SE_binary);
TP_lasso_1SE = confMat_lasso_1SE(2,2);
TN_lasso_1SE = confMat_lasso_1SE(1,1);
FP_lasso_1SE = confMat_lasso_1SE(1,2);
FN_lasso_1SE = confMat_lasso_1SE(2,1);

accuracy_lasso_1SE = (TP_lasso_1SE + TN_lasso_1SE) / (TP_lasso_1SE+TN_lasso_1SE+FP_lasso_1SE+FN_lasso_1SE);
precision_lasso_1SE = TP_lasso_1SE / (TP_lasso_1SE + FP_lasso_1SE);
recall_lasso_1SE = TP_lasso_1SE / (TP_lasso_1SE + FN_lasso_1SE);
f1_lasso_1SE = 2 * (precision_lasso_1SE * recall_lasso_1SE) / (precision_lasso_1SE + recall_lasso_1SE);

% SVM:
[confMat_SVM,~] = confusionmat(Y1_test,ypred_SVM);
TP_SVM = confMat_SVM(2,2);
TN_SVM = confMat_SVM(1,1);
FP_SVM = confMat_SVM(1,2);
FN_SVM = confMat_SVM(2,1);

accuracy_SVM = (TP_SVM + TN_SVM) / (TP_SVM+TN_SVM+FP_SVM+FN_SVM);
precision_SVM = TP_SVM / (TP_SVM + FP_SVM);
recall_SVM = TP_SVM / (TP_SVM + FN_SVM);
f1_SVM = 2 * (precision_SVM * recall_SVM) / (precision_SVM + recall_SVM);

% Classification tree:
[confMat_ctree,~] = confusionmat(Y1_test,ypred_ctree);
TP_ctree = confMat_ctree(2,2);
TN_ctree = confMat_ctree(1,1);
FP_ctree = confMat_ctree(1,2);
FN_ctree = confMat_ctree(2,1);

accuracy_ctree = (TP_ctree + TN_ctree) / (TP_ctree+TN_ctree+FP_ctree+FN_ctree);
precision_ctree = TP_ctree / (TP_ctree + FP_ctree);
recall_ctree = TP_ctree / (TP_ctree + FN_ctree);
f1_ctree = 2 * (precision_ctree * recall_ctree) / (precision_ctree + recall_ctree);

% Summary table of the three models:
model_names = {'Lasso 1SE','SVM (RBF)','Optimal Tree'};
metrics_table = table(...
    [accuracy_lasso_1SE;accuracy_SVM;accuracy_ctree],...
    [precision_lasso_1SE;precision_SVM;precision_ctree],...
    [recall_lasso_1SE;recall_SVM;recall_ctree],...
    [f1_lasso_1SE;f1_SVM;f1_ctree],...
    'VariableNames',{'Accuracy','Precision','Recall','F1Score'},...
    'RowNames',model_names)