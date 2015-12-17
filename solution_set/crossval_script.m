clear all; close all; clc;
%% Load Data & Set Params
fn = SpaRSA_Models();
load('fMRIGroup.mat'); % provides dataset called "dataset"

testing_function = 'group_lasso'; % 'group_lasso' or 'enet'

lambda_low = -2;
lambda_high = 2;
num_lambdas = 10;
num_alphas = 5;
num_outer = 10;
num_inner = num_outer-1;

subject = dataset(2);
labels = subject.y;
data = subject.X;
grps = subject.roi;
% The groups are not the same size as A; let's fix that putting all the
% stray voxels in their own group
num_grp = size(grps,1);
num_grpd = size(grps,2);
num_feat = size(data,2);
grps = [grps zeros(num_grp, num_feat - num_grpd); zeros(1, num_feat)];
grps(end, sum(grps)==0) = grps(end, sum(grps)==0) + 1;

%% Cross Validation loop
% Ian's function will take inputs of train and test data
if strcmp(testing_function, 'group_lasso'),
    num_alphas = 1;
end
CVouter = cvpartition(labels, 'k', num_outer);
err = zeros(CVouter.NumTestSets, num_alphas);
nnz_coef = zeros(CVouter.NumTestSets, num_alphas);

for i = 1:CVouter.NumTestSets
    disp('****')
    disp(i);
    Errors_ = zeros(num_alphas, num_lambdas, 5);
    alpha_vals = linspace(0,1,num_alphas);
    lambda_vals = logspace(lambda_high, lambda_low, num_lambdas);
    
    trainIdx = CVouter.training(i);
    holdIdx = CVouter.test(i);
    
    % Separate test and training sets for the 10 fold
    % Xtr_out == X_train_out
    Xtr_outer = data(trainIdx,:);
    ytr_outer = labels(trainIdx);
    Xtest_outer = data(holdIdx,:);
    ytest_outer = labels(holdIdx);
    
    CVinner = cvpartition(labels(trainIdx), 'k', num_inner);
    for j = 1:CVinner.NumTestSets
        % Separate test and training sets for the 9 fold
        X_train = Xtr_outer(CVinner.training(j),:);
        y_train = ytr_outer(CVinner.training(j));
        X_test = Xtr_outer(CVinner.test(j),:);
        y_test = ytr_outer(CVinner.test(j));
        
        % Do a grid search over all alphas and all lambdas
        i2 = 0;
        w_hat = zeros(size(X_train, 2), 1);
        for alpha = alpha_vals
            i2 = i2 + 1;
            j2 = 0;
            for lam = lambda_vals
                j2 = j2 + 1;
                if strcmp(testing_function, 'enet'),
                    [error_, w_hat, ~] = fn.Elastic_Net(X_train, y_train, X_test, ...
                        y_test, lam, alpha, w_hat);
                end
                if strcmp(testing_function, 'group_lasso'),
                    [error_, w_hat, ~] = fn.Group_LASSO(X_train, y_train, X_test, ...
                        y_test, lam, grps, w_hat);
                end
                %disp(error_)
                non_zeros = sum(abs(w_hat) > 1e-6);
                Errors_(i2, j2, 1:4) = Errors_(i2, j2, 1:4) + ...
                    reshape([1.0*error_, non_zeros, lam, alpha], [1,1,4])...
                    / CVinner.NumTestSets;
            end
        end
        j
    end
    % pick best lambda for each alpha
    best_lambdas_inds = zeros(num_alphas,1);  % vector of indexs of best lambda
    best_Errors_ = zeros(num_alphas,1);
    for al = 1:size(Errors_,1)
        % find largest lambda w/in 10% of minimum
        Errors_(al,:,5) = (Errors_(al,:,1) - 1.1*min(Errors_(al,:,1))) <= 1e-6;
        best_lambdas_inds(al) = find(Errors_(al,:,5) == 1, 1, 'first');
        best_Errors_(al) = Errors_(al, best_lambdas_inds(al),1);
        % Retrain with best lambda, get Final Fit
        w_hat = zeros(size(X_train, 2), 1);
        if strcmp(testing_function, 'enet'),
            [err(i, al), w_hat, ~] = fn.Elastic_Net(Xtr_outer, ytr_outer,...
                Xtest_outer, ytest_outer, lambda_vals(best_lambdas_inds(al)),...
                alpha_vals(al), w_hat);
        end
        if strcmp(testing_function, 'group_lasso'),
            [err(i, al), w_hat, ~] = fn.Group_LASSO(Xtr_outer, ytr_outer,...
                Xtest_outer, ytest_outer, lambda_vals(best_lambdas_inds(al)),...
                grps, w_hat);
        end
        nnz_coef(i, al) =  sum(abs(w_hat) > 1e-6);
    end
end

% calculate average error:
err_avg = mean(err);
nnz_avg = mean(nnz_coef);

%% Meaningful Results: E-Net Sweep Also Optimizing Over Lambda
% Important: Did this when summing over alphas... notice best prediction
% accuracies align with folds that has the sparsest solutions... this tells
% us something about our signal: mainly it is sparse
% nnz_coef
%
% nnz_coef =
%
%         3546
%          618
%         2455
%          194
%         1754
%         3191
%          924
%         2537
%          644
%         2601
%
% err
%
% err =
%
%     0.1250
%          0
%     0.1250
%          0
%     0.1250
%     0.1250
%          0
%     0.3750
%          0
%     0.5000
%
%% Meaningful Results: E-Net Sweep W/ Separate Lambdas
% The important thing here is that all the non-zero alphas performed better
% than the alpha=0 (ridge) we can explain why, because we want a sparse
% solution. Also, it is important to point out the avg nnz weights obtained
% by the E-net (more weights with the same pred accuracy) which is
% preferrable for robustness/interpretability with low alpha as we can
% capture more correlated voxels...emphasize that alpha=1 (LASSO) had the
% sparsest solution.
%
% err_avg
%
% err_avg =
%
%     0.2000    0.1250    0.1250    0.1250    0.1250
%
% nnz_avg
%
% nnz_avg =
%
%    1.0e+03 *
%
%     4.9470    2.3456    1.6535    1.5768    1.4289
%
%% Group Lasso
% Important to note that It did not do as well this is due to the fact that
% it selected groups as all or none. The fact that it did not do well could
% lead us to believe one or more of these three things:
% things:
%
% 1.) The Group Structured signal was not actually present in the groups we
% defined. Maybe if we had a different way of detecting group (overlapping
% group lasso for example) we could do better.
%
% 2.) The Signal within the groups was sparse meaning that we would want to
% use something like the sparse group lasso which deals better with these
% types of situations.
%
% 3.) The Signal isn't even grouped so we are shooting ourselves in the
% foot by using the group lasso.
%
% err
% 
% err =
% 
%     0.1250
%     0.2500
%     0.1250
%     0.2500
%          0
%     0.2500
%          0
%     0.2500
%     0.2500
%     0.1250
% 
% err_avg
% 
% err_avg =
% 
%     0.1625
% nnz_coef
% 
% nnz_coef =
% 
%         4949
%         4947
%         4949
%         4947
%         4946
%         4946
%         4801
%         4947
%         4945
%         4948
%         
%         nnz_avg
% 
% nnz_avg =
% 
%    4.9325e+03