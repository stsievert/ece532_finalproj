lambda_low = 1e-6;
lambda_high = 1;
num_lambdas = 10;
num_alphas = 11;

%% Data is input here
% lower loop takes matrix called data

%% Cross Validation loop
% Ian's function will take inputs of train and test data
CVouter = cvpartition(size(data,1), 'k', 10);
err = zeros(CVouter.numTestSets,1);

for i = 1:CVouter.numTestSets
    MSEs = zeros(num_lambdas,num_alphas);
    alpha_vals = linspace(0,1,num_alphas);
    lambda_vals = logspace(lambda_low, lambda_high, num_lambdas);
    
    trainIdx = CVouter.training(i);
    holdIdx = CVouter.test(i);
    
    % Separate test and training sets for the 10 fold
    Xtr_outer = data(trainIdx,:);
    ytr_outer = labels(trainIdx);
    Xtest_outer = data(holdIdx,:);
    ytest_outer = labels(holdIdx);
     
    CVinner = cvpartition(size(Xtr_outer,1), 'k', 9);
    for j = 1:CVinner.numTestSets
        % Separate test and training sets for the 9 fold
        X_train = Xtr_outer(CVinner.training(j),:);
        y_train = ytr_outer(CVinner.training(j));
        X_test = Xtr_outer(CVinner.test(j),:);
        y_test = ytr_outer(CVinner.test(j));
        
        % Do a grid search over all alphas and all lambdas
        for alpha = alpha_vals
            for lam = lambda_vals
                % Ian: put your code here.
                % outputs the MSE
                MSEs(lam, alpha) = inf; % put the actual MSE value here
            end
        end
        
        % pick best lambda for each alpha
        best_lambdas_inds = zeros(num_alphas,1);  % vector of indexs of best lambda
        best_MSEs = zeros(num_alphas,1);
        for al = 1:size(MSEs,2)
            % find largest lambda w/in 10% of minimum
            MSEs(:,al) = (MSEs(:,al) - 1.1*min(MSEs(:,al))) <= 1e-6;
            best_lambdas_inds(al) = find(MSEs(:,al) == 1, 1, 'last');
            best_MSEs(al) = MSEs(best_lambdas_inds(al), al);
        end
        
        [min_mse, idx] = min(best_MSEs);
        best_lambda_val = lambda_vals(best_lambdas_inds(idx));
        best_alpha_val = alpha_vals(idx);  
    end
    
    % Ian retrains with best lambda and alpha values, gets MSE_final
    err(i) = MSE_final;
end

% calculate average error:
err_avg = mean(err);
