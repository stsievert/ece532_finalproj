fn = SpaRSA_Models();
load('fMRIGroup.mat'); % provides dataset called "dataset"

testing_function = 'group_lasso'; % 'group_lasso' or 'enet'

lambda_low = -1;
lambda_high = 2;
num_lambdas = 8;
num_alphas = 8;
num_outer = 8;
num_inner = num_outer-1;

patient = dataset(2);
labels = patient.y;
data = dataset.X;
grps = dataset.roi;
% The groups are not the same size as A; let's fix that by padding on zeros
grps = [grps zeros(25, 4698-4402)];

%% Cross Validation loop
% Ian's function will take inputs of train and test data
CVouter = cvpartition(size(data,1), 'k', num_outer);
err = zeros(CVouter.NumTestSets,1);

avg_errors = zeros(num_lambdas, num_alphas, 4);
for i = 1:CVouter.NumTestSets
    disp('****')
    disp(i);
    Errors_ = 0 * avg_errors;
    alpha_vals = linspace(0,1,num_alphas);
    lambda_vals = logspace(lambda_low, lambda_high, num_lambdas);

    trainIdx = CVouter.training(i);
    holdIdx = CVouter.test(i);

    % Separate test and training sets for the 10 fold
    % Xtr_out == X_train_out
    Xtr_outer = data(trainIdx,:);
    ytr_outer = labels(trainIdx);
    Xtest_outer = data(holdIdx,:);
    ytest_outer = labels(holdIdx);

    CVinner = cvpartition(size(Xtr_outer,1), 'k', num_inner);
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
                disp(error_)
                non_zeros = sum(abs(w_hat) > 1e-6);
                Errors_(i2, j2, :) = [1.0*error_, non_zeros, lam, alpha];
            end
        end
    avg_errors = avg_errors + Errors_;
    end
end

avg_errors = avg_errors / (CVouter.NumTestSets * CVinner.NumTestSets);

% Outputs this matrix:
% avg_errors(:, :, 1) is the prediction accuracy for that alpha and lambda
% avg_errors(:, :, 2) is the number of non-zeros for that alpha and lambda
% avg_errors(:, :, 3) is the lambda value for that error
% avg_errors(:, :, 4) is the alpha value for that error

%% ELASTIC NET errors
% >> avg_errors
%
% avg_errors(:,:,1) =
%
%     0.4750    0.2179    0.1464    0.1411    0.1411    0.1411    0.1411    0.1411
%     1.0000    0.3536    0.1429    0.0964    0.0911    0.0839    0.0839    0.0857
%     1.0000    0.6536    0.1875    0.0982    0.0857    0.0786    0.0786    0.0786
%     1.0000    0.9804    0.2714    0.1054    0.0875    0.0839    0.0804    0.0804
%     1.0000    1.0000    0.3554    0.1089    0.0857    0.0839    0.0839    0.0839
%     1.0000    1.0000    0.3589    0.1196    0.0857    0.0857    0.0839    0.0839
%     1.0000    1.0000    0.4786    0.1339    0.0893    0.0875    0.0875    0.0875
%     1.0000    1.0000    0.5768    0.1464    0.0929    0.0857    0.0839    0.0875
%
%
% avg_errors(:,:,2) =
%
%    1.0e+03 *
%
%          0    4.6928    4.6957    4.6955    4.6955    4.6954    4.6956    4.6959
%          0         0    0.1820    0.3344    0.6605    1.3766    2.0399    2.6831
%          0         0    0.0530    0.1166    0.3620    0.7716    1.4319    2.2664
%          0         0    0.0243    0.0571    0.2065    0.5560    1.1715    2.0058
%          0         0    0.0109    0.0343    0.1393    0.4030    0.9358    1.6787
%          0         0    0.0042    0.0260    0.0939    0.3485    0.7258    1.4849
%          0         0    0.0014    0.0200    0.0727    0.2772    0.7554    1.3991
%          0         0    0.0008    0.0152    0.0616    0.2736    0.6683    1.2017
%
%
% avg_errors(:,:,3) =
%
%    1.0e+03 *
%
%     1.0000    0.2683    0.0720    0.0193    0.0052    0.0014    0.0004    0.0001
%     1.0000    0.2683    0.0720    0.0193    0.0052    0.0014    0.0004    0.0001
%     1.0000    0.2683    0.0720    0.0193    0.0052    0.0014    0.0004    0.0001
%     1.0000    0.2683    0.0720    0.0193    0.0052    0.0014    0.0004    0.0001
%     1.0000    0.2683    0.0720    0.0193    0.0052    0.0014    0.0004    0.0001
%     1.0000    0.2683    0.0720    0.0193    0.0052    0.0014    0.0004    0.0001
%     1.0000    0.2683    0.0720    0.0193    0.0052    0.0014    0.0004    0.0001
%     1.0000    0.2683    0.0720    0.0193    0.0052    0.0014    0.0004    0.0001
%
%
% avg_errors(:,:,4) =
%
%          0         0         0         0         0         0         0         0
%     0.1429    0.1429    0.1429    0.1429    0.1429    0.1429    0.1429    0.1429
%     0.2857    0.2857    0.2857    0.2857    0.2857    0.2857    0.2857    0.2857
%     0.4286    0.4286    0.4286    0.4286    0.4286    0.4286    0.4286    0.4286
%     0.5714    0.5714    0.5714    0.5714    0.5714    0.5714    0.5714    0.5714
%     0.7143    0.7143    0.7143    0.7143    0.7143    0.7143    0.7143    0.7143
%     0.8571    0.8571    0.8571    0.8571    0.8571    0.8571    0.8571    0.8571
%     1.0000    1.0000    1.0000    1.0000    1.0000    1.0000    1.0000    1.0000
%
% GROUP LASSO
% > avg_errors
%
% avg_errors(:,:,1) =
%
%     1.0000    1.0000    0.3882
%     0.3882    0.3882    0.3882
%     0.3882    0.3882    0.3882
%
%
% avg_errors(:,:,2) =
%
%    1.0e+03 *
%
%          0         0    4.3988
%     4.3988    4.3988    4.3990
%     4.3990    4.3990    4.3993
%
%
% avg_errors(:,:,3) =
%
%    1.0e+03 *
%
%     1.0000    0.0100    0.0001
%     1.0000    0.0100    0.0001
%     1.0000    0.0100    0.0001
%
%
% avg_errors(:,:,4) =
%
%          0         0         0
%     0.5000    0.5000    0.5000
%     1.0000    1.0000    1.0000
