
load('fMRIGroup.mat'); % provides dataset called "dataset"
fn = SpaRSA_Models();

%% Question 2: Implement Elastic Net


patient = dataset(2);
y = patient.y;
X = dataset.X;

[~, w_hat, ~] = fn.Elastic_Net(X, y, X, y, 30, 0.1);

figure; hold on
plot(w_hat)

