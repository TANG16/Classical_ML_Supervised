clc
clear all
%% Load Data 
load('data.mat')
data = reshape(face,[],600)';
% data = zscore(data);  % Standardize the data to have 0 mean and std 1
data = data - mean(data,1);

%% PCA
PCA_data = PCA_fun(data);
%% Test train split
test_data = PCA_data(2:3:end,:);
train_data = PCA_data;
train_data(2:3:end,:) = [];
[train_len,data_dim] = size(train_data);
[test_len,~] = size(test_data);
y_test = (1:200)';
y_train = repelem(y_test,2); 
M = 200; % Number of classes
%% Compute ML Estimates
% temp = zeros(1,100);
Mu = zeros(M,data_dim);
C = zeros(data_dim,data_dim,M);
for j = 0:M-1
    Mu(j+1,:) = mean(train_data(j*2+1:j*2+2,:),1);
    C(:,:,j+1) = cov([train_data(j*2+1,:);train_data(j*2+2,:)]);
end
C = C + eye(data_dim,data_dim); % Add identity matrix to make Covariance Matrices Invertible
%% Bayes Classifier
[train_accuracy,test_accuracy] = Bayes_Classisifer_fun(Mu,C,train_data,test_data,y_train,y_test,M);

disp('The training accuracy for Bayes is ');
disp(train_accuracy);

disp('The testing accuracy for Bayes is ');
disp(test_accuracy);

