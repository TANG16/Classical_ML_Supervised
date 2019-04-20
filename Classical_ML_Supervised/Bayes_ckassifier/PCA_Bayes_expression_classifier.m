clc
clear all
%% Load Data 
load('data.mat')
data = reshape(face,[],600)';
data(3:3:end,:) = []; % Throw away the illumination images
data = data - mean(data,1);
% data = zscore(data);  % Standardize the data to have 0 mean and std 1
%% PCA
PCA_data = PCA_fun(data);
test_data = PCA_data(1:100,:);
train_data = PCA_data(101:end,:);
[train_len,data_dim] = size(train_data);
[test_len,~] = size(test_data);
y_train = ones(train_len,1);
y_train(2:2:end,:) = 2;
y_test = y_train(1:test_len);
M = 2;
%% Bayes Classifier
Mu = zeros(M,data_dim);
C = zeros(data_dim,data_dim,M);
for j = 1:M
    Mu(j,:) = mean(train_data(j:2:end,:),1);
    C(:,:,j) = cov(train_data(j:2:end,:));
end
C = C + eye(data_dim,data_dim);
[train_accuracy,test_accuracy] = Bayes_Classisifer_fun(Mu,C,train_data,test_data,y_train,y_test,M);
disp('The training accuracy for Bayes is ');
disp(train_accuracy);
disp('The testing accuracy for Bayes is ');
disp(test_accuracy);