clc
clear all
%% Load Data 
load('data.mat')
data = reshape(face,[],600)';
% data = zscore(data);  % Standardize the data to have 0 mean and std 1
data(3:3:end,:) = []; % Throw away the illumination images
% data = data - mean(data,1);
data = zscore(data);
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
%% K-Nearest Neighbour Classifier
K = 5;
accuracy = KNN_fun(test_data,train_data,y_test,y_train,K);
disp('The testing accuracy for k-nn is ');
disp(accuracy);