clc
clear all
%% Load Data 
load('data.mat')
data = reshape(face,[],600)';
data = zscore(data);  % Standardize the data to have 0 mean and std 1
% data = data - mean(data,1);

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
%% K - NN classifier
K = 1;

accuracy = KNN_fun(test_data,train_data,y_test,y_train,K);
disp('The testing accuracy for k-nn is ');
disp(accuracy);

