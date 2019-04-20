clc
clear all
%% Load Data 
load('data.mat')
data = reshape(face,[],600)';
% data = zscore(data);  % Standardize the data to have 0 mean and std 1
data(3:3:end,:) = []; % Throw away the illumination images
data = data - mean(data,1);
data = PCA_fun(data);
test_data = data(1:100,:);
train_data = data(101:end,:);
[train_len,data_dim] = size(train_data);
[test_len,~] = size(test_data);
y_train = ones(train_len,1);
y_train(2:2:end,:) = 2;
y_test = y_train(1:test_len);
M = 2;
%% LDA
out_dim = 1;
M_0 = mean(train_data,1);
Mu_m = zeros(M,data_dim);
C_m = zeros(data_dim,data_dim,M);
for j = 1:M
    Mu_m(j,:) = mean(train_data(j:2:end,:),1);
    C_m(:,:,j) = cov(train_data(j:2:end,:));
end
theta_F = LDA_fun(M_0,Mu_m,C_m,M,out_dim);
train_data = (theta_F'*train_data')';
test_data = (theta_F'*test_data')';
[train_len,data_dim] = size(train_data);
[test_len,~] = size(test_data);
%% K - Nearest Neighbour Classifier
N = 1;
accuracy = KNN_fun(test_data,train_data,y_test,y_train,N);
disp('The testing accuracy for k-nn is ');
disp(accuracy);