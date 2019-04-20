clc
clear all
%% Load Data 
load('data.mat')
data = reshape(face,[],600)';
data(3:3:end,:) = []; % Throw away the illumination images
test_data = data(1:100,:);
train_data = data(101:end,:);
[train_len,data_dim] = size(train_data);
[test_len,~] = size(test_data);
y_train = ones(train_len,1);
y_train(2:2:end,:) = -1;
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
%% Kernel SVM
C = 100;
d = 1; % 1
K = (train_data*train_data' + 1).^d; % compute the polynomial Kernel
H = K.*(y_train*y_train');
f = -ones(size(y_train));

% Solve the dual problem
alpha = quadprog(H,f,[],[],y_train',0,zeros(size(y_train)),C*ones(size(y_train)));

% Compute the Bias 
temp1 = zeros(train_len/2,1);
for i = 0:train_len/2 -1
    for j = 1:train_len
        temp1(i+1) = temp1(i+1) + alpha(j)*y_train(j)*(train_data(j,:)*train_data(i*2+1,:)' + 1)^d;
    end
end

min_pos_labes = min(temp1);

temp2 = zeros(train_len/2,1);
for i = 0:train_len/2 -1
    for j = 1:train_len
        temp2(i+1) = temp2(i+1) + alpha(j)*y_train(j)*(train_data(j,:)*train_data(i*2+2,:)' + 1)^d;
    end
end
   
max_neg_labes = max(temp2);

theta_0 = - 0.5 * (min_pos_labes + max_neg_labes);

% Compute testing accuracy
pred= zeros(train_len,1);
for j = 1:train_len
    for i = 1:train_len
        pred(j) = pred(j) + alpha(i)*y_train(i)*(train_data(i,:)*train_data(j,:)' + 1)^d;
    end
    pred(j) = pred(j) + theta_0;
end
pred_per = sum(sign(pred.*y_train)>0)/train_len;
disp('The training accuracy for RBF Kernel SVM is ');
disp(pred_per);

% Compute testing accuracy
pred_test = zeros(test_len,1);
for j = 1:test_len
    for i = 1:train_len
        pred_test(j) = pred_test(j) + alpha(i)*y_train(i)*(train_data(i,:)*test_data(j,:)' + 1)^d;
    end
    pred_test(j) = pred_test(j) + theta_0;
end

pred_test_per = sum(sign(pred_test.*y_test)>0)/test_len;
disp('The training accuracy for RBF Kernel SVM is ');
disp(pred_test_per);

