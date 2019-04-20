clc
clear all
%% Load Data 
load('data.mat')
data = reshape(face,[],600)';
data(3:3:end,:) = []; % Throw away the illumination images
test_data = data(1:80,:);
train_data = data(81:end,:);
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
%% Boosting Algorithm
% Initialization
C = 100;
wts = (1/length(y_train))*ones(size(y_train));
a = [];  % Vecotor for storing a_i's
W = [];  % Matrix for storing the weights
B = [];  % Vector for storing Biases
% Iterations
for i = 1:100
    close all
    % Train the classifier Using Linear SVM
    Wts = wts';
    cvx_begin
        variables e(size(y_train)) w(d) b(1)
        minimize C*Wts*e + norm(w)^2
        subject to 
            e >= 0
            e >= One - y_train.*(train_data*w + b)
    cvx_end
    
    W(i,:) = w;
    B(i) = b;
    
    % Compute the Error
    pred = train_data*W(i,:)'+B(i);
    error = sum((sign(pred.*y_train)<0).*wts);
    
    % Compute a_i
    a(i) = 0.5*log((1-error)/error);
    
    % Update Weights
    wts = wts.*exp(-a(i).*(pred.*y_train));
    wts = wts/sum(wts);
    
    % Compute predictions on training data
    PRED = zeros(size(y_train));
    for j = 1:i
        for k = 1:length(y_train)
            PRED(k) = PRED(k) + a(j).*(train_data(k,:)*W(j,:)' + B(j));
        end
    end

    % Compute predictions on training data
    test_pred = zeros(test_len,1);
    for j = 1:i
        for k = 1:test_len
            test_pred(k) = test_pred(k) + a(j).*(test_data(k,:)*W(j,:)' + B(j));
        end
    end
    
    % Exit condition:
    train_misclass =  sum((sign(PRED.*y_train)<0));
    test_missclass = sum((sign(test_pred.*y_train(1:test_len))<0));
    if(test_missclass < 5) || (train_misclass == 0)
        disp(' Exit condition achieved! \n Exiting');
        break;
    end
     

    disp('The training accuracy for Boosted SVM in this iteration is ');
    disp((train_len-train_misclass)/train_len);

    disp('The testing accuracy for Boosted SVM in this iteration is ');
    disp((test_len-test_missclass)/test_len);
    
    pause(2)

    
end

disp('The training accuracy for Boosted SVM is ');
disp((train_len-train_misclass)/train_len);

disp('The testing accuracy for Boosted SVM is ');
disp((test_len-test_missclass)/test_len);
