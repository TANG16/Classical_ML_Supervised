function [train_accuracy,test_accuracy] = Bayes_Classisifer_fun(Mu,C,train_data,test_data,y_train,y_test,M)
[train_len,~] = size(train_data);
[test_len,~] = size(test_data);
%% Training Accuracy
P = zeros(M,1);
cnt_train = 0;
for idx = 1:train_len
    for k = 1:M
        P(k) = (1/sqrt(det(C(:,:,k))))*exp(-0.5*((train_data(idx,:)-Mu(k,:))*inv(C(:,:,k))*(train_data(idx,:)-Mu(k,:))'));  % Compute the posteriors
    end
    [~,b] = max(P);   % assign to the maximum posterior 
    if b == y_train(idx)
        cnt_train = cnt_train + 1;
    end    
end
train_accuracy = cnt_train/train_len;
%% Testing Error
P = zeros(1,200);
cnt_test = 0;
for idx = 1:test_len
    for k = 1:M
        P(k) = (1/sqrt(det(C(:,:,k))))*exp(-0.5*((test_data(idx,:)-Mu(k,:))*inv(C(:,:,k))*(test_data(idx,:)-Mu(k,:))'));
    end
    [~,b] = max(P);
    if b == y_test(idx) %ceil(idx/2)
        cnt_test = cnt_test + 1;
    end
end
test_accuracy = cnt_test/test_len;