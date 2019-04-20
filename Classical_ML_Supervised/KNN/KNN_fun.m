function accuracy = KNN_fun(test_data,train_data,y_test,y_train,K)
[train_len,~] = size(train_data);
[test_len,~] = size(test_data);
distance = zeros(1,train_len);
cnt = 0;
for k = 1:test_len
    for i = 1:train_len
        distance(i) = norm(train_data(i,:)-test_data(k,:));
    end
    [~,b] = mink(distance,K);  % Compute k nearest neighbours
    b = y_train(b);            % find labels of the neighbors
    [m,f] = mode(b);
    if f == 1                  % if no repeated labels assign to the closest neighbour
        class_assign = b(1);
    else
        class_assign = m;      % else assign to the class of most repeated neighbour
    end
    if class_assign == y_test(k)  % Compute accuracy
        cnt = cnt + 1;         
    end
end

accuracy = cnt/test_len;
