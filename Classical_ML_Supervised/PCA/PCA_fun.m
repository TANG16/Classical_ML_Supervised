function PCA_data = PCA_fun(data)
E = cov(data);
[V,D] = eig(E);
[D,I] = sort(diag(D),'descend');
V = V(:, I);
for i = 1:length(diag(D))
    tol = sum(D(1:i))/sum(D);
    if tol > .95                    % take eig vectors such that 95% of data is preserved
        break;
    end
end
PCA_data = data*V(:,1:i);

