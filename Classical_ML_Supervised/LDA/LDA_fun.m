function theta_F = LDA_fun(M_0,Mu_m,C_m,M,out_dim)
data_dim = length(M_0);
S_b = zeros(size(data_dim));
for i = 1:M
    S_b = S_b + (Mu_m(i,:) - M_0)'*(Mu_m(i,:) - M_0);   % Compute between class scatter matrix
end
S_w = sum(C_m,3);                                       % compute within class scatter matrix
[V,D] = eig(pinv(S_w)*S_b);
[~,I] = sort(diag(D),'descend');
V = V(:, I);
theta_F = V(:,1:out_dim);                               % Extract fisher's discriminants