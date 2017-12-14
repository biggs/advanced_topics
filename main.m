%% -------------  Load the data
load x_test.mat
load y_test.mat
subset_n = 300; % how many to use
X_samples = x(1:subset_n,:);
Y_samples = y(1:subset_n,:);
n = size(Y_samples,1);

% -------------  Gram matrices
eta = 5;
K = gaussianGram(X_samples, eta);
L = gaussianGram(Y_samples, eta);

% -------------  Centring
H = eye(n) - ones(n)/n;
Kt = H*K*H;  % centred kernel K tilde
Lt = H*L*H;  %centred kernel L tilde

% -------------  Kernel CCA block matrices
LH_block = [zeros(n) Kt*Lt/n; Lt*Kt/n zeros(n)];
normalising_constant = 0.1;
RH_block_cca = blkdiag(Kt^2,Lt^2) + normalising_constant*blkdiag(Kt,Lt);


%% --------------- CCA
fprintf('\nCCA\n----------\n')

[V, d] = eig(pinv(RH_block_cca)*LH_block, 'vector');
[max_gen_eig, max_gen_eig_index] = max(d);
fprintf("Largest eigenvalue: %f\n", max_gen_eig)

% a and b unnormalised parts of biggest eigenvector; alpha and beta normalised
a = V(1:n,max_gen_eig_index);
b = V(n+1:end,max_gen_eig_index);
alpha = a ./ (a'*(Kt*Kt + normalising_constant*Kt)*a);
beta = b ./ (b'*(Lt*Lt + normalising_constant*Lt)*b);


%% ----------- Projections
g_y = @(y) gaussian_kernel_projection(y,beta,Y_samples,eta);
g_y_samples = arrayfun(f_y, Y_samples);

f_x = @(x) gaussian_kernel_projection(x,alpha,X_samples,eta);
f_x_samples = zeros(n,1);
for i = 1:n
  f_x_samples(i,1) = f_x(X_samples(i,:));
end

% ------------ Plots
figure; scatter(Y_samples,g_y_samples)
title('Plot of largest kernel canonical projection g for data Y')
xlabel('y')
ylabel('g(y)')

figure; scatter(f_x_samples,f_y_samples)

% CCA correlation value
corr(f_x_samples, g_y_samples)



% -------------- Functions

function K = gaussianGram(X, eta)
  N = length(X);
  K = zeros(N);
  for i = 1:N
    for j = 1:N
      K(i,j) = norm(X(i,:) - X(j,:));
    end
  end
  K = exp(- K.^2 /2/eta^2);
end


