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
fprintf("Largest eigenvalue: %f\n",max_gen_eig)

% a and b unnormalised parts of biggest eigenvector; alpha and beta normalised
a = V(1:n,max_gen_eig_index);
b = V(n+1:end,max_gen_eig_index);
alpha = a ./ (a'*(Kt*Kt + normalising_constant*Kt)*a);
beta = b ./ (b'*(Lt*Lt + normalising_constant*Lt)*b);


%%----------- Plotting
figure; f_y = @(y) plot_dual_gaussian(y,beta,Y_samples,eta);
f_y_samples = arrayfun(f_y, Y_samples);
scatter(Y_samples,f_y_samples)
title('Plot of largest kernel canonical projection g for data Y')
xlabel('y')
ylabel('g(y)')




% -------------- Functions

function g = plot_dual_gaussian(y, dual, samples, eta)
  n = numel(samples);
  k_yjs_y = @(yjs, y) exp(-((yjs-y).^2)/2/eta);
  k_samples_y = k_yjs_y(samples,y);
  g = (k_samples_y - sum(k_samples_y)/n)'*dual;
end

function kxy = gaussian_kernel(diff,eta)
  kxy = exp(-(norm(diff)^2)/2/eta^2);
end

function K = gaussianGram(X, eta)
  n = length(X);
  K = zeros(n);
  for i = 1:n
    for j = 1:n
      K(i,j) = gaussian_kernel(X(i,:) - X(j,:), eta);
    end
  end
end


