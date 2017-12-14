%% -------------  Load the data
X_samples = x_test;
Y_samples = y_test;

% -------------  Gram matrices
K = gaussianGram(X_samples);
L = gaussianGram(Y_samples);

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

% a and b unnormalised parts of biggest eigenvector; alpha and beta normalised
a = V(1:n,max_gen_eig_index);
b = V(n+1:end,max_gen_eig_index);
alpha = a ./ (a'*(Kt*Kt + normalising_constant*Kt)*a);
beta = b ./ (b'*(Kt*Kt + normalising_constant*Kt)*b);

%%----------- Plotting
figure; f_y = @(y) plot_dual_gaussian(y,beta,Y_samples);
%f_y_samples = arrayfun(f_y, Y_samples);
points = linspace(200,350); % then plot mapping function on top
f_y_points = arrayfun(f_y, points);
plot(points,f_y_points);
title('Plot of largest kernel canonical projection g for data Y')
xlabel('y')
ylabel('g(y)')

% -------------- Functions

function g = plot_dual_gaussian(y, dual, samples)
  n = numel(samples);
  k_yjs_y = @(yjs, y) exp(-((yjs-y).^2)/50);
  k_samples_y = k_yjs_y(samples,y);
  g = (k_samples_y - sum(k_samples_y)/n)'*dual;
end

function kxy = gaussian_kernel(diff)
  eta = 5;
  kxy = exp(-(norm(diff)^2) / (2*eta^2));
end

function K = gaussianGram(X)
                                % Gram matrix for Gaussian kernel
  n = length(X);
  K = zeros(n);
  for i = 1:n
    for j = 1:n
      K(i,j) = gaussian_kernel(X(i,:) - X(j,:));
    end
  end
end





function [H]=GaussKern(x,y,sig)
  H=sqdistance(x',y');
  H=exp(-H/2/sig^2);
end



function d = sqdistance(a,b)
  if (nargin ~= 2)
    error('Not enough input arguments');
  end
  if (size(a,1) ~= size(b,1))
    error('Inputs must have same dimensionality');
  end
  aa=sum(a.*a,1); bb=sum(b.*b,1); ab=a'*b; 
  d = abs(repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab);
end
