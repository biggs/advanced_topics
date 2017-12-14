function f = gaussian_kernel_projection(x, dual, samples, eta)
  % samples: N x D
  % x: 1 x D
  % dual: M x 1
  N = size(samples,1);
  k_xj_x = @(xj, x) exp(-(norm(xj-x).^2)/2/eta^2); % kernel between a sample and x
  k_samples_x = zeros(N,1);
  for i = 1:N
    k_samples_x(i,1) = k_xj_x(samples(i,:), x);
  end
  f = (k_samples_x - sum(k_samples_x)/N)'*dual;
end
