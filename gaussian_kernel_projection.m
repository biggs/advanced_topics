function g = gaussian_kernel_projection(y, dual, samples, eta)
  n = numel(samples);
  k_yjs_y = @(yjs, y) exp(-((yjs-y).^2)/2/eta);
  k_samples_y = k_yjs_y(samples,y);
  g = (k_samples_y - sum(k_samples_y)/n)'*dual;
end

