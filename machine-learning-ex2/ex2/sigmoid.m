function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).





% =============================================================
[nr,nc] = size(z);
for i = 1:nr
  for j = 1:nc
     g(i,j) = (1 + e**(-z(i,j)))**(-1);
  end
end
end
