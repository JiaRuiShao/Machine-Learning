function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

n = length(theta);
J = (1/m * (-y.' * log(sigmoid(X*theta)) - (1-y).' * log(1-sigmoid(X*theta)))) + ((lambda/(2*m)) * (theta([2:n],1).' * theta([2:n],1)));
% OR THE LATTER PART coule write: 
% (lambda/(2*m)) * (theta([2:n],:).' * theta([2:n],:))

% Matlab index starts from 1 rather than 0 and here theta 0 is excluded

theta_zero = theta;
theta_zero(1) = 0;
grad = (1/m * (((sigmoid(X*theta)-y).' * X).')) + ((lambda/m) * theta_zero);

% grad(1,:) = 1/m * (((sigmoid(X(:,1)*theta(1,:))-y).' * X(:,1)).');
% grad([2:n],:) = (1/m * (((sigmoid(X(:,[2:n])*theta([2:n],:))-y).' * X(:,[2:n])).')) + ((lambda/m) * theta([2:n],:));

% =============================================================

end
