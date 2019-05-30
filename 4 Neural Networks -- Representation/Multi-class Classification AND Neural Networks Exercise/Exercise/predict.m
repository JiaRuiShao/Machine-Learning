function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1); % dim is (m,1)

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

a1 = [ones(m, 1) X]; 
% dim of a1 is (m, # of 1st layer units/example features + 1), here is (5000,401)

a2 = sigmoid(a1 * Theta1.'); 
% dim of a2 is (m, # of 2nd layer units), here is (5000,25)

a2_2 = [ones(m, 1) a2]; 
% dim of a2_2 is (m, # of 2nd layer units + 1), here is (5000,26)

a3 = sigmoid(a2_2 * Theta2.') 
% dim of a3 is (m, # of 3nd layer units/output layer here), here is (5000,10)

[Y,I] = max(a3, [], 2) 
% return the max value and col index along the row
% dim of Y and I is (m,1)

p = I;

% =========================================================================


end
