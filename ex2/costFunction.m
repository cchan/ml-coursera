function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

h = sigmoid(X*theta);
J = sum(-y.*log(h)-(1-y).*log(1-h)) / m; %could also be -y log(h) -(1-y)log(h)
grad = X' * (h - y) ./ m;

%========TRYING TO PROVE CONVEXITY=======
% factor * sum ( cost(h_th(x),y) )
% h = 1/(1+e^-(th*x))

% J = sum(
	%-log(h) if y=1
	%-log(1-h) if y=0
%) ------> Conveniently, value of higher at 1 is 0 and value of lower at 0 is 0. And they reach up to infinity at the opposite end.
% WHEN WE VARY THETA.... uhm. whatever x and y are, there is convex function J w/ global minimum w.r.t th. :( How do you prove convexity??

%(Hm false positive and false negative may have different costs - how can this be reflected preserving convexity?)

% Oh... knowing that both log error functions are convex w.r.t. h_th(x), if sigmoid(x) isn't bendy enough to change the convexity of J(sigmoid(x)) w.r.t. x, then that's nice. It's just annoying to do.

% GOT DIS: http://mathgotchas.blogspot.com/2011/10/why-is-error-function-minimized-in.html
%=========================================


% =============================================================

end
