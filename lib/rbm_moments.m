function [mean_x1, mean_x2, var_x1, var_x2, cor_x1x2] = rbm_moments(mu1, mu2, alpha, beta)
% Stationary moments of a 2D reflected Brownian motion (Foddy 1984 formulas).
% Inputs:
%   mu1, mu2     : drift parameters (must be negative)
%   alpha, beta  : reflection parameters with alpha*beta < 1; the reflection
%                  vector is (1, -beta) on {x1 = 0} and (-alpha, 1) on
%                  {x2 = 0}, i.e. R = [1 -alpha; -beta 1]
% Outputs:
%   mean_x1, mean_x2 : first moments (means)
%   var_x1, var_x2   : variances
%   cor_x1x2         : correlation coefficient

% Numerical tolerance used in floating-point comparisons below.
global epsilon;
epsilon = 1e-4;

if alpha*beta >= 1
    error('alpha*beta < 1 is required');
end
if alpha < 0 || beta < 0
    error('alpha and beta must be non-negative');
end

if mu1 >= 0 || mu2 >= 0
    error('mu1 and mu2 must be non-positive');
end

% |mu|^2 = mu1^2 + mu2^2
mu_abs_squared = mu1^2 + mu2^2;
mu_abs = sqrt(mu_abs_squared);

% Substitutions C1, C2 used throughout the formulas.
C1 = -(mu2 + beta*mu1)/(1 - alpha*beta);
C2 = -(mu1 + alpha*mu2)/(1 - alpha*beta);

[A, B, C, D, E, F, G, H, p, q] = initialize_parameters(mu1, mu2, alpha, beta, mu_abs_squared);

% Swap mu1<->mu2 and alpha<->beta for the second coordinate.
[A_, B_, C_, D_, E_, F_, G_, H_, p_, q_] = initialize_parameters(mu2, mu1, beta, alpha, mu_abs_squared);

% h1'(0) - Eq. (31)
h1_prime_0 = calculate_h1_prime_0(mu1, mu2, mu_abs, A, B, C, D, E, F, G, H, p, q);

% h2'(0) - Eq. (31), with mu and (alpha,beta) swapped
h2_prime_0 = calculate_h1_prime_0(mu2, mu1, mu_abs, A_, B_, C_, D_, E_, F_, G_, H_, p_, q_);

% h1''(0) - Eqs. (31) and (38)
h1_double_prime_0 = calculate_h1_double_prime_0(mu1, mu2, mu_abs, A, B, C, D, E, F, G, H, p, q, h1_prime_0);

% h2''(0) - Eqs. (31) and (38), with mu and (alpha,beta) swapped
h2_double_prime_0 = calculate_h1_double_prime_0(mu2, mu1, mu_abs, A_, B_, C_, D_, E_, F_, G_, H_, p_, q_, h2_prime_0);

% First moments - Eqs. (26) and (27)
mean_x1 = (-alpha*mu1*C1*h1_prime_0 - 0.5*alpha*C1 + 0.5*C2)/(mu1^2);
mean_x2 = (-beta*mu2*C2*h2_prime_0 - 0.5*beta*C2 + 0.5*C1)/(mu2^2);

% Second moments - Eqs. (28) and (29)
moment2_x1 = alpha*C1*mu1^(-1)*(h1_double_prime_0 + h1_prime_0^2 + mu1^(-1)*h1_prime_0 + 0.5*mu1^(-2)) - 0.5*C2*mu1^(-3);
moment2_x2 = beta*C2*mu2^(-1)*(h2_double_prime_0 + h2_prime_0^2 + mu2^(-1)*h2_prime_0 + 0.5*mu2^(-2)) - 0.5*C1*mu2^(-3);

% Cross moment - Eq. (30)
moment2_x1x2 = -1/(mu1^3)*(0.5*mu2 + (alpha*mu2 + 0.5*mu1)*C1*h1_prime_0 + 0.5*mu1*C2*h2_prime_0 + ...
            0.5*mu1*(mu1 + alpha*mu2)*C1*(h1_double_prime_0 + (h1_prime_0)^2));

var_x1 = moment2_x1 - mean_x1^2;
var_x2 = moment2_x2 - mean_x2^2;
cov_x1x2 = moment2_x1x2 - mean_x1*mean_x2;
cor_x1x2 = cov_x1x2/(sqrt(var_x1)*sqrt(var_x2));

end

function result = calculate_h1_prime_0(mu1, mu2, mu_abs, A, B, C, D, E, F, G, H, p, q)
% h1'(0) - Eq. (31)
global epsilon;

% Integral of g1'(x)/(x^2+mu1^2)
if abs(p) < epsilon
    integral_g1 = calculate_g1_prime_integral_zero(mu1, mu2, mu_abs, A, C);
elseif abs(abs(p) - abs(mu1)) < epsilon
    integral_g1 = calculate_g1_prime_integral_equal(mu1, mu2, mu_abs, A, B, C, D);
else
    integral_g1 = calculate_g1_prime_integral_general(mu1, mu2, mu_abs, A, B, C, D, p);
end

% Integral of g2'(x)/(x^2+mu1^2)
if abs(q) < epsilon
    integral_g2 = calculate_g2_prime_integral_zero(mu1, mu2, mu_abs, E, G);
elseif abs(abs(q) - abs(mu1)) < epsilon
    integral_g2 = calculate_g2_prime_integral_equal(mu1, mu2, mu_abs, E, F, G, H);
else
    integral_g2 = calculate_g2_prime_integral_general(mu1, mu2, mu_abs, E, F, G, H, q);
end

result = -2*mu1/pi * (integral_g1 + integral_g2);
end

function result = calculate_h1_double_prime_0(mu1, mu2, mu_abs, A, B, C, D, E, F, G, H, p, q, h1_prime_0)
% h1''(0) - Eq. (38)
global epsilon;

% Integral of g1'(x)/(x^2+mu1^2)^2
if abs(p) < epsilon
    integral_g1 = calculate_g1_prime_integral_second_zero(mu1, mu2, mu_abs, A, C);
elseif abs(abs(p) - abs(mu1)) < epsilon
    integral_g1 = calculate_g1_prime_integral_second_equal(mu1, mu2, mu_abs, A, B, C, D);
else
    integral_g1 = calculate_g1_prime_integral_second_general(mu1, mu2, mu_abs, A, B, C, D, p);
end

% Integral of g2'(x)/(x^2+mu1^2)^2
if abs(q) < epsilon
    integral_g2 = calculate_g2_prime_integral_second_zero(mu1, mu2, mu_abs, E, G);
elseif abs(abs(q) - abs(mu1)) < epsilon
    integral_g2 = calculate_g2_prime_integral_second_equal(mu1, mu2, mu_abs, E, F, G, H);
else
    integral_g2 = calculate_g2_prime_integral_second_general(mu1, mu2, mu_abs, E, F, G, H, q);
end

integral_term = 2*mu1^2 * (integral_g1 + integral_g2);
result = (-pi/(2*mu1) * h1_prime_0 - integral_term) * (2/pi);
end

% Integral of g1'(x)/(x^2+mu1^2), general case - Eq. (32)
function result = calculate_g1_prime_integral_general(mu1, mu2, mu_abs, A, B, C, D, p)
global epsilon;

result = A/((p^2-mu1^2)^2)*(pi*p^2/(2*abs(mu1)) + mu1^2*(p^2-mu1^2)*pi/(4*mu1^3) - pi*p^2/(2*abs(p)));
result = result + C/((p^2-mu1^2)^2)*(p^2/(mu1*mu2)*atan(mu2/mu1) - mu1^2*(p^2-mu1^2)/(2*mu1^3*mu2)*((mu2^2-mu1^2)/mu2^2*atan(mu2/mu1) + mu1/mu2));

% Case p != |mu|
if abs(p - mu_abs) > epsilon
    result = result + C/((p^2-mu1^2)^2)*(-p^2/(p*sqrt(mu_abs^2-p^2))*atan(sqrt(mu_abs^2-p^2)/p));
else
    result = result + C/((p^2-mu1^2)^2)*(-1);
end

result = result + B/((p^2-mu1^2)^2)*(pi/(2*mu1) - pi*(p^2-mu1^2)/(4*mu1^3) + pi/(2*abs(p)));
result = result + D/((p^2-mu1^2)^2)*(-1/(mu1*mu2)*atan(mu2/mu1) + (p^2-mu1^2)/(2*mu1^3*mu2)*((mu2^2-mu1^2)/mu2^2*atan(mu2/mu1) + mu1/mu2));

% Case p != |mu|
if abs(p - mu_abs) > epsilon
    result = result + D/((p^2-mu1^2)^2)*(1/(p*sqrt(mu_abs^2-p^2))*atan(sqrt(mu_abs^2-p^2)/p));
else
    result = result + D/((p^2-mu1^2)^2)*(1/mu_abs^2);
end
end

function result = calculate_g1_prime_integral_equal(mu1, mu2, mu_abs, A, B, C, D)
% Integral of g1'(x)/(x^2+mu1^2) for p = mu1 - Eq. (33)
global epsilon;

result = A*(-pi/(4*mu1^3) + 3*pi*mu1^2/(16*mu1^5));
result = result + C*(1/(2*mu1^3*mu2)*((mu2^2-mu1^2)/mu2^2*atan(mu2/mu1) + mu1/mu2)) + C*(-mu1^2/(4*mu1^5*mu2))*((3/2 - mu1^2/mu2^2 + 3*mu1^4/(2*mu2^4))*atan(mu2/mu1) + 3*mu1*(mu2^2-mu1^2)/(2*mu2^3));
result = result - B*(3*pi/(16*mu1^5));
result = result + D*(1/(4*mu1^5*mu2))*((3/2 - mu1^2/mu2^2 + 3*mu1^4/(2*mu2^4))*atan(mu2/mu1) + 3*mu1*(mu2^2-mu1^2)/(2*mu2^3));
end

function result = calculate_g1_prime_integral_zero(mu1, mu2, mu_abs, A, C)
% Integral of g1'(x)/(x^2+mu1^2) for p = 0 - Eq. (34)
global epsilon;

result = A*(-pi/(4*mu1^3)) + C*(1/(2*mu1^3*mu2))*((mu2^2-mu1^2)/mu2^2*atan(mu2/mu1) + mu1/mu2);
end

function result = calculate_g2_prime_integral_general(mu1, mu2, mu_abs, E, F, G, H, q)
% Integral of g2'(x)/(x^2+mu1^2), general case - Eq. (35)
global epsilon;

result = E/((q^2-mu1^2)^2)*(pi*q^2/(2*abs(mu1)) + mu1^2*(q^2-mu1^2)*pi/(4*mu1^3) - pi*q^2/(2*abs(q)));
result = result + G/((q^2-mu1^2)^2)*(q^2/(mu1*mu2)*atan(mu2/mu1) - mu1^2*(q^2-mu1^2)/(2*mu1^3*mu2)*((mu2^2-mu1^2)/mu2^2*atan(mu2/mu1) + mu1/mu2));

% Case q != -|mu|
if abs(q + mu_abs) > epsilon
    result = result + G/((q^2-mu1^2)^2)*(-q^2/(q*sqrt(mu_abs^2-q^2))*atan(sqrt(mu_abs^2-q^2)/q));
else
    result = result + G/((q^2-mu1^2)^2)*(-1);
end

result = result + F/((q^2-mu1^2)^2)*(pi/(2*mu1) - pi*(q^2-mu1^2)/(4*mu1^3) + pi/(2*abs(q)));
result = result + H/((q^2-mu1^2)^2)*(-1/(mu1*mu2)*atan(mu2/mu1) + (q^2-mu1^2)/(2*mu1^3*mu2)*((mu2^2-mu1^2)/mu2^2*atan(mu2/mu1) + mu1/mu2));

% Case q != -|mu|
if abs(q + mu_abs) > epsilon
    result = result + H/((q^2-mu1^2)^2)*(1/(q*sqrt(mu_abs^2-q^2))*atan(sqrt(mu_abs^2-q^2)/q));
else
    result = result + H/((q^2-mu1^2)^2)*(1/mu_abs^2);
end
end

function result = calculate_g2_prime_integral_equal(mu1, mu2, mu_abs, E, F, G, H)
% Integral of g2'(x)/(x^2+mu1^2) for q = mu1 - Eq. (36)
result = E*(-pi/(4*mu1^3) + 3*pi*mu1^2/(16*mu1^5));
result = result + G*(1/(2*mu1^3*mu2)*((mu2^2-mu1^2)/mu2^2*atan(mu2/mu1) + mu1/mu2) - mu1^2/(4*mu1^5*mu2)*((3/2 - mu1^2/mu2^2 + 3*mu1^4/(2*mu2^4))*atan(mu2/mu1) + 3*mu1*(mu2^2-mu1^2)/(2*mu2^3)));
result = result - F*(3*pi/(16*mu1^5));
result = result + H*(1/(4*mu1^5*mu2))*((3/2 - mu1^2/mu2^2 + 3*mu1^4/(2*mu2^4))*atan(mu2/mu1) + 3*mu1*(mu2^2-mu1^2)/(2*mu2^3));
end

function result = calculate_g2_prime_integral_zero(mu1, mu2, mu_abs, E, G)
% Integral of g2'(x)/(x^2+mu1^2) for q = 0 - Eq. (37)
result = E*(-pi/(4*mu1^3)) + G*(1/(2*mu1^3*mu2))*((mu2^2-mu1^2)/mu2^2*atan(mu2/mu1) + mu1/mu2);
end

% Integral of g1'(x)(x^2-mu1^2)/(x^2+mu1^2)^2
function result = calculate_g1_prime_integral_second_general(mu1, mu2, mu_abs, A, B, C, D, p)
% General case - Eq. (39)
global epsilon;

result = A/((p^2-mu1^2)^3)*(-p^2*pi/(2*abs(mu1)) - pi*p^2*(p^2-mu1^2)/(4*mu1^3) + 3*pi*mu1^2*(p^2-mu1^2)^2/(16*mu1^5) + p^2*pi/(2*abs(p)));

result = result + C/((p^2-mu1^2)^3)*(-p^2/(mu1*mu2)*atan(mu2/mu1) + ...
    p^2*(p^2-mu1^2)/(2*mu1^3*mu2)*((mu2^2-mu1^2)/mu2^2*atan(mu2/mu1) + mu1/mu2) - ...
    mu1^2*(p^2-mu1^2)^2/(4*mu1^5*mu2)*((3/2 - mu1^2/mu2^2 + 3*mu1^4/(2*mu2^4))*atan(mu2/mu1) + 3*mu1*(mu2^2-mu1^2)/(2*mu2^3)));

% Case p != |mu|
if abs(p - mu_abs) > epsilon
    result = result + C/((p^2-mu1^2)^3)*(p^2/(sqrt(mu_abs^2-p^2)*p)*atan(sqrt(mu_abs^2-p^2)/p));
else
    result = result + C/((p^2-mu1^2)^3)*(p^2/mu_abs^2);
end

result = result + B/((p^2-mu1^2)^3)*(pi/(2*abs(mu1)) + (p^2-mu1^2)*pi/(4*mu1^3) - 3*pi*(p^2-mu1^2)^2/(16*mu1^5) - pi/(2*abs(p)));

result = result + D/((p^2-mu1^2)^3)*(1/(mu1*mu2)*atan(mu2/mu1) - ...
    (p^2-mu1^2)/(2*mu1^3*mu2)*((mu2^2-mu1^2)/mu2^2*atan(mu2/mu1) + mu1/mu2) + ...
    (p^2-mu1^2)^2/(4*mu1^5*mu2)*((3/2 - mu1^2/mu2^2 + 3*mu1^4/(2*mu2^4))*atan(mu2/mu1) + 3*mu1*(mu2^2-mu1^2)/(2*mu2^3)));

% Case p != |mu|
if abs(p - mu_abs) > epsilon
    result = result + D/((p^2-mu1^2)^3)*(-1/(p*sqrt(mu_abs^2-p^2))*atan(sqrt(mu_abs^2-p^2)/p));
else
    result = result + D/((p^2-mu1^2)^3)*(-1/mu_abs^2);
end
end

function result = calculate_g1_prime_integral_second_equal(mu1, mu2, mu_abs, A, B, C, D)
% p = mu1 - Eq. (40)
result = A*(-3*pi/(16*mu1^5) + 5*pi*mu1^2/(32*mu1^7)) + B*(-5*pi/(32*mu1^7));

result = result + C*(1/(4*mu1^5*mu2))*((3/2 - mu1^2/mu2^2 + 3*mu1^4/(2*mu2^4))*atan(mu2/mu1) + 3*mu1*(mu2^2-mu1^2)/(2*mu2^3));

result = result + (-mu1^2*C + D)*(1/(mu1^7*mu2))*((16*mu2^6 - 24*mu2^4*mu_abs^2 + 18*mu2^2*mu_abs^4 - 5*mu_abs^6)/(16*mu2^6)*atan(mu2/mu1) + ...
    (3*mu1/(2*mu2) - 3*mu1*mu_abs^2/(2*mu2^3) + 5*mu1*mu_abs^4/(12*mu2^5)) + ...
    (mu1^2-mu2^2)*(3*mu1/(8*mu2^3) - 5*mu1*mu_abs^2/(48*mu2^5)) + mu1/(6*mu2));
end

function result = calculate_g1_prime_integral_second_zero(mu1, mu2, mu_abs, A, C)
% p = 0 - Eq. (41)
result = A*(-3*pi/(16*mu1^5)) + C*(1/(4*mu1^5*mu2))*((3/2 - mu1^2/mu2^2 + 3*mu1^4/(2*mu2^4))*atan(mu2/mu1) + 3*mu1*(mu2^2-mu1^2)/(2*mu2^3));
end

function result = calculate_g2_prime_integral_second_general(mu1, mu2, mu_abs, E, F, G, H, q)
% General case - Eq. (42)
global epsilon;

result = E/((q^2-mu1^2)^3)*(-q^2*pi/(2*abs(mu1)) - pi*q^2*(q^2-mu1^2)/(4*mu1^3) + 3*pi*mu1^2*(q^2-mu1^2)^2/(16*mu1^5) + q^2*pi/(2*abs(q)));

result = result + G/((q^2-mu1^2)^3)*(-q^2/(mu1*mu2)*atan(mu2/mu1) + ...
    q^2*(q^2-mu1^2)/(2*mu1^3*mu2)*((mu2^2-mu1^2)/mu2^2*atan(mu2/mu1) + mu1/mu2) - ...
    mu1^2*(q^2-mu1^2)^2/(4*mu1^5*mu2)*((3/2 - mu1^2/mu2^2 + 3*mu1^4/(2*mu2^4))*atan(mu2/mu1) + 3*mu1*(mu2^2-mu1^2)/(2*mu2^3)));

% Case q != -|mu|
if abs(q + mu_abs) > epsilon
    result = result + G/((q^2-mu1^2)^3)*(q^2/(sqrt(mu_abs^2-q^2)*q)*atan(sqrt(mu_abs^2-q^2)/q));
else
    result = result + G/((q^2-mu1^2)^3)*(q^2/mu_abs^2);
end

result = result + F/((q^2-mu1^2)^3)*(pi/(2*abs(mu1)) + (q^2-mu1^2)*pi/(4*mu1^3) - 3*pi*(q^2-mu1^2)^2/(16*mu1^5) - pi/(2*abs(q)));

result = result + H/((q^2-mu1^2)^3)*(1/(mu1*mu2)*atan(mu2/mu1) - ...
    (q^2-mu1^2)/(2*mu1^3*mu2)*((mu2^2-mu1^2)/mu2^2*atan(mu2/mu1) + mu1/mu2) + ...
    (q^2-mu1^2)^2/(4*mu1^5*mu2)*((3/2 - mu1^2/mu2^2 + 3*mu1^4/(2*mu2^4))*atan(mu2/mu1) + 3*mu1*(mu2^2-mu1^2)/(2*mu2^3)));

% Case q != -|mu|
if abs(q + mu_abs) > epsilon
    result = result + H/((q^2-mu1^2)^3)*(-1/(q*sqrt(mu_abs^2-q^2))*atan(sqrt(mu_abs^2-q^2)/q));
else
    result = result + H/((q^2-mu1^2)^3)*(-1/mu_abs^2);
end
end

function result = calculate_g2_prime_integral_second_equal(mu1, mu2, mu_abs, E, F, G, H)
% q = mu1 - Eq. (43)
result = E*(-3*pi/(16*mu1^5) + 5*pi*mu1^2/(32*mu1^7)) + F*(-5*pi/(32*mu1^7));

result = result + G*(1/(4*mu1^5*mu2))*((3/2 - mu1^2/mu2^2 + 3*mu1^4/(2*mu2^4))*atan(mu2/mu1) + 3*mu1*(mu2^2-mu1^2)/(2*mu2^3));

result = result + (-mu1^2*G + H)*(1/(mu1^7*mu2))*((16*mu2^6 - 24*mu2^4*mu_abs^2 + 18*mu2^2*mu_abs^4 - 5*mu_abs^6)/(16*mu2^6)*atan(mu2/mu1) + ...
    (3*mu1/(2*mu2) - 3*mu1*mu_abs^2/(2*mu2^3) + 5*mu1*mu_abs^4/(12*mu2^5)) + ...
    (mu1^2-mu2^2)*(3*mu1/(8*mu2^3) - 5*mu1*mu_abs^2/(48*mu2^5)) + mu1/(6*mu2));
end

function result = calculate_g2_prime_integral_second_zero(mu1, mu2, mu_abs, E, G)
% q = 0 - Eq. (44)
result = E*(-3*pi/(16*mu1^5)) + G*(1/(4*mu1^5*mu2))*((3/2 - mu1^2/mu2^2 + 3*mu1^4/(2*mu2^4))*atan(mu2/mu1) + 3*mu1*(mu2^2-mu1^2)/(2*mu2^3));
end

function [A, B, C, D, E, F, G, H, p, q] = initialize_parameters(mu1, mu2, alpha, beta, mu_abs_squared)
% Poles p, q
    p = (mu1*alpha^2 - 2*mu2*alpha - mu1)/(alpha^2 + 1);
    q = (mu1*beta^2 + 2*mu2*beta - mu1)/(beta^2 + 1);

% Coefficients A, B, C, D of g1'(x)
A = (-mu1*alpha^4 + mu2*alpha^3 - mu1*alpha^2 + mu2*alpha)/((alpha^2 + 1)^2);
B = (-mu1^3*alpha^4 + 3*mu1^2*mu2*alpha^3 + (mu1^3 - 2*mu1*mu2^2)*alpha^2 - mu1^2*mu2*alpha)/((alpha^2 + 1)^2);
C = (-(mu1^2 - mu2^2)*alpha^3 + 4*mu1*mu2*alpha^2 + (mu1^2 - mu2^2)*alpha)/((alpha^2 + 1)^2);
D = (-mu1^2*mu_abs_squared*alpha^3 + 2*mu1*mu2*mu_abs_squared*alpha^2 + mu1^2*mu_abs_squared*alpha)/((alpha^2 + 1)^2);

% Coefficients E, F, G, H of g2'(x)
E = (-mu2*beta^3 + mu1*beta^2 - mu2*beta + mu1)/((beta^2 + 1)^2);
F = (mu1^2*mu2*beta^3 + (2*mu1*mu2^2 - mu1^3)*beta^2 - 3*mu1^2*mu2*beta + mu1^3)/((beta^2 + 1)^2);
G = (-(mu1^2 - mu2^2)*beta^3 - 4*mu1*mu2*beta^2 + (mu1^2 - mu2^2)*beta)/((beta^2 + 1)^2);
H = (-mu1^2*mu_abs_squared*beta^3 - 2*mu1*mu2*mu_abs_squared*beta^2 + mu1^2*mu_abs_squared*beta)/((beta^2 + 1)^2);
end
