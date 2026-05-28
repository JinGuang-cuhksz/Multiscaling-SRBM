function [t, ccdf, pdf, info] = hypoexp_ccdf(means, opts)
% HYPOEXP_CCDF  CCDF and PDF of S = X_1 + ... + X_d where X_k ~ Exp(1/means(k))
% are independent (NOT i.i.d. — different rates allowed). Computed via the
% matrix exponential of the phase-type sub-generator.
%
%   [t, ccdf, pdf] = hypoexp_ccdf(means)
%   [t, ccdf, pdf, info] = hypoexp_ccdf(means, opts)
%
% Inputs
%   means  d-vector of mean values m_1, ..., m_d (rates lambda_k = 1/m_k)
%   opts   struct (all optional):
%            .target_ccdf  stop t-grid where ccdf reaches this; default 1e-2
%            .n_pts        number of grid points (default 600)
%            .t_max        explicit upper limit (overrides target)
%
% Outputs
%   t      1 x n_pts grid
%   ccdf   P(S > t)
%   pdf    f_S(t)
%   info   struct with .Q, .lambdas, .ES (=sum means), .t_target

    if nargin < 2, opts = struct(); end
    target = local_field(opts, 'target_ccdf', 1e-2);
    n_pts  = local_field(opts, 'n_pts',       600);

    means   = means(:);
    d       = numel(means);
    lambdas = 1 ./ means;

    Q      = -diag(lambdas) + diag(lambdas(1:end-1), 1);
    alpha  = [1, zeros(1, d-1)];
    one_d  = ones(d, 1);
    q_exit = -Q * one_d;       % (0,...,0,lambda_d)

    if isfield(opts, 't_max') && ~isempty(opts.t_max)
        t_max = opts.t_max;
    else
        % adaptive: double t until ccdf <= target
        t_try = max(2 * sum(means), 1);
        for j = 1:40
            c_try = alpha * expm(Q * t_try) * one_d;
            if c_try <= target, break; end
            t_try = 2 * t_try;
        end
        t_max = t_try;
    end

    t    = linspace(0, t_max, n_pts);
    ccdf = zeros(1, n_pts);
    pdf  = zeros(1, n_pts);
    for k = 1:n_pts
        E       = expm(Q * t(k));
        ccdf(k) = alpha * E * one_d;
        pdf(k)  = alpha * E * q_exit;
    end

    % Find t_target where ccdf first crosses target
    idx = find(ccdf <= target, 1, 'first');
    if isempty(idx), t_target = NaN; else, t_target = t(idx); end

    info = struct('Q', Q, 'lambdas', lambdas, 'ES', sum(means), ...
                  't_target', t_target, 'target_ccdf', target);
end

function v = local_field(s, name, default)
    if isfield(s, name) && ~isempty(s.(name)), v = s.(name); else, v = default; end
end
