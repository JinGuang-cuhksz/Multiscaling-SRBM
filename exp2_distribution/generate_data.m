% generate_data.m
%
% 3D SRBM with MLMC and a per-path raw-sum CCDF estimator, producing the data
% for the distribution/tail figure (make_figure.py).
%
% Per-task environment variables (set before launching MATLAB):
%   WARMSTART      : one of {origin, skew, multi_scaling}      (required)
%   SEC33_L        : MLMC depth L                              (required)
%   SEC33_SIGMA    : covariance matrix tag                     (optional)
%                      'identity' (default) -> SigmaMat = I_3
%                      'corr_v1'            -> [1 -.6 -.3; -.6 1 -.4; -.3 -.4 1]
%   SEC33_GAMMA, SEC33_T, SEC33_NEST, SEC33_NEPSILON  optional
%                      (defaults: 0.2 / 1e5 / 50 / 2000)
%
% Output:
%   results/data/<WS>[_sigma_<TAG>]_L<L>_T<T>_...mat
%   (sigma tag is omitted from the filename when SEC33_SIGMA=identity.)

clear; clc;

this_dir  = fileparts(mfilename('fullpath'));
repo_root = fileparts(this_dir);
addpath(fullfile(repo_root, 'lib'));
addpath(this_dir);

results_dir = fullfile(this_dir, 'results', 'data');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

%% ---------- Fixed SRBM setting ----------
d        = 3;
sigma_tag = getenv('SEC33_SIGMA');
if isempty(sigma_tag), sigma_tag = 'identity'; end
switch sigma_tag
    case 'identity'
        SigmaMat = eye(d);
    case 'corr_v1'
        SigmaMat = [ 1.0, -0.6, -0.3;
                    -0.6,  1.0, -0.4;
                    -0.3, -0.4,  1.0];
    otherwise
        error('Unknown SEC33_SIGMA="%s". Use identity or corr_v1.', sigma_tag);
end
if any(eig(SigmaMat) <= 0)
    error('SigmaMat is not positive definite for SEC33_SIGMA="%s".', sigma_tag);
end
R        = [ 1.0, -0.6, -0.4;
            -0.5,  1.0, -0.4;
            -0.2, -0.3,  1.0];
r_scale  = 0.2;
delta    = [r_scale; r_scale^2; r_scale^3];
u        = -R * delta;
C        = cholcov(SigmaMat);

warmstart_table = struct( ...
    'origin',        [0;     0;      0     ], ...
    'skew',          [2.5;   12.5;   62.5  ], ...
    'multi_scaling', [2.5;   22.32;  179.69]);

%% ---------- Hyperparameters from env ----------
warmstart = getenv('WARMSTART');
if isempty(warmstart)
    warmstart = 'multi_scaling';
end
if ~isfield(warmstart_table, warmstart)
    error('WARMSTART must be one of {origin, skew, multi_scaling}. Got "%s".', warmstart);
end
m0 = warmstart_table.(warmstart);

gamma = get_env_double('SEC33_GAMMA',    0.2);
L     = get_env_double('SEC33_L',        4);
T     = get_env_double('SEC33_T',        100000);
Nest  = get_env_double('SEC33_NEST',     50);
Neps  = get_env_double('SEC33_NEPSILON', 2000);

gamma_n = round(1/gamma);
if abs(gamma - 1/gamma_n) > 1e-12
    error('SEC33_GAMMA must have integer reciprocal. Got %.16g.', gamma);
end

%% ---------- Reference CCDF (multi-scaling warmstart, raw sum) ----------
% Use the multi-scaling m0 as the canonical t-grid reference so that all
% (warmstart, L) runs share the same grid and can be overlaid.
m_ref = warmstart_table.multi_scaling;
ccdf_opts = struct('target_ccdf', 0.01, 'n_pts', 800);
[raw_t_grid, raw_ref_ccdf, raw_ref_pdf, raw_ccdf_info] = hypoexp_ccdf(m_ref, ccdf_opts);

%% ---------- Parallel pool ----------
ncpus = str2double(getenv('SLURM_CPUS_PER_TASK'));
if isnan(ncpus) || ncpus < 1
    ncpus = max(1, feature('numcores') - 1);
end
pool = gcp('nocreate');
if isempty(pool)
    pool = parpool('local', ncpus);
elseif pool.NumWorkers ~= ncpus
    delete(pool);
    pool = parpool('local', ncpus);
end

%% ---------- File tagging ----------
job_id  = getenv('SLURM_ARRAY_JOB_ID');
if isempty(job_id); job_id = getenv('SLURM_JOB_ID'); end
if isempty(job_id); job_id = 'local';                end
task_id = get_env_double('SLURM_ARRAY_TASK_ID', 0);

if strcmp(sigma_tag, 'identity')
    sigma_suffix = '';   % preserve filename compatibility with the original 4 runs
else
    sigma_suffix = sprintf('_sigma_%s', sigma_tag);
end
tag = sprintf('%s%s_gamma_%s_L_%d_T_%d_Nest_%d_Neps_%d', ...
              warmstart, sigma_suffix, sanitize_num(gamma), L, T, Nest, Neps);
out_file = fullfile(results_dir, sprintf('%s.mat', tag));
csv_file = fullfile(results_dir, sprintf('%s_summary.csv', tag));

fprintf('\n=== warmstart + CCDF MLMC ===\n');
fprintf('SigmaMat tag = %s\n%s\n', sigma_tag, mat2str(SigmaMat, 6));
fprintf('warmstart=%s   m0 = %s\n', warmstart, mat2str(m0', 6));
fprintf('job_id=%s array_task=%d\n', job_id, task_id);
fprintf('gamma=%.6g gamma_n=%d L=%d T=%d Nest=%d Nepsilon=%d workers=%d\n', ...
        gamma, gamma_n, L, T, Nest, Neps, pool.NumWorkers);
fprintf('delta = %s\n', mat2str(delta', 6));
fprintf('reference m_ref (multi-scaling) = %s\n', mat2str(m_ref', 6));
fprintf('raw t_star = %.4f, target CCDF = %.4g, n_grid = %d\n', ...
        raw_ccdf_info.t_target, raw_ccdf_info.target_ccdf, numel(raw_t_grid));
fprintf('out_file = %s\n', out_file);

M_vec = 0:L-1;
vPM   = gamma.^M_vec / sum(gamma.^M_vec);

raw_est_means      = zeros(Nest, d);
raw_est_S_mean     = zeros(Nest, 1);
raw_est_ccdf_grid  = zeros(Nest, numel(raw_t_grid));
est_complexity     = zeros(Nest, 1);
est_runtime        = zeros(Nest, 1);
est_level_cnt      = zeros(Nest, L);

config = struct( ...
    'section',          '3.3', ...
    'quantity',         'warmstart_raw_sum_ccdf', ...
    'warmstart',        warmstart, ...
    'sigma_tag',        sigma_tag, ...
    'm0',               m0,        ...
    'dimension',        d,         ...
    'gamma',            gamma,     ...
    'gamma_n',          gamma_n,   ...
    'L',                L,         ...
    'T',                T,         ...
    'Nest',             Nest,      ...
    'Nepsilon',         Neps,      ...
    'M_vec',            M_vec,     ...
    'vPM',              vPM,       ...
    'R',                R,         ...
    'SigmaMat',         SigmaMat,  ...
    'delta',            delta,     ...
    'drift',            u,         ...
    'm_ref',            m_ref,     ...
    'raw_t_grid',       raw_t_grid,    ...
    'raw_ref_ccdf',     raw_ref_ccdf,  ...
    'raw_ref_pdf',      raw_ref_pdf,   ...
    'raw_ccdf_info',    raw_ccdf_info, ...
    'job_id',           job_id,    ...
    'array_task_id',    task_id);

global_t0 = tic;

for iE = 1:Nest
    sampleM = randsample(M_vec, Neps, true, vPM);
    counts  = histc(sampleM, M_vec);
    ePM     = counts / Neps;

    stateData = zeros(Neps, 3*d+2);

    t0 = tic;
    parfor i = 1:Neps
        y0 = exprnd(m0);  % exprnd(0) == 0 — gives the origin warmstart literally
        stateData(i,:) = MLMC_RBM_Trial_Mem(sampleM(i), u, C, gamma_n, d, ePM, y0, R, T);
    end
    est_runtime(iE) = toc(t0);

    Z_raw = evaluate_identity(stateData, d);
    raw_est_means(iE, :)     = mean(Z_raw, 1);
    raw_est_S_mean(iE)       = mean(sum(Z_raw, 2));
    est_complexity(iE)       = sum(stateData(:, 3*d+2));
    est_level_cnt(iE, :)     = counts(:)';
    raw_est_ccdf_grid(iE, :) = estimate_raw_sum_ccdf_grid(stateData, d, raw_t_grid);

    raw_ccdf_tstar_i = interp1(raw_t_grid, raw_est_ccdf_grid(iE,:), ...
                               raw_ccdf_info.t_target, 'linear', 'extrap');
    fprintf('estimator %d/%d runtime=%.1fs raw_S_mean=%.6f raw_CCDF(t*)=%.6f elapsed=%.1fmin\n', ...
            iE, Nest, est_runtime(iE), raw_est_S_mean(iE), raw_ccdf_tstar_i, ...
            toc(global_t0)/60);

    completed = iE; %#ok<NASGU>
    save(out_file, 'config', 'completed', 'raw_est_means', 'raw_est_S_mean', ...
         'raw_est_ccdf_grid', 'est_complexity', 'est_runtime', 'est_level_cnt', '-v7.3');

    clear stateData Z_raw sampleM counts ePM;
end

mean_Z_raw   = mean(raw_est_means, 1);
std_Z_raw    = std(raw_est_means, 0, 1);
raw_ccdf_tstar = mean(interp1(raw_t_grid, raw_est_ccdf_grid.', ...
                              raw_ccdf_info.t_target, 'linear', 'extrap'));

summary = table({warmstart}, gamma, L, T, Nest, Neps, ...
                {mean_Z_raw}, {std_Z_raw}, {m0'}, ...
                mean(raw_est_S_mean), std(raw_est_S_mean), ...
                raw_ccdf_info.t_target, raw_ccdf_tstar, ...
                mean(est_complexity), sum(est_runtime), toc(global_t0), ...
                'VariableNames', {'warmstart','gamma','L','T','Nest','Nepsilon', ...
                                  'mean_Z_raw','std_Z_raw','m0', ...
                                  'mean_S_raw','std_S_raw','raw_t_star','raw_ccdf_tstar', ...
                                  'mean_complexity','sum_runtime','walltime'});
writetable(summary, csv_file);

fprintf('\nDone. Total wall time %.2f min.\n', toc(global_t0)/60);
fprintf('mean raw Z = %s\n', mat2str(mean_Z_raw, 6));
fprintf('Saved %s\n', out_file);
fprintf('Saved %s\n', csv_file);
exit(0);

function v = get_env_double(name, default_value)
    raw = getenv(name);
    if isempty(raw)
        v = default_value;
    else
        v = str2double(raw);
        if isnan(v)
            error('Environment variable %s must be numeric. Got "%s".', name, raw);
        end
    end
end

function s = sanitize_num(x)
    s = strrep(sprintf('%.12g', x), '.', 'p');
    s = strrep(s, '-', 'm');
end

function Z = evaluate_identity(stateData, d)
    YM1 = stateData(:, 1:d);
    YM  = stateData(:, d+1:2*d);
    y0  = stateData(:, 2*d+1:3*d);
    w   = stateData(:, 3*d+1);
    Z   = (YM1 - YM) ./ w + y0;
end

function ccdf_hat = estimate_raw_sum_ccdf_grid(stateData, d, t_grid)
    YM1 = stateData(:, 1:d);
    YM  = stateData(:, d+1:2*d);
    y0  = stateData(:, 2*d+1:3*d);
    w   = stateData(:, 3*d+1);

    S_fine   = sum(YM1, 2);
    S_coarse = sum(YM, 2);
    S0       = sum(y0, 2);

    ccdf_hat = zeros(1, numel(t_grid));
    for j = 1:numel(t_grid)
        tj = t_grid(j);
        vals = (double(S_fine > tj) - double(S_coarse > tj)) ./ w + double(S0 > tj);
        ccdf_hat(j) = mean(vals);
    end
end
