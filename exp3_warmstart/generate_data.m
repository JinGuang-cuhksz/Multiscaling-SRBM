% generate_data.m — Warm-start MLMC data generation.
%
% SRBM setting:
%   d = 3, Gamma = I_3,
%   R = [1 -0.6 -0.4; -0.5 1 -0.4; -0.2 -0.3 1],
%   delta = (r, r^2, r^3) with r = 0.2, mu = -R*delta.
%
% Three initial distributions:
%   (1) origin                        m0 = (0, 0, 0)
%   (2) skew-symmetric  Exp           m0 = (2.5, 12.5, 62.5)
%   (3) multi-scaling   Exp           m0 = (2.5, 22.32, 179.69)
%
% MLMC hyperparameter grid:
%   L     in {2, 4, 6, 8}
%   T     in {5000, 100000}
%   gamma in {0.05, 0.1, 0.2}    (paper figure shows only gamma = 0.2)
%
% For each (init, L, T, gamma) configuration we generate
%   Nest_per_config = 50 independent MLMC estimators,
%   each estimator built from Nepsilon = 2000 sample paths.
%
% Output:
%   results/data/all_results.mat   (struct array, one row per config)
%
% Parallelism:
%   The inner 2000-path loop is parallelised with parfor over the workers
%   of the local pool. Pool size is taken from $SLURM_CPUS_PER_TASK if set.
%
% Quick smoke test:
%   set environment variable SEC33_SMOKE=1 before launching MATLAB.
%   That switches to a tiny grid (L=2, single T, single gamma, 5 estimators
%   x 100 paths) which finishes in a couple of minutes — use it to verify
%   the pipeline before queueing the multi-day job.

clear; clc;
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'lib'));

if ~exist('results', 'dir');               mkdir('results');               end
if ~exist('results/data', 'dir');  mkdir('results/data');  end

%% ---------- Fixed SRBM parameters ----------
d        = 3;
SigmaMat = eye(d);
R        = [ 1.0, -0.6, -0.4;
            -0.5,  1.0, -0.4;
            -0.2, -0.3,  1.0];
r_scale  = 0.2;
delta    = [r_scale; r_scale^2; r_scale^3];
u        = -R * delta;
C        = cholcov(SigmaMat);

% Initial distributions
init_specs(1) = struct('name','origin',        'label','Origin',                    'm_vec',[0;     0;      0     ]);
init_specs(2) = struct('name','skew',          'label','Skew symmetric',            'm_vec',[2.5;   12.5;   62.5  ]);
init_specs(3) = struct('name','multi_scaling', 'label','Multi-scaling',             'm_vec',[2.5;   22.32;  179.69]);

%% ---------- MLMC hyperparameter grid ----------
smoke = strcmp(getenv('SEC33_SMOKE'), '1');
if smoke
    fprintf('*** SMOKE TEST MODE — tiny grid, low sample count ***\n');
    gamma_grid      = 0.2;
    T_grid          = 5000;
    L_grid          = 2;
    Nest_per_config = 5;
    Nepsilon        = 100;
else
    gamma_grid      = [0.2];                  % extend to [0.05 0.1 0.2] if you want all three
    T_grid          = [5000, 100000];
    L_grid          = [2, 4, 6, 8];
    Nest_per_config = 50;
    Nepsilon        = 2000;
end

%% ---------- Build flat config list ----------
% Order matters: (gamma, T, L, init). The flat index becomes the
% SLURM_ARRAY_TASK_ID in array mode.
configs = struct([]);
for ig = 1:numel(gamma_grid)
    for iT = 1:numel(T_grid)
        for iL = 1:numel(L_grid)
            for iI = 1:numel(init_specs)
                c = struct( ...
                    'gammas',  gamma_grid(ig), ...
                    'T',       T_grid(iT),     ...
                    'L',       L_grid(iL),     ...
                    'init_idx', iI);
                if isempty(configs); configs = c; else; configs(end+1) = c; end %#ok<AGROW>
            end
        end
    end
end
n_configs = numel(configs);

%% ---------- Decide which configs this process owns ----------
% Array mode: SLURM_ARRAY_TASK_ID = 1..n_configs picks one config.
% Single-job mode: run all n_configs sequentially.
task_id_str = getenv('SLURM_ARRAY_TASK_ID');
task_id     = str2double(task_id_str);
if ~isnan(task_id) && task_id >= 1
    if task_id > n_configs
        error('SLURM_ARRAY_TASK_ID=%d exceeds n_configs=%d', task_id, n_configs);
    end
    config_indices = task_id;
    out_file       = sprintf('results/data/config_%02d.mat', task_id);
    fprintf('ARRAY MODE — task %d/%d, output -> %s\n', task_id, n_configs, out_file);
else
    config_indices = 1:n_configs;
    out_file       = 'results/data/all_results.mat';
    fprintf('SINGLE-JOB MODE — running all %d configs, output -> %s\n', n_configs, out_file);
end

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
fprintf('Parallel pool: %d workers\n', pool.NumWorkers);

%% ---------- Main loop ----------
all_results = struct([]);
global_t0   = tic;

for k = 1:numel(config_indices)
    cfg     = configs(config_indices(k));
    gammas  = cfg.gammas;
    gamma_n = round(1/gammas);
    T       = cfg.T;
    L       = cfg.L;
    spec    = init_specs(cfg.init_idx);
    M_vec   = 0:L-1;
    vPM     = gammas.^M_vec / sum(gammas.^M_vec);

    fprintf('\n[%d/%d] init=%s | gamma=%.3g | T=%d | L=%d\n', ...
            k, numel(config_indices), spec.name, gammas, T, L);

    est_means      = zeros(Nest_per_config, d);
    est_complexity = zeros(Nest_per_config, 1);
    est_runtime    = zeros(Nest_per_config, 1);

    for iE = 1:Nest_per_config
        sampleM = randsample(M_vec, Nepsilon, true, vPM);
        counts  = histc(sampleM, M_vec);
        ePM     = counts / Nepsilon;

        stateData = zeros(Nepsilon, 3*d+2);
        m0_vec    = spec.m_vec;

        t0 = tic;
        parfor i = 1:Nepsilon
            y0 = exprnd(m0_vec);                 % exprnd(0) == 0 deterministically
            stateData(i,:) = MLMC_RBM_Trial_Mem( ...
                sampleM(i), u, C, gamma_n, d, ePM, y0, R, T);
        end
        rt = toc(t0);

        Z = evaluate_Z(stateData, d);
        est_means(iE, :)   = mean(Z, 1);
        est_complexity(iE) = sum(stateData(:, 3*d+2));
        est_runtime(iE)    = rt;

        if mod(iE, max(1, floor(Nest_per_config/10))) == 0 || iE == Nest_per_config
            elapsed = toc(global_t0);
            fprintf('   estimator %2d/%2d  rt=%.1fs  E[Z]=%s  totalElapsed=%.1fmin\n', ...
                    iE, Nest_per_config, rt, mat2str(est_means(iE,:),4), elapsed/60);
        end
    end

    row = struct( ...
        'init',         spec.name,            ...
        'init_label',   spec.label,           ...
        'm0',           spec.m_vec(:)',       ...
        'gamma',        gammas,               ...
        'T',            T,                    ...
        'L',            L,                    ...
        'Nest',         Nest_per_config,      ...
        'Nepsilon',     Nepsilon,             ...
        'means',        est_means,            ...
        'complexity',   est_complexity,       ...
        'runtime',      est_runtime);

    if isempty(all_results)
        all_results = row;
    else
        all_results(end+1) = row; %#ok<AGROW>
    end

    % Incremental save so the run can be interrupted and resumed
    save(out_file, 'all_results', '-v7.3');
    fprintf('   -> saved %d config(s) to %s\n', numel(all_results), out_file);
end

fprintf('\nDone. %d config(s) processed in %.1f min.\n', numel(config_indices), toc(global_t0)/60);
if isnan(task_id)
    fprintf('Run make_figure to assemble the figure.\n');
else
    fprintf('After all array tasks finish, run aggregate then make_figure.\n');
end

%% ---------- Helpers ----------
function Z = evaluate_Z(stateData, d)
    YM1   = stateData(:, 1:d);
    YM    = stateData(:, d+1:2*d);
    y0    = stateData(:, 2*d+1:3*d);
    w     = stateData(:, 3*d+1);
    Z     = (YM1 - YM) ./ w + y0;
end
