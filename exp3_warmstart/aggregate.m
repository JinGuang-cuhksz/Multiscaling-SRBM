% aggregate.m — Combine per-task config_*.mat files (produced by
% generate_data in array mode) into a single all_results.mat that
% make_figure can consume.

addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'lib'));

dir_in  = 'results/data';
out     = fullfile(dir_in, 'all_results.mat');
files   = dir(fullfile(dir_in, 'config_*.mat'));
if isempty(files)
    error('No config_*.mat files in %s', dir_in);
end

[~, idx] = sort({files.name});
files    = files(idx);

all_results = struct([]);
for i = 1:numel(files)
    S = load(fullfile(files(i).folder, files(i).name));
    if ~isfield(S, 'all_results') || isempty(S.all_results)
        warning('skipping %s — no all_results field', files(i).name);
        continue
    end
    rows = S.all_results;
    for j = 1:numel(rows)
        if isempty(all_results)
            all_results = rows(j);
        else
            all_results(end+1) = rows(j); %#ok<AGROW>
        end
    end
    fprintf('  + %s (%d row[s])\n', files(i).name, numel(rows));
end

save(out, 'all_results', '-v7.3');
fprintf('\nAggregated %d configs into %s\n', numel(all_results), out);
