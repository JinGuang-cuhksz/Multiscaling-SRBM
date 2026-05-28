% make_figure.m — "Warm Start for MLMC Simulation" figure.
%
% Reproduces the 1x3 panel figure (sec3_3_warmstart_3panel.pdf): the MLMC
% estimate of E[Z_k^{(0.2)}] for k = 1, 2, 3 (left-to-right panels) plotted
% against computational complexity (total discretization steps, log x-axis),
% for the three initial distributions
%       origin / skew symmetric / multi-scaling
% sweeping the number of levels L in {2, 4, 6, 8} at fixed T = 5000,
% gamma = 0.2. Shaded bands are 95% confidence intervals from Nest = 50
% independent estimators. Dashed grey lines are the MLMC reference means.
%
% Input:
%   results/data/all_results.mat   (run generate_data then
%                                    aggregate to build it)
% Output:
%   results/data/sec3_3_warmstart_3panel.{pdf,png}
%
% Run with the working directory set to this experiment folder.

addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'lib'));

in_file = 'results/data/all_results.mat';
if ~exist(in_file, 'file')
    error(['%s not found. Generate the warm-start sweep with ', ...
           'generate_data (array mode) and combine the per-config ', ...
           'files with aggregate first.'], in_file);
end
S = load(in_file);
all_results = S.all_results;

%% ---------- figure configuration (paper warm-start setting) ----------
T_plot     = 5000;                    % base simulation horizon
gamma_plot = 0.2;                     % discretization base
d          = 3;                       % dimensions / panels
ref_means  = [2.75, 24.86, 166.06];   % MLMC reference (L=8, T=1e5 run)

inits        = {'origin', 'skew', 'multi_scaling'};
init_labels  = {'origin', 'skew symmetric', 'multi-scaling'};
init_colors  = [0.15 0.15 0.15;       % origin        - near black
                0.12 0.47 0.71;       % skew          - blue  (tab:blue)
                0.84 0.15 0.16];      % multi-scaling - red   (tab:red)
init_markers = {'o', 's', 'd'};       % circle / square / diamond
ref_color    = [0.50 0.50 0.50];

%% ---------- assemble figure ----------
figure('Name', 'Warm start for MLMC', 'Position', [100 100 1300 380], 'Color', 'w');
tiledlayout(1, d, 'Padding', 'compact', 'TileSpacing', 'compact');

for k = 1:d
    ax = nexttile(k);
    hold(ax, 'on');
    series_handles = gobjects(1, numel(inits));

    for ci = 1:numel(inits)
        xs = []; ys = []; lo = []; hi = [];
        for i = 1:numel(all_results)
            r = all_results(i);
            if ~strcmp(r.init, inits{ci}),             continue; end
            if r.T ~= T_plot || r.gamma ~= gamma_plot, continue; end
            m_k   = r.means(:, k);
            mbar  = mean(m_k);
            halfw = 1.96 * std(m_k) / sqrt(numel(m_k));
            xs(end+1) = mean(r.complexity); %#ok<AGROW>
            ys(end+1) = mbar;               %#ok<AGROW>
            lo(end+1) = mbar - halfw;       %#ok<AGROW>
            hi(end+1) = mbar + halfw;       %#ok<AGROW>
        end
        if isempty(xs)
            warning('No data for init=%s at T=%d, gamma=%g.', ...
                    inits{ci}, T_plot, gamma_plot);
            continue
        end
        [xs, ord] = sort(xs);
        ys = ys(ord); lo = lo(ord); hi = hi(ord);

        col = init_colors(ci, :);
        % 95% CI band (drawn first, hidden from legend)
        fill(ax, [xs, fliplr(xs)], [lo, fliplr(hi)], col, ...
             'FaceAlpha', 0.18, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        series_handles(ci) = plot(ax, xs, ys, ['-' init_markers{ci}], ...
             'Color', col, 'MarkerFaceColor', col, 'MarkerEdgeColor', col, ...
             'LineWidth', 1.5, 'MarkerSize', 6);
    end

    set(ax, 'XScale', 'log');
    box(ax, 'on');
    xl = xlim(ax); xlim(ax, xl);          % freeze x-limits before annotating

    % MLMC reference mean for this dimension, with value label on the line
    yline(ax, ref_means(k), '--', 'Color', ref_color, 'LineWidth', 1.0);
    text(ax, xl(1), ref_means(k), ['  ' num2str(ref_means(k))], ...
         'Color', [0.40 0.40 0.40], 'FontWeight', 'bold', 'FontSize', 9, ...
         'VerticalAlignment', 'top', 'HorizontalAlignment', 'left');

    xlabel(ax, 'Complexity');
    if k == 1
        ylabel(ax, 'MLMC estimate');
    end
    title(ax, sprintf('E[Z_%d^{(0.2)}]', k), 'FontWeight', 'normal');

    if k == 1
        valid = isgraphics(series_handles);
        legend(ax, series_handles(valid), init_labels(valid), ...
               'Location', 'southeast', 'Box', 'on');
    end
end

%% ---------- save ----------
out_pdf = 'results/data/sec3_3_warmstart_3panel.pdf';
out_png = 'results/data/sec3_3_warmstart_3panel.png';
exportgraphics(gcf, out_pdf, 'ContentType', 'vector');
exportgraphics(gcf, out_png, 'Resolution', 200);
fprintf('Wrote %s\n      %s\n', out_pdf, out_png);
