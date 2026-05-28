% Mean approximation accuracy for the 2D SRBM with R = [1 -alpha; -beta 1].
% For fixed beta, varying alpha, with r in (beta, 1).

addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'lib'));

if ~exist('results', 'dir'); mkdir('results'); end

% Set flag for logarithmic x-axis
use_logscale = false;  % Set to true for logarithmic scale, false for linear

% Set beta value for this case
beta = 0.25;  % Fixed beta value (alpha in paper)

% Define alpha values to analyze
alpha_values = [1]; % beta in paper
num_alpha = length(alpha_values);

% Define colors for different alpha values
colors = [0, 0, 1;      % Blue
          0, 0.5, 0;    % Dark Green
          1, 0, 0;      % Red
          0.8, 0, 0.8;  % Purple
          1, 0.5, 0;    % Orange
          0, 0.7, 0.7]; % Cyan

% Number of r values to plot
n_points = 100;

% Create range of r values from (beta+epsilon) to (1-epsilon)
epsilon = 0.001;  % Small buffer to avoid numerical issues at boundaries
if use_logscale
    % Custom logspace between (beta+epsilon) and (1-epsilon)
    r_min = beta + epsilon;
    r_max = 1 - epsilon;
    r_ratio = r_max / r_min;
    r_values = r_min * (r_ratio).^((0:n_points-1)/(n_points-1));
else
    r_values = linspace(beta + epsilon, 1 - epsilon, n_points);
end

% Initialize arrays to store results for all alpha values
mean_x1_all = zeros(n_points, num_alpha);
mean_x2_scaled_all = zeros(n_points, num_alpha);
var_x1_all = zeros(n_points, num_alpha);
var_x2_scaled_all = zeros(n_points, num_alpha);
cor_all = zeros(n_points, num_alpha);

% Calculate statistics for each value of r and alpha
for a = 1:num_alpha
    curr_alpha = alpha_values(a);
    R_i = [1, -curr_alpha; -beta, 1];

    for i = 1:n_points
        r_i = r_values(i);
        delta_i = [1; r_i];
        mu_i = -R_i * delta_i;

        [mean_x1_i, mean_x2_i, var_x1_i, var_x2_i, cor_x1x2_i] = rbm_moments(mu_i(1), mu_i(2), curr_alpha, beta);

        mean_x1_all(i, a) = mean_x1_i;
        mean_x2_scaled_all(i, a) = r_i * mean_x2_i;
        var_x1_all(i, a) = var_x1_i;
        var_x2_scaled_all(i, a) = r_i^2 * var_x2_i;
        cor_all(i, a) = cor_x1x2_i;
    end
end

% Create a single figure with a 3x2 subplot layout
figure;
set(gcf, 'Position', [100, 100, 1200, 800]);  % Adjust figure size

% Subplot 1: mean_x1
subplot(3, 2, 1);
for a = 1:num_alpha
    if use_logscale
        semilogx(r_values, mean_x1_all(:, a), 'Color', colors(a,:), 'LineWidth', 2);
    else
        plot(r_values, mean_x1_all(:, a), 'Color', colors(a,:), 'LineWidth', 2);
    end
    hold on;
end
% Plot theoretical reference line at y=0.5 for station 1
if use_logscale
    semilogx([min(r_values), max(r_values)], [0.5, 0.5], 'r--', 'LineWidth', 1.5);
else
    plot([min(r_values), max(r_values)], [0.5, 0.5], 'r--', 'LineWidth', 1.5);
end
hold off;
xlabel('r');
ylabel('Mean Value');
title('\delta_1*mean\_x1 vs r (theoretical value = 0.5)');
grid on;
xlim([0, 1]);  % Set x-axis limits to (0,1)
legend_first_lines(num_alpha, arrayfun(@(x) "\alpha = " + x, alpha_values, 'UniformOutput', false), 'Location', 'best');

% Subplot 2: mean_x2*r
subplot(3, 2, 2);
for a = 1:num_alpha
    if use_logscale
        semilogx(r_values, mean_x2_scaled_all(:, a), 'Color', colors(a,:), 'LineWidth', 2);
    else
        plot(r_values, mean_x2_scaled_all(:, a), 'Color', colors(a,:), 'LineWidth', 2);
    end
    hold on;
end
% Plot theoretical reference lines for each alpha value
for a = 1:num_alpha
    curr_alpha = alpha_values(a);
    theoretical_mean = (1 + beta^2)/(2*(1 - curr_alpha*beta));

    if use_logscale
        semilogx([min(r_values), max(r_values)], [theoretical_mean, theoretical_mean], '--', 'Color', colors(a,:), 'LineWidth', 1);
    else
        plot([min(r_values), max(r_values)], [theoretical_mean, theoretical_mean], '--', 'Color', colors(a,:), 'LineWidth', 1);
    end
end
hold off;
xlabel('r');
ylabel('Scaled Mean Value');
title('\delta_2*mean\_x2 vs r (with theoretical values)');
grid on;
xlim([0, 1]);  % Set x-axis limits to (0,1)
legend_first_lines(num_alpha, arrayfun(@(x) "\alpha = " + x, alpha_values, 'UniformOutput', false), 'Location', 'best');

% Subplot 3: var_x1
subplot(3, 2, 3);
for a = 1:num_alpha
    if use_logscale
        semilogx(r_values, var_x1_all(:, a), 'Color', colors(a,:), 'LineWidth', 2);
    else
        plot(r_values, var_x1_all(:, a), 'Color', colors(a,:), 'LineWidth', 2);
    end
    hold on;
end
% Plot theoretical reference line at y=0.25 for station 1
if use_logscale
    semilogx([min(r_values), max(r_values)], [0.25, 0.25], 'r--', 'LineWidth', 1.5);
else
    plot([min(r_values), max(r_values)], [0.25, 0.25], 'r--', 'LineWidth', 1.5);
end
hold off;
xlabel('r');
ylabel('Variance');
title('\delta_1*var\_x1 vs r (theoretical value = 0.25)');
grid on;
xlim([0, 1]);  % Set x-axis limits to (0,1)
legend_first_lines(num_alpha, arrayfun(@(x) "\alpha = " + x, alpha_values, 'UniformOutput', false), 'Location', 'best');

% Subplot 4: var_x2*r^2
subplot(3, 2, 4);
for a = 1:num_alpha
    if use_logscale
        semilogx(r_values, var_x2_scaled_all(:, a), 'Color', colors(a,:), 'LineWidth', 2);
    else
        plot(r_values, var_x2_scaled_all(:, a), 'Color', colors(a,:), 'LineWidth', 2);
    end
    hold on;
end
% Plot theoretical reference lines for each alpha value
for a = 1:num_alpha
    curr_alpha = alpha_values(a);
    theoretical_variance = (1 + beta^2)^2/(4*(1 - curr_alpha*beta)^2);

    if use_logscale
        semilogx([min(r_values), max(r_values)], [theoretical_variance, theoretical_variance], '--', 'Color', colors(a,:), 'LineWidth', 1);
    else
        plot([min(r_values), max(r_values)], [theoretical_variance, theoretical_variance], '--', 'Color', colors(a,:), 'LineWidth', 1);
    end
end
hold off;
xlabel('r');
ylabel('Scaled Variance');
title('\delta_2*var\_x2 vs r (with theoretical values)');
grid on;
xlim([0, 1]);  % Set x-axis limits to (0,1)
legend_first_lines(num_alpha, arrayfun(@(x) "\alpha = " + x, alpha_values, 'UniformOutput', false), 'Location', 'best');

% Subplot 5: Correlation
subplot(3, 2, 5);
for a = 1:num_alpha
    if use_logscale
        semilogx(r_values, cor_all(:, a), 'Color', colors(a,:), 'LineWidth', 2);
    else
        plot(r_values, cor_all(:, a), 'Color', colors(a,:), 'LineWidth', 2);
    end
    hold on;
end
% Plot reference line at y=0
if use_logscale
    semilogx([min(r_values), max(r_values)], [0, 0], 'r--', 'LineWidth', 1.5);
else
    plot([min(r_values), max(r_values)], [0, 0], 'r--', 'LineWidth', 1.5);
end
hold off;
xlabel('r');
ylabel('Correlation');
title('cor\_x1x2 vs r');
grid on;
xlim([0, 1]);  % Set x-axis limits to (0,1)
legend_first_lines(num_alpha, arrayfun(@(x) "\alpha = " + x, alpha_values, 'UniformOutput', false), 'Location', 'best');

% Add information about theoretical values in a text box
subplot(3, 2, 6);
axis off;
text_info = {
    'Theoretical Values:'
    ''
    ['Mean of station 1 = 0.5']
    ['Mean of station 2 = (1+\beta^2)/(2(1-\alpha\beta))']
    ''
    ['Variance of station 1 = 0.25']
    ['Variance of station 2 = (1+\beta^2)^2/(4(1-\alpha\beta)^2)']
    ''
    ['For \beta = ' num2str(beta)]
};
text(0.05, 0.5, text_info, 'FontSize', 12, 'VerticalAlignment', 'middle');

% Add a main title
if use_logscale
    scale_type = 'Log';
else
    scale_type = 'Linear';
end
sgtitle(['RBM Statistics vs r for Different \alpha Values (\beta = ', num2str(beta), ', ', scale_type, ' Scale)'], 'FontSize', 14);

% Save the figure
if use_logscale
    scale_suffix = 'log';
else
    scale_suffix = 'linear';
end
saveas(gcf, ['results/rbm_statistics_fixed_beta_', num2str(beta), '_', scale_suffix, '.png']);
close;

% Create a new figure for mean_x2 only with reversed x-axis
figure;
set(gcf, 'Position', [100, 100, 800, 600]);  % Adjust figure size

% Reversed r grid (r near 1 on the left, near 0 on the right)
r_values_reversed = flip(r_values);
mean_x2_scaled_all_reversed = flip(mean_x2_scaled_all, 1);

% Line styles and markers for black-and-white printing
line_styles = {'-', '--', ':'};
markers = {'o', 's', '^'};
marker_spacing = 10; % place a marker every 10 points

% Plot mean_x2 with reversed x-axis
for a = 1:num_alpha
    if use_logscale
        semilogx(r_values_reversed, mean_x2_scaled_all_reversed(:, a), 'Color', colors(a,:), 'LineStyle', line_styles{a}, 'LineWidth', 2);
    else
        plot(r_values_reversed, mean_x2_scaled_all_reversed(:, a), 'Color', colors(a,:), 'LineStyle', line_styles{a}, 'LineWidth', 2);
    end
    hold on;
end

% Theoretical reference lines, from 0 to 1
for a = 1:num_alpha
    curr_alpha = alpha_values(a);
    theoretical_mean = (1 + beta^2)/(2*(1 - curr_alpha*beta));

    if use_logscale
        semilogx([0, 1], [theoretical_mean, theoretical_mean], 'Color', colors(a,:), 'LineStyle', line_styles{a}, 'LineWidth', 1.5);
    else
        plot([0, 1], [theoretical_mean, theoretical_mean], 'Color', colors(a,:), 'LineStyle', line_styles{a}, 'LineWidth', 1.5);
    end

    % Annotate the theoretical value near the left edge
    text(0.05, theoretical_mean*1.02, num2str(theoretical_mean, '%.3f'), 'Color', colors(a,:), 'FontSize', 12, 'VerticalAlignment', 'bottom');
end

% Add the 0.5 line (skew-symmetric prediction)
plot([0, 1], [0.5, 0.5], 'k-.', 'LineWidth', 1.5);
text(0.15, 0.51, 'skew symmetric', 'FontSize', 12, 'VerticalAlignment', 'bottom');

hold off;
xlabel('r');
ylabel('E[r^2Z_2^{(r)}]');
grid on;
legend_text = arrayfun(@(x) ["\alpha = " + x], alpha_values, 'UniformOutput', false);
legend_first_lines(num_alpha, legend_text, 'Location', 'best');
set(gca, 'XDir', 'reverse');  % x-axis reversed
xlim([0, 1]);  % x range (axis reversed)
ylim([0.45, 0.75]);  % y limits

% Save figure
saveas(gcf, ['results/rbm_mean_x2_reversed_fixed_beta_', num2str(beta), '_', scale_suffix, '.png']);
close;

% Black-and-white version for printing
figure;
set(gcf, 'Position', [100, 100, 800, 600]);  % Adjust figure size

% B/W version: distinguish series by line style
for a = 1:num_alpha
    if use_logscale
        semilogx(r_values_reversed, mean_x2_scaled_all_reversed(:, a), 'Color', 'k', 'LineStyle', line_styles{a}, 'LineWidth', 2);
    else
        plot(r_values_reversed, mean_x2_scaled_all_reversed(:, a), 'Color', 'k', 'LineStyle', line_styles{a}, 'LineWidth', 2);
    end
    hold on;
end

% Theoretical reference lines, from 0 to 1 (b/w)
for a = 1:num_alpha
    curr_alpha = alpha_values(a);
    theoretical_mean = (1 + beta^2)/(2*(1 - curr_alpha*beta));

    if use_logscale
        semilogx([0, 1], [theoretical_mean, theoretical_mean], 'Color', 'k', 'LineStyle', line_styles{a}, 'LineWidth', 1.5);
    else
        plot([0, 1], [theoretical_mean, theoretical_mean], 'Color', 'k', 'LineStyle', line_styles{a}, 'LineWidth', 1.5);
    end

    % Annotate the theoretical value near the left edge
    text(0.05, theoretical_mean*1.02, num2str(theoretical_mean, '%.3f'), 'Color', 'k', 'FontSize', 12, 'VerticalAlignment', 'bottom');
end

% Add the 0.5 line (skew-symmetric prediction)
plot([0, 1], [0.5, 0.5], 'k-.', 'LineWidth', 2);
text(0.15, 0.51, 'skew symmetric', 'FontSize', 12, 'VerticalAlignment', 'bottom');

hold off;
xlabel('r');
ylabel('E[r^2Z_2^{(r)}]');
grid on;
legend_text = arrayfun(@(x) ["\alpha = " + x], alpha_values, 'UniformOutput', false);
legend_first_lines(num_alpha, legend_text, 'Location', 'best');
set(gca, 'XDir', 'reverse');  % x-axis reversed
xlim([0, 1]);  % x range (axis reversed)
ylim([0.45, 0.75]);  % y limits

% Save b/w figure
saveas(gcf, ['results/rbm_mean_x2_reversed_fixed_beta_', num2str(beta), '_', scale_suffix, '_bw.png']);
close;

% Create a new figure for relative error with reversed x-axis
figure('Position', [150, 150, 550, 280]); % wide, short figure

% Reversed r grid (r near 1 on the left, near beta on the right)
r_values_reversed = flip(r_values);
mean_x2_scaled_all_reversed = flip(mean_x2_scaled_all, 1);

% Colors and line styles
multi_color = '#FFA500';   % used for multi-scaling
skew_color = '#00008B';    % used for skew symmetric
line_styles = {'-', '--', '-.'};  % line style per alpha

% Compute and plot relative errors
for a = 1:num_alpha
    curr_alpha = alpha_values(a);
    theoretical_mean = (1 + beta^2)/(2*(1 - curr_alpha*beta));

    % Exact value from rbm_moments
    true_values = mean_x2_scaled_all_reversed(:, a);

    % Multi-scaling relative error
    rel_error_multi = abs(theoretical_mean - true_values) ./ true_values;

    % Skew-symmetric relative error
    rel_error_skew = abs(0.5 - true_values) ./ true_values;

    % Plot multi-scaling relative error
    if use_logscale
        semilogx(r_values_reversed, rel_error_multi, 'Color', multi_color, 'LineStyle', line_styles{1}, 'LineWidth', 2);
    else
        plot(r_values_reversed, rel_error_multi, 'Color', multi_color, 'LineStyle', line_styles{1}, 'LineWidth', 2);
    end
    hold on;

    % Plot skew-symmetric relative error
    if use_logscale
        semilogx(r_values_reversed, rel_error_skew, 'Color', skew_color, 'LineStyle', line_styles{1}, 'LineWidth', 2);
    else
        plot(r_values_reversed, rel_error_skew, 'Color', skew_color, 'LineStyle', line_styles{2}, 'LineWidth', 2);
    end
end

hold off;
xlabel('r', 'FontSize', 18, 'FontWeight', 'bold');
ylabel('Relative Error', 'FontSize', 18, 'FontWeight', 'bold');
grid on;

% Legend
legend_text = ["Multi-scaling", "Skew symmetric"];
legend_first_lines(numel(legend_text), legend_text, 'Location', 'northwest');

set(gca, 'XDir', 'reverse');  % x-axis reversed
xlim([beta, 1]);  % x range from 1 to beta (axis reversed)
ylim([0, 0.6]);

% Reduce surrounding whitespace
set(gca, 'FontSize', 13); % larger tick font
% Trim PDF margins
ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset;
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax.Position = [left bottom ax_width ax_height];
fig = gcf;
fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
print(gcf, ['results/rbm_relative_error_fixed_beta_', num2str(beta), '_', num2str(alpha_values(1)), '_',  scale_suffix, '.pdf'], '-dpdf', '-bestfit');
saveas(gcf, ['results/rbm_relative_error_fixed_beta_', num2str(beta), '_', num2str(alpha_values(1)), '_',  scale_suffix, '.png']);
saveas(gcf, ['results/rbm_relative_error_fixed_beta_', num2str(beta), '_', num2str(alpha_values(1)), '_',  scale_suffix, '.fig']);
close;
