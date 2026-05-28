function lgd = legend_first_lines(n, labels, varargin)
%LEGEND_FIRST_LINES Legend for the first n line objects drawn in current axes.
% Use this when parameter curves are plotted before reference lines.

    ax = gca;
    children = ax.Children;
    child_types = get(children, 'Type');
    if ischar(child_types)
        child_types = {child_types};
    end
    line_handles = children(strcmp(child_types, 'line'));
    line_handles = flipud(line_handles(:));

    if numel(line_handles) < n
        error('legend_first_lines:NotEnoughLines', ...
              'Expected at least %d line objects, found %d.', n, numel(line_handles));
    end

    lgd = legend(ax, line_handles(1:n), labels, varargin{:});
    set(lgd, 'AutoUpdate', 'off');
end
