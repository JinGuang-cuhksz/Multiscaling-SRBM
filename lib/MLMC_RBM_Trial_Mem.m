function [result] = MLMC_RBM_Trial_Mem(M, drift, C, n, dim, weights, y0, R, T)
    % Memory-batched MLMC single-path simulator for an SRBM. The fine/coarse
    % Euler grids are advanced in batches to bound peak memory.
    %
    % Input parameters:
    %   M       - Current level, ranging from 0 to L-1
    %   drift   - Drift term, d-dimensional vector
    %   C       - Cholesky decomposition of covariance matrix, satisfying C'*C=SigmaMat
    %   n       - Step ratio between consecutive levels (n = 1/gamma, integer)
    %   dim     - Problem dimension
    %   weights - Empirical probability weights
    %   y0      - Initial position, d-dimensional vector
    %   R       - Reflection matrix, d×d matrix
    %   T       - Time range (assumed to be integer)
    %
    % Output parameters:
    %   result - Structure containing:
    %       .YM1T    - Position Y^(M+1) at time (M+1)T with fine grid
    %       .YMT     - Position Y^M at time MT with coarse grid
    %       .y0      - Initial position
    %       .weight  - Current weight
    %       .complexity - Computational complexity

    gamma = 1/n;

    % Batch size bounded by a memory budget, rounded down to a multiple of n.
    available_mem = 1.5e9;
    batch_size = floor((available_mem / (8 * dim * 3)) / n) * n;
    if batch_size == 0
        batch_size = n;   % process at least n steps
    end

    % For level 0, special handling
    if M == 0
        n_steps_fine = T * n^1;
        dt_fine = T / n_steps_fine;

        Y_fine = y0;

        for batch_start = 1:batch_size:n_steps_fine
            batch_end = min(batch_start + batch_size - 1, n_steps_fine);
            batch_size_actual = batch_end - batch_start + 1;

            dB_fine_batch = sqrt(dt_fine) * randn(dim, batch_size_actual);
            dX_fine_batch = drift * dt_fine + C' * dB_fine_batch;

            Y_fine = processBatch(Y_fine, dX_fine_batch, R);

            clear dB_fine_batch dX_fine_batch
        end

        complexity = dim * n_steps_fine;

        result = zeros(1, 3*dim+2);
        result(1:dim) = Y_fine';
        result(dim+1:2*dim) = y0';
        result(2*dim+1:3*dim) = y0';
        result(3*dim+1) = weights(M+1);
        result(3*dim+2) = complexity;
    else
        total_time = (M+1) * T;
        n_steps_fine = total_time * n^(M+1);
        n_steps_coarse = total_time * n^M;
        dt_fine = total_time / n_steps_fine;

        Y_fine = y0;              % tracks Y^(M+1)((M+1)T)
        Y_coarse = y0;            % tracks Y^M(MT)
        T_index_coarse = T * n^M; % coarse index of time T
        current_coarse_index = 0; % coarse index processed so far

        for batch_start = 1:batch_size:n_steps_fine
            batch_end = min(batch_start + batch_size - 1, n_steps_fine);

            dB_fine_batch = sqrt(dt_fine) * randn(dim, batch_end - batch_start + 1);
            dX_fine_batch = drift * dt_fine + C' * dB_fine_batch;

            Y_fine = processBatch(Y_fine, dX_fine_batch, R);

            % Aggregate fine increments into coarse increments.
            dX_fine_reshaped = reshape(dX_fine_batch, dim, n, []);
            dX_coarse_batch = squeeze(sum(dX_fine_reshaped, 2));

            if size(dX_coarse_batch, 2) == 1
                dX_coarse_batch = reshape(dX_coarse_batch, dim, 1);
            end

            next_coarse_index = current_coarse_index + size(dX_coarse_batch, 2);

            % Advance the coarse path only for indices past time T.
            if next_coarse_index > T_index_coarse
                start_process_idx = max(1, T_index_coarse - current_coarse_index + 1);

                if start_process_idx <= size(dX_coarse_batch, 2)
                    dX_coarse_process = dX_coarse_batch(:, start_process_idx:end);
                    Y_coarse = processBatch(Y_coarse, dX_coarse_process, R);
                end
            end

            current_coarse_index = next_coarse_index;

            clear dB_fine_batch dX_fine_batch dX_fine_reshaped dX_coarse_batch dX_coarse_process
        end

        complexity = dim * (n_steps_fine + n_steps_coarse);

        result = zeros(1, 3*dim+2);
        result(1:dim) = Y_fine';
        result(dim+1:2*dim) = Y_coarse';
        result(2*dim+1:3*dim) = y0';
        result(3*dim+1) = weights(M+1);
        result(3*dim+2) = complexity;
    end
end

function final_position = processBatch(initial_position, dX_batch, R)
    [~, n_steps] = size(dX_batch);
    current_position = initial_position;

    for k = 1:n_steps
        next_position = current_position + dX_batch(:, k);
        current_position = Skorokhod_linear(size(current_position, 1), next_position, R);
    end

    final_position = current_position;
end
