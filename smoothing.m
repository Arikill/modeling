function [Y] = smoothing(X, Fs, Ts)
    [R, C] = size(X);
    Y = zeros(R, C);
    if (R > C)
        for c = 1: 1: C
            Y_length = R;
            append_samples_length = floor(Fs*Ts);
            append_start_samples = ones(append_samples_length, 1)*X(1, c);
            append_end_samples = ones(append_samples_length, 1)*X(R, c);
            Y1 = [append_start_samples; X(:, c); append_end_samples];
            for n = (append_samples_length+1): 1: (Y_length+append_samples_length)
                m = n-append_samples_length;
                Y(m, c) = sum(Y1(n-append_samples_length:n+append_samples_length))/(2*append_samples_length + 1);
            end
            Y = Delay(Y, Ts, Fs);
        end
    else
        for r = 1: 1: R
            Y_length = C;
            append_samples_length = floor(Fs*Ts);
            append_start_samples = ones(1, append_samples_length)*X(r, 1);
            append_end_samples = ones(1, append_samples_length)*X(r, C);
            Y1 = [append_start_samples, X(r, :), append_end_samples];
            for n = (append_samples_length+1): 1: (Y_length+append_samples_length)
                m = n-append_samples_length;
                Y(r, m) = sum(Y1(n-append_samples_length:n+append_samples_length))/(2*append_samples_length + 1);
            end
            Y = Delay(Y, Ts, Fs);
        end
    end
end
% Y_length = size(Y, 1);
% append_samples_length = floor(Fs*Ts);
% append_start_samples = ones(append_samples_length, 1)*X(1);
% append_end_samples = ones(append_samples_length, 1)*X(length(X));
% Y1 = [append_start_samples; X; append_end_samples];
% for n = (append_samples_length+1): 1: (Y_length+append_samples_length)
%     m = n-append_samples_length;
%     Y(m) = sum(Y1(n-append_samples_length:n+append_samples_length))/(2*append_samples_length + 1);
% end