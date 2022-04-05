function y = Delay(x, delay_time, sampling_rate)
    delay_samples = floor(delay_time*sampling_rate);
    delay_vector = zeros(1, delay_samples);
    y = cat(2, delay_vector, x(:, 1:end-delay_samples));
end