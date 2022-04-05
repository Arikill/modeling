%% Parameters
fs = 1e4;
response_duration = 1;
nPulses = 5;
carrier_frequency = 1000;

%% Stimulus Generation
speaker = pulse(nPulses, "triangular", fs, carrier_frequency, 0.5, 10, 80);
[stim, trigs] = speaker.generate_stimulus();

%% Synaptic Response Generation
response = synaptic_response(speaker, "excitatory", 0.05, 0.03, "summation", response_duration);
inputs = response.generate_response(trigs);
stim = cat(2, stim, zeros(1, size(inputs, 2) - size(stim, 2)));

%% Cellular Response Generation
neuron = cellular_response(0.01, [1; 0.5], fs);
neural_output = neuron.get_response(inputs);

%% Plot signals
time = 0: (1/fs): response_duration - (1/fs);
figure();
tiledlayout(3, 1);
ax1 = nexttile;
plot(time, stim, 'b');
ax2 = nexttile;
plot(time, inputs, 'b');
ax3 = nexttile;
plot(time, neural_output, 'k');
linkaxes([ax1, ax2, ax3], 'x');
