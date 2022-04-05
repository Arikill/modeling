classdef pulse
    %PULSE Summary of this class goes here
    %   Detailed explanation goes here
    
    properties(Access=private)
        shape
        sampling_rate
        carrier_frequency
        duty_cycle
        rise_percent
        pulse_period
        state
    end
    
    methods
        function obj = pulse(nPulses, shape, sampling_rate, carrier_frequency, duty_cycle, pulse_rate, rise_percent)
            if nargin > 0
                obj(nPulses, 1) = obj;
                for k = 1: 1: nPulses
                    obj(k).shape = shape;
                    obj(k).sampling_rate = sampling_rate;
                    obj(k).carrier_frequency = carrier_frequency;
                    obj(k).duty_cycle = duty_cycle;
                    obj(k).pulse_period = 1/pulse_rate;
                    obj(k).rise_percent = rise_percent;
                    obj(k).state = zeros(1, floor(obj(k).pulse_period*obj(k).sampling_rate));
                end
            end
        end
        function obj = set_shape(obj, shape)
            for k = 1: 1: size(obj, 1)
               obj(k).shape = shape; 
            end
            obj = obj.generate_pulse();
        end
        function [y, t] = generate_stimulus(obj)
           obj = obj.generate_pulse();
           [y, t] = obj.get_stimulus();
        end
        function [y, t] = get_stimulus(obj)
            y = zeros(size(obj, 1), size(obj(1).state, 2));
            t = false(size(obj, 1), size(obj(1).state, 2));
            for k = 1: 1: size(obj, 1)
                y(k, :) = obj(k).state;
                t(k, 1) = true;
            end
            y = reshape(y',1,[]);
            t = reshape(t', 1, []);
            total_time = size(y, 2)/obj(1).sampling_rate;
            time_array = 0:(1/obj(1).sampling_rate):total_time-(1/obj(1).sampling_rate);
            t = t.*time_array;
        end
        function pulse_period = get_pulse_period(obj, k)
           pulse_period = obj(k).pulse_period; 
        end
        function sampling_rate = get_sampling_rate(obj, k)
            sampling_rate = obj(k).sampling_rate;
        end
    end
    methods(Access=private)
        function obj = generate_pulse(obj)
            for k = 1: 1: size(obj, 1)
                if strcmp(obj(k).shape, "triangular")
                    on_period = obj(k).pulse_period*obj(k).duty_cycle;
                    off_period = obj(k).pulse_period - on_period;
                    rise_time = on_period * obj(k).rise_percent/100;
                    fall_time = on_period - rise_time;
                    time = 0: (1/obj(k).sampling_rate):(rise_time - (1/obj(k).sampling_rate));
                    rise_pulse = (1/rise_time).*time;
                    time = rise_time: (1/obj(k).sampling_rate): on_period - (1/obj(k).sampling_rate);
                    fall_pulse = ((-1/fall_time).*(time-rise_time)) + 1;
                    obj(k).state = cat(2, rise_pulse, fall_pulse, zeros(1, floor(off_period*obj(k).sampling_rate)));
                    time = 0: (1/obj(k).sampling_rate): obj(k).pulse_period - (1/obj(k).sampling_rate);
                    obj(k).state = obj(k).state.*sin(2*pi*obj(k).carrier_frequency.*time);
                end
            end
        end
    end
end

