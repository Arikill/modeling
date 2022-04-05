classdef synaptic_response
    properties(Access=private)
        type
        delay
        time_constant
        integration
        state
        sampling_rate
        response_time
    end
    
    methods
        function obj = synaptic_response(pulse_obj, type, delay, time_constant, integration, response_time)
            if nargin > 0
               obj(size(pulse_obj, 1), 1) = obj;
               for k = 1: 1: size(obj, 1)
                   obj(k).type = type;
                   obj(k).delay = delay;
                   obj(k).time_constant = time_constant;
                   obj(k).integration = integration;
                   obj(k).sampling_rate = pulse_obj.get_sampling_rate(k);
                   obj(k).response_time = response_time;
                   obj(k).state = zeros(1, floor(obj(k).response_time*obj(k).sampling_rate));
               end
            end
        end
        
        function y = generate_response(obj, pulse_triggers)
           y = zeros(size(obj, 1), size(obj(1).state, 2));
           obj = obj.propagate(pulse_triggers);
           for k = 1: 1: size(obj, 1)
                y(k, :) = obj(k).state;
           end
           if strcmp(obj(1).integration, "summation")
               y = sum(y, 1);
           elseif strcmp(obj(1).integration, "maximum")
               if strcmp(obj(1).type, "inhibitory")
                    y = min(y, [], 1);
               else
                   y = max(y, [], 1);
               end
           end
        end
    end
    methods(Access=private)
        function obj = propagate(obj, pulse_triggers)
            trig_times = pulse_triggers(pulse_triggers~=false);
            trig_times = cat(2, 0, trig_times);
            for k = 1: 1: size(obj, 1)
                time = 0: (1/obj(k).sampling_rate): obj(k).response_time - (1/obj(k).sampling_rate);
                obj(k).delay = obj(k).delay + trig_times(k);
                exponent = (time-obj(k).delay)/obj(k).time_constant;
                obj(k).state = (time > obj(k).delay).*exponent.*exp(-1*exponent + 1);
                if strcmp(obj(k).type, "inhibitory")
                    obj(k).state = obj(k).state*-1;
                end
            end
        end
    end
end

