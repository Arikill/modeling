classdef cellular_response
    %UNTITLED3 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties(Access=private)
        time_constant
        weights;
        sampling_rate
    end
    
    methods
        function obj = cellular_response(time_constant, weights, sampling_rate)
            %UNTITLED3 Construct an instance of this class
            %   Detailed explanation goes here
            obj.time_constant = time_constant;
            obj.weights = weights;
            obj.sampling_rate = sampling_rate;
        end
        
        function y = get_response(obj, inputs)
            y = obj.propagate(inputs);
        end
    end
    methods(Access=private)
        function y = propagate(obj, inputs)
            y = sum(inputs.*obj.weights, 1);
            y = smoothing(y, obj.time_constant, obj.sampling_rate);
        end
    end
end

