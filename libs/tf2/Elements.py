import tensorflow as tf

class Neuron(tf.Module):
    def __init__(self, nSynapses) -> None:
        self.nSynapses = nSynapses
        self.built = False
        pass

    def build(self, input_shape):
        self.td = tf.Variable(tf.random.normal((input_shape[-1], self.nSynapses), mean=0.5, stddev=0.5, dtype=tf.float32))
        self.tau = tf.Variable(tf.random.normal((input_shape[-1], self.nSynapses), mean=0.5, stddev=0.5, dtype=tf.float32))
        self.amp = tf.Variable(tf.random.normal((input_shape[-1], self.nSynapses), dtype=tf.float32))
        self.built = True
        pass

    def __call__(self, inputs, time):
        if not self.built:
            self.build(inputs.shape)
        exponent = tf.cast(tf.greater(time, self.td), dtype=tf.float32)*(time-self.td)/self.tau
        return tf.tanh(tf.matmul(inputs, self.amp*exponent*tf.exp(-1*exponent + 1)))

class Network(tf.Module):
    def __init__(self, structure):
        self.structure = structure
        self.built = False
        pass

    def build(self, input_shape):
        self.pathway = [Neuron(nSynapses) for nSynapses in self.structure]
        inputs = tf.zeros(input_shape, dtype=tf.float32)
        for index, neuron in enumerate(self.pathway):
            if index == 0:
                outputs = neuron(inputs, 0.0)
            else:
                outputs = neuron(outputs, 0.0)
        self.built = True
        pass

    def __call__(self, inputs, time):
        if not self.built:
            self.build(inputs.shape)
        for index, neuron in enumerate(self.pathway):
            if index == 0:
                outputs = neuron(inputs, time)
            else:
                outputs = neuron(outputs, time)
        return outputs

class Container(tf.Module):
    def __init__(self, structure):
        self.network = Network(structure)
        self.cost = None
        self.built = False
        pass

    def build(self, input_shape, target_shape):
        nTimesteps = input_shape[1]
        inputs = tf.zeros(input_shape, dtype=tf.float32)
        targets = tf.zeros(target_shape, dtype=tf.float32)
        if targets.shape[1] == 1:
            for timestep in range(nTimesteps):
                self.network(inputs[:, timestep:timestep+1, :], 0.0)
        elif targets.shape[1] == inputs.shape[1]:
            outputs = [None for _ in range(nTimesteps)]
            for timestep in range(nTimesteps):
                outputs[timestep] = self.network(inputs[:, timestep:timestep+1, :], 0.0)
            tf.concat(outputs, axis=1)
        self.compute_cost(targets, outputs)
        self.built = True
        pass
    
    def compute_cost(self, targets, outputs):
        return tf.reduce_mean(tf.square(targets-outputs))

    @tf.function
    def __call__(self, inputs, targets, fs, tstart):
        if not self.built:
            self.build(inputs.shape)
        t = tstart
        nTimesteps = inputs.shape[-1]
        if targets.shape[1] == 1:
            for timestep in range(nTimesteps):
                outputs = self.network(inputs[:, timestep:timestep+1, :], t)
                t = t+(1/fs)
        elif targets.shape[1] == inputs.shape[1]:
            outputs = [None for _ in range(nTimesteps)]
            for timestep in range(nTimesteps):
                outputs[timestep] = self.network(inputs[:, timestep:timestep+1, :], t)
                t = t+(1/fs)
            tf.concat(outputs, axis=1)
        return self.compute_cost(targets, outputs)

if __name__ == "__main__":
    import numpy as np
    import time as time
    from Optimizer import NelderMead
    fs = 1e3
    tstart = 0.0
    tend = 0.1
    nTimesteps = int(fs*(tend-tstart))
    nBatches = 100000
    nInputs = 1
    inputs = np.asarray(np.random.normal(size=(nBatches, nTimesteps, nInputs)), dtype=np.float32)
    nOutputs = 1
    targets = np.asarray(np.random.normal(size=(nBatches, nTimesteps, nOutputs)), dtype=np.float32)
    structure = [4, 2, nOutputs]
    optimizer = NelderMead(soln_structure=structure, nSolutions= 4)
    cost = optimizer(inputs, targets, fs, tstart)
    # container = Container(structure)
    # container.build(inputs.shape, targets.shape)
    # with tf.device("/GPU:0"):
    #     startTime = time.time()
    #     for iter in range(100):
    #         _, cost = container(inputs, targets, fs, tstart)
    #     stopTime = time.time()
    # container.cost = cost
    # print("Elapsed Time on GPU: {}, cost: {}".format(stopTime-startTime, container.cost.numpy()))
    # with tf.device("/CPU:0"):
    #     startTime = time.time()
    #     for iter in range(100):
    #         _, cost = container(inputs, targets, fs, tstart)
    #     stopTime = time.time()
    # container.cost = cost
    # print("Elapsed Time on CPU: {}, cost: {}".format(stopTime-startTime, container.cost.numpy()))
    
    