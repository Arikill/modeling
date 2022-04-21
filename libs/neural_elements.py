import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class Synapse:
    def __init__(self, amp:float = 1e-9, td:float = 0.01, tau:float = 0.01) -> None:
        self.amp = amp
        self.td = td
        self.tau = tau
        pass

    def __call__(self, t):
        exponent = tf.cast(tf.greater_equal(t, self.td), dtype=tf.float32)*(t - self.td)/self.tau
        return self.amp*exponent*tf.exp(-1*exponent + 1)

class Neuron:
    def __init__(self, Cm:float = 0.14e-12, Rin:float = 0.5e9, Er:float = -65e-3, Ee:float = -30e-3, Ei:float = -100e-3) -> None:
        self.Cm = Cm
        self.Rin = Rin
        self.Er = Er
        self.Ee = Ee
        self.Ei = Ei
        self.Vm = Er
        pass
    
    def set_synapses(self, ge:Synapse, gi:Synapse):
        self.ge = ge
        self.gi = gi
        pass

    @tf.function
    def __call__(self, Vm, ge, gi, Iinj, t, fs):
        delt = 1/fs
        ge = self.ge(t-delt)
        gi = self.gi(t-delt)
        return Vm + delt*(1/self.Cm)*(Iinj - ge*(Vm-self.Ee) - gi*(Vm-self.Ei) - (1/self.Rin)*(Vm-self.Er)), ge, gi

if __name__ == "__main__":
    batches = 10
    fs = 1e4
    tstart = 0.0
    tend = 1.0
    timesteps = int(fs*(tend-tstart))
    Cm = 0.01
    Rin = 1
    Er = 0
    Ee = 0.5
    Ei = -0.5
    t = np.reshape(np.asarray(np.linspace(tstart, tend, timesteps), dtype=np.float32), (1, timesteps)) + np.zeros((batches, timesteps), dtype=np.float32)
    Iinj = np.zeros(t.shape, dtype=t.dtype) + np.asarray(np.random.normal(size=(batches, 1), loc=0, scale=0.05), dtype=np.float32)*0
    Vm = Er+Rin*Iinj
    exe = np.zeros(t.shape, dtype=t.dtype)
    inh = np.zeros(t.shape, dtype=t.dtype)
    with tf.device("/CPU:0"):
        ge = Synapse(amp=1, td=0.02, tau=0.03)
        gi = Synapse(amp=-1, td=0.01, tau=0.03)
        neuron = Neuron(Cm, Rin, Er, Ee, Ei)
        neuron.set_synapses(ge, gi)
        for i in range(1, timesteps):
            Vm[:, i], exe[:, i], inh[:, i] = neuron(Vm[:, i-1], exe[:, i-1], inh[:, i-1], Iinj[:, i-1], t[:, i-1], fs)
    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(t[0, :], Vm[0, :], c='k')
    ax[1].plot(t[0, :], Iinj[0, :], c='m')
    ax[2].plot(t[0, :], exe[0, :],'r', t[0, :], inh[0, :],'b')
    plt.show()

