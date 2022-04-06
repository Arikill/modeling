import numpy as np

class timeseries:
    def __init__(self, Fs:float=1e4, tstart:float=0.0, tend:float=1.0, nbatches:int=1, precision=np.float32) -> None:
        self.Fs=Fs
        self.precision=precision
        self.timesteps = int((tend-tstart)*Fs)
        self.built = False
        self.values = np.zeros((nbatches, self.timesteps), dtype=self.precision)
        self.time = np.reshape(np.asarray(np.linspace(tstart, tend, self.timesteps), dtype=self.precision), (1, self.timesteps))
        pass

    def delay(self, t:float) -> None:
        delay_timesteps = int(np.abs(t)*self.Fs)
        if delay_timesteps > self.values.shape[-1]:
                delay_timesteps = self.values.shape[-1]
        if delay_timesteps > 0:
            self.values = np.concatenate([np.ones((self.values.shape[0], delay_timesteps)*self.values[:, 0], dtype=self.precision), self.values[:, :-delay_timesteps]], axis=-1)
        elif delay_timesteps < 0:
            self.values = np.concatenate([self.values[:, delay_timesteps:], np.ones((self.values.shape[0], delay_timesteps)*self.values[:, -1], dtype=self.precision)], axis=-1)
        pass

    

if __name__=="__main__":
    ts = timeseries(nbatches=10)
    ts.alpha(0, 0)