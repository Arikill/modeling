import tensorflow as tf
from Elements import Container

class NelderMead:
    def __init__(self, soln_structure, nSolutions=4, alpha=1, gamma=2, rho=0.5, sigma=0.5):
        self.structure = soln_structure
        self.nSolutions = nSolutions
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.sigma = sigma
        self.built = False
        pass

    def build(self, input_shape, target_shape):
        self.solutions = [Container(self.structure) for _ in range(self.nSolutions)]
        for solution in self.solutions:
            solution.build(input_shape, target_shape)
        self.centroid = Container(self.structure)
        self.centroid.build(input_shape, target_shape)
        self.reflection = Container(self.structure)
        self.reflection.build(input_shape, target_shape)
        self.contraction = Container(self.structure)
        self.contraction.build(input_shape, target_shape)
        self.expansion = Container(self.structure)
        self.expansion.build(input_shape, target_shape)
        self.built = True
        pass

    def center(self):
        for index, _ in enumerate(self.centroid.network.structure):
            self.centroid.network.pathway[index].td = self.centroid.network.pathway[index].td*0
            self.centroid.network.pathway[index].tau = self.centroid.network.pathway[index].tau*0
            self.centroid.network.pathway[index].amp = self.centroid.network.pathway[index].amp*0
        for n in range(self.nSolutions-1):
            for index, _ in enumerate(self.centroid.network.structure):
                self.centroid.network.pathway[index].td = self.centroid.network.pathway[index].td + self.solutions[n].network.pathway[index].td
                self.centroid.network.pathway[index].tau = self.centroid.network.pathway[index].tau + self.solutions[n].network.pathway[index].tau
                self.centroid.network.pathway[index].amp = self.centroid.network.pathway[index].amp + self.solutions[n].network.pathway[index].amp
        for index, _ in enumerate(self.centroid.network.structure):
            self.centroid.network.pathway[index].td = self.centroid.network.pathway[index].td/(self.nSolutions-1)
            self.centroid.network.pathway[index].tau = self.centroid.network.pathway[index].tau/(self.nSolutions-1)
            self.centroid.network.pathway[index].amp = self.centroid.network.pathway[index].amp/(self.nSolutions-1)
        pass

    def reflect(self, solution):
        for index, _ in enumerate(self.reflection.network.structure):
            self.reflection.network.pathway[index].td = self.centroid.network.pathway[index].td + self.alpha*(self.centroid.network.pathway[index].td - solution.network.pathway[index].td)
            self.reflection.network.pathway[index].tau = self.centroid.network.pathway[index].tau + self.alpha*(self.centroid.network.pathway[index].tau - solution.network.pathway[index].tau)
            self.reflection.network.pathway[index].amp = self.centroid.network.pathway[index].amp + self.alpha*(self.centroid.network.pathway[index].amp - solution.network.pathway[index].amp)
        pass

    def contract(self, solution):
        for index, _ in enumerate(self.reflection.network.structure):
            self.contraction.network.pathway[index].td = self.centroid.network.pathway[index].td - self.rho*(self.centroid.network.pathway[index].td - solution.network.pathway[index].td)
            self.contraction.network.pathway[index].tau = self.centroid.network.pathway[index].tau - self.rho*(self.centroid.network.pathway[index].tau - solution.network.pathway[index].tau)
            self.contraction.network.pathway[index].amp = self.centroid.network.pathway[index].amp - self.rho*(self.centroid.network.pathway[index].amp - solution.network.pathway[index].amp)
        pass

    def expand(self):
        for index, _ in enumerate(self.expansion.network.structure):
            self.expansion.network.pathway[index].td = self.centroid.network.pathway[index].td + self.gamma*(self.reflection.network.pathway[index].td - self.centroid.network.pathway[index].td)
            self.expansion.network.pathway[index].tau = self.centroid.network.pathway[index].tau + self.gamma*(self.reflection.network.pathway[index].tau - self.centroid.network.pathway[index].tau)
            self.expansion.network.pathway[index].amp = self.centroid.network.pathway[index].amp + self.gamma*(self.reflection.network.pathway[index].amp - self.centroid.network.pathway[index].amp)
        pass

    def shrink(self):
        for solution in self.solutions[1:]:
            for index, _ in enumerate(solution.network.structure):
                solution.network.pathway[index].td = self.solutions[0].network.pathway[index].td + self.sigma*(solution.network.pathway[index].td - self.solutions[0].network.pathway[index].td)
                solution.network.pathway[index].tau = self.solutions[0].network.pathway[index].tau + self.sigma*(solution.network.pathway[index].tau - self.solutions[0].network.pathway[index].tau)
                solution.network.pathway[index].amp = self.solutions[0].network.pathway[index].amp + self.sigma*(solution.network.pathway[index].amp - self.solutions[0].network.pathway[index].amp)
        pass

    def copy(self, s1, s2):
        for index, _ in enumerate(s1.network.structure):
            s1.network.pathway[index].td = s2.network.pathway[index].td
            s1.network.pathway[index].tau = s2.network.pathway[index].tau
            s1.network.pathway[index].amp = s2.network.pathway[index].amp
            s1.cost = s2.cost
        pass
    
    def sort_solutions(self):
        print(self.solutions)
        self.solutions.sort(key=lambda x: x.cost)
        pass

    def compute_all_costs(self, inputs, targets, fs, tstart):
        self.compute_solution_costs(inputs, targets, fs, tstart)
        self.reflection.cost = self.reflection(inputs, targets, fs, tstart)
        self.contraction.cost = self.contraction(inputs, targets, fs, tstart)
        self.expansion.cost = self.expansion(inputs, targets, fs, tstart)
        pass

    def compute_solution_costs(self, inputs, targets, fs, tstart):
        for solution in self.solutions:
            solution.cost = solution(inputs, targets, fs, tstart)
        pass

    # @tf.function
    def __call__(self, inputs, targets, fs, tstart):
        if not self.built:
            self.build(inputs.shape, targets.shape)
            self.compute_all_costs(inputs, targets, fs, tstart)
        self.center()
        self.reflect(self.solutions[-1])
        self.reflection.cost = self.reflection(inputs, targets, fs, tstart)
        if self.solutions[0].cost <= self.reflection.cost and self.reflection.cost < self.solutions[-2].cost:
            self.copy(self.solutions[-1], self.reflection)
        elif self.reflection.cost < self.solutions[0].cost:
            self.expand()
            self.expansion.cost = self.expansion(inputs, targets, fs, tstart)
            if self.expansion.cost < self.reflection.cost:
                self.copy(self.solutions[-1], self.expansion)
            else:
                self.copy(self.solutions[-1], self.reflection)
        elif self.reflection.cost < self.solutions[-1].cost:
            self.contract(self.reflection)
            self.contraction.cost = self.contraction(inputs, targets, fs, tstart)
            if self.contraction.cost < self.reflection.cost:
                self.copy(self.solutions[-1], self.contraction)
            else:
                self.shrink()
        elif self.reflection.cost >= self.solutions[-1].cost:
            self.contract(self.solutions[-1])
            self.contraction.cost = self.contraction(inputs, targets, fs, tstart)
            if self.contraction.cost < self.solutions[-1].cost:
                self.copy(self.solutions[-1], self.contraction)
            else:
                self.shrink()
        self.compute_solution_costs(inputs, targets, fs, tstart)
        self.sort_solutions()
        return self.solutions[0].cost
