import numpy as np

class LinearElastic:
    def __init__(self, E, nu):
        self.E = E
        self.nu = nu

    def C_matrix(self):
        """Retorna la matriz constitutiva C (3x3) para estado plano de tension."""
        factor = self.E / (1.0 - self.nu**2)
        return factor * np.array([
            [1.0,     self.nu, 0.0],
            [self.nu, 1.0,     0.0],
            [0.0,     0.0,     (1.0 - self.nu) / 2.0]
        ])
