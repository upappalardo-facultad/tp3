import numpy as np

class LinearElastic:
    def __init__(self, E: float, nu: float):
        self.E = E
        self.nu = nu

    def C_matrix(self, plane_state: str = "plane_strain") -> np.ndarray:
        """
        Retorna la matriz constitutiva C (3x3).
        plane_state: 'plane_strain' (por defecto) o 'plane_stress'
        """
        E, nu = self.E, self.nu

        if plane_state.lower() == "plane_strain":
            # Estado plano de deformación (Caso de estudio: Presa)
            factor = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
            return factor * np.array([
                [1.0 - nu,     nu,     0.0],
                [    nu,   1.0 - nu,   0.0],
                [   0.0,      0.0,   (1.0 - 2.0 * nu) / 2.0]
            ])

        elif plane_state.lower() == "plane_stress":
            # Estado plano de tensión
            factor = E / (1.0 - nu**2)
            return factor * np.array([
                [ 1.0,   nu, 0.0],
                [  nu,  1.0, 0.0],
                [ 0.0,  0.0, (1.0 - nu) / 2.0]
            ])

        else:
            raise ValueError(f"Estado desconocido: {plane_state}. Usar 'plane_strain' o 'plane_stress'.")