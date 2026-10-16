import numpy as np

class Q8:
    def __init__(self, global_id, node_global_ids, mesh, material, thickness=1.0):
        self.global_id = global_id
        self.node_global_ids = node_global_ids  # Lista con los IDs globales de los 8 nodos
        self.mesh = mesh
        self.material = material
        self.thickness = thickness
        
        # Recupera las instancias de las coordenadas nodales desde la malla
        self.nodes = [self.mesh.nodes[nid] for nid in self.node_global_ids]

    def shape_functions(self, xi, eta):
        
        N = np.array([
            0.25 * (1 - xi) * (1 - eta) * (-xi - eta - 1),  # N1
            0.25 * (1 + xi) * (1 - eta) * (xi - eta - 1),   # N2
            0.25 * (1 + xi) * (1 + eta) * (xi + eta - 1),   # N3
            0.25 * (1 - xi) * (1 + eta) * (-xi + eta - 1),  # N4
            0.5 * (1 - xi**2) * (1 - eta),                  # N5 (entre 1 y 2)
            0.5 * (1 + xi) * (1 - eta**2),                  # N6 (entre 2 y 3)
            0.5 * (1 - xi**2) * (1 + eta),                  # N7 (entre 3 y 4)
            0.5 * (1 - xi) * (1 - eta**2)                   # N8 (entre 4 y 1)
        ])

        # Derivadas respecto a xi (fila 0) y eta (fila 1) -> (2x8)
        dN_dxi_eta = np.array([
            [
                0.25 * (1 - eta) * (2*xi + eta),    # dN1/dxi
                0.25 * (1 - eta) * (2*xi - eta),    # dN2/dxi
                0.25 * (1 + eta) * (2*xi + eta),    # dN3/dxi
                0.25 * (1 + eta) * (2*xi - eta),    # dN4/dxi
                -xi * (1 - eta),                    # dN5/dxi
                0.5 * (1 - eta**2),                 # dN6/dxi
                -xi * (1 + eta),                    # dN7/dxi
                -0.5 * (1 - eta**2)                 # dN8/dxi
            ],
            [
                0.25 * (1 - xi) * (xi + 2*eta),     # dN1/deta
                0.25 * (1 + xi) * (-xi + 2*eta),    # dN2/deta
                0.25 * (1 + xi) * (xi + 2*eta),     # dN3/deta
                0.25 * (1 - xi) * (-xi + 2*eta),    # dN4/deta
                -0.5 * (1 - xi**2),                 # dN5/deta
                -eta * (1 + xi),                    # dN6/deta
                0.5 * (1 - xi**2),                  # dN7/deta
                -eta * (1 - xi)                     # dN8/deta
            ]
        ])
        return N, dN_dxi_eta

    def B_matrix(self, xi, eta):
        
        _, dN_dxi_eta = self.shape_functions(xi, eta)
        
        coords = np.array([[node.x, node.y] for node in self.nodes])  # (8x2)

        # Matriz Jacobiana (2x2)
        J = dN_dxi_eta @ coords
        detJ = np.linalg.det(J)
        
        if detJ <= 0:
            raise ValueError(f"Determinante del Jacobiano invalido ({detJ}) en elemento Q8 {self.global_id}.")

        invJ = np.linalg.inv(J)
        dN_dxy = invJ @ dN_dxi_eta  # (2x8)

        # Construccion de la matriz B (3x16)
        B = np.zeros((3, 16))
        for i in range(8):
            dN_dx = dN_dxy[0, i]
            dN_dy = dN_dxy[1, i]

            B[0, 2*i]     = dN_dx
            B[1, 2*i + 1] = dN_dy
            B[2, 2*i]     = dN_dy
            B[2, 2*i + 1] = dN_dx

        return B, detJ

    def stiffness_matrix(self):
        """Integra Ke (16x16) usando Cuadratura de Gauss 2x2."""
        D = self.material.D_matrix()
        Ke = np.zeros((16, 16))

        p = 1.0 / np.sqrt(3.0)
        gauss_points = [-p, p]
        weights = [1.0, 1.0]

        for xi, w_xi in zip(gauss_points, weights):
            for eta, w_eta in zip(gauss_points, weights):
                B, detJ = self.B_matrix(xi, eta)
                weight_total = w_xi * w_eta * detJ * self.thickness
                Ke += B.T @ D @ B * weight_total

        return Ke

    def global_dofs(self):
        """Mapea los grados de libertad globales del elemento Q8 (16 DOFs)."""
        dofs = []
        for nid in self.node_global_ids:
            dofs.extend(self.mesh.node_global_dofs(nid))
        return dofs