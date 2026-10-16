import numpy as np

class T6:
    def __init__(self, global_id, node_global_ids, mesh, material, thickness=1.0):
        self.global_id = global_id
        self.node_global_ids = node_global_ids  # Lista con los IDs globales de los 6 nodos
        self.mesh = mesh
        self.material = material
        self.thickness = thickness
        
        # Recupera las instancias de las coordenadas nodales desde la malla
        self.nodes = [self.mesh.nodes[nid] for nid in self.node_global_ids]

    def shape_functions(self, xi, eta):
        """Funciones de forma cuadraticas triangulares (6 nodos) y sus derivadas."""
        L3 = 1.0 - xi - eta  # Coordenada del 1er nodo relativo
        L1 = xi
        L2 = eta

        # Funciones de forma N
        N = np.array([
            L3 * (2 * L3 - 1),  # N1 (esquina)
            L1 * (2 * L1 - 1),  # N2 (esquina)
            L2 * (2 * L2 - 1),  # N3 (esquina)
            4.0 * L1 * L3,      # N4 (medio entre 1 y 2)
            4.0 * L1 * L2,      # N5 (medio entre 2 y 3)
            4.0 * L2 * L3       # N6 (medio entre 3 y 1)
        ])

        # Derivadas respecto a xi (fila 0) y eta (fila 1) -> (2x6)
        dN_dxi_eta = np.array([
            [
                1.0 - 4.0 * L3,         # dN1/dxi
                4.0 * L1 - 1.0,         # dN2/dxi
                0.0,                    # dN3/dxi
                4.0 * (L3 - L1),        # dN4/dxi
                4.0 * L2,               # dN5/dxi
                -4.0 * L2               # dN6/dxi
            ],
            [
                1.0 - 4.0 * L3,         # dN1/deta
                0.0,                    # dN2/deta
                4.0 * L2 - 1.0,         # dN3/deta
                -4.0 * L1,              # dN4/deta
                4.0 * L1,               # dN5/deta
                4.0 * (L3 - L2)         # dN6/deta
            ]
        ])
        return N, dN_dxi_eta

    def B_matrix(self, xi, eta):
       
        _, dN_dxi_eta = self.shape_functions(xi, eta)
        
        coords = np.array([[node.x, node.y] for node in self.nodes])  # (6x2)

        # Matriz Jacobiana (2x2)
        J = dN_dxi_eta @ coords
        detJ = np.linalg.det(J)
        
        if detJ <= 0:
            raise ValueError(f"Determinante del Jacobiano invalido ({detJ}) en elemento T6 {self.global_id}.")

        invJ = np.linalg.inv(J)
        dN_dxy = invJ @ dN_dxi_eta  # (2x6)

        # Construccion de la matriz B (3x12)
        B = np.zeros((3, 12))
        for i in range(6):
            dN_dx = dN_dxy[0, i]
            dN_dy = dN_dxy[1, i]

            B[0, 2*i]     = dN_dx
            B[1, 2*i + 1] = dN_dy
            B[2, 2*i]     = dN_dy
            B[2, 2*i + 1] = dN_dx

        return B, detJ

    def stiffness_matrix(self):
        """Integra Ke (12x12) para T6 usando 3 puntos de Gauss triangulares."""
        D = self.material.D_matrix()
        Ke = np.zeros((12, 12))

        # Puntos de integracion de Gauss para triangulo (3 puntos)
        gauss_pts = [
            (0.5, 0.0),
            (0.5, 0.5),
            (0.0, 0.5)
        ]
        w_g = 1.0 / 6.0  # Peso para cada uno de los 3 puntos

        for xi_g, eta_g in gauss_pts:
            B, detJ = self.B_matrix(xi_g, eta_g)
            weight_total = w_g * detJ * self.thickness
            Ke += B.T @ D @ B * weight_total

        return Ke

    def global_dofs(self):
        """Mapea los grados de libertad globales del elemento T6 (12 DOFs)."""
        dofs = []
        for nid in self.node_global_ids:
            dofs.extend(self.mesh.node_global_dofs(nid))
        return dofs