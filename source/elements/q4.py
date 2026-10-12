import numpy as np

class Q4:
    dofs_per_node = 2
    n_nodes = 4

    def __init__(self, global_id, node_global_ids, mesh, material, thickness=1.0):
        self.global_id = global_id
        self.node_global_ids = node_global_ids
        self.mesh = mesh
        self.material = material
        self.thickness = thickness  # Espesor (t) para estado plano de tensión

        # Coordenadas x,y de los 4 nodos: array de 4x2
        self.coords = np.array([
            [mesh.node(nid).x, mesh.node(nid).y] for nid in node_global_ids
        ])

        # Coordenadas naturales de los nodos del Q4
        self.node_nat_coords = np.array([
            [-1.0, -1.0],
            [ 1.0, -1.0],
            [ 1.0,  1.0],
            [-1.0,  1.0]
        ])

    def shape_functions_natural(self, xi, eta):
        """Devuelve el vector de funciones de forma N = [N1, N2, N3, N4]"""
        N = np.zeros(4)
        for i in range(4):
            xi_i, eta_i = self.node_nat_coords[i]
            N[i] = 0.25 * (1.0 + xi_i * xi) * (1.0 + eta_i * eta)
        return N

    def shape_derivatives_natural(self, xi, eta):
        """
        Devuelve la matriz de derivadas respecto a (xi, eta) de tamaño (2, 4)
        Fila 0: dN/dxi , Fila 1: dN/deta
        """
        dN = np.zeros((2, 4))
        for i in range(4):
            xi_i, eta_i = self.node_nat_coords[i]
            dN[0, i] = 0.25 * xi_i * (1.0 + eta_i * eta)
            dN[1, i] = 0.25 * eta_i * (1.0 + xi_i * xi)
        return dN

    def jacobian(self, xi, eta):
        """Calcula la matriz Jacobiana J (2x2) y su determinante det(J)"""
        dN_dxi_eta = self.shape_derivatives_natural(xi, eta)  # (2, 4)
        J = dN_dxi_eta @ self.coords  # (2, 2)
        detJ = np.linalg.det(J)
        if detJ <= 0:
            raise ValueError(f"Jacobiano no positivo en elem {self.global_id}")
        return J, detJ

    def B_matrix(self, xi, eta):
        """Matriz de deformación B (3, 8) en el punto (xi, eta)"""
        J, detJ = self.jacobian(xi, eta)
        invJ = np.linalg.inv(J)
        dN_dxi_eta = self.shape_derivatives_natural(xi, eta)
        
        # Derivadas respecto a (x, y): inv(J) @ dN/d_xi_eta
        dN_dx_dy = invJ @ dN_dxi_eta  # (2, 4)

        B = np.zeros((3, 8))
        for i in range(4):
            dN_dx = dN_dx_dy[0, i]
            dN_dy = dN_dx_dy[1, i]

            B[0, 2*i]   = dN_dx
            B[1, 2*i+1] = dN_dy
            B[2, 2*i]   = dN_dy
            B[2, 2*i+1] = dN_dx
        
        return B, detJ

    def stiffness_matrix(self):
        """Integra Ke usando Cuadratura de Gauss 2x2"""
        C = self.material.C_matrix()  # Matriz constitutiva 3x3 para Estado Plano de Tensión
        Ke = np.zeros((8, 8))

        # Puntos y pesos de Gauss de 2do orden (2x2)
        p = 1.0 / np.sqrt(3.0)
        gauss_points = [-p, p]
        weights = [1.0, 1.0]

        for xi, w_xi in zip(gauss_points, weights):
            for eta, w_eta in zip(gauss_points, weights):
                B, detJ = self.B_matrix(xi, eta)
                weight_total = w_xi * w_eta * detJ * self.thickness
                Ke += B.T @ C @ B * weight_total

        return Ke

    def global_dofs(self):
        dofs = []
        for nid in self.node_global_ids:
            dofs.extend(self.mesh.node_global_dofs(nid))
        return dofs