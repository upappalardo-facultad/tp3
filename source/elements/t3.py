import numpy as np

class T3:
    def __init__(self, global_id, node_global_ids, mesh, material, thickness=1.0):
        self.global_id = global_id
        self.node_global_ids = node_global_ids  # Lista con los IDs globales de los 3 nodos
        self.mesh = mesh
        self.material = material
        self.thickness = thickness
        
        # Recupera las instancias de las coordenadas nodales desde la malla
        self.nodes = [self.mesh.nodes[nid] for nid in self.node_global_ids]

    def shape_functions(self, xi, eta):
        """Funciones de forma en coordenadas naturales triangulares (xi, eta)."""
        N = np.array([
            1.0 - xi - eta,
            xi,
            eta
        ])
        
        # Derivadas respecto a xi y eta: [dN/dxi, dN/deta] (2x3)
        dN_dxi_eta = np.array([
            [-1.0,  1.0,  0.0],
            [-1.0,  0.0,  1.0]
        ])
        return N, dN_dxi_eta

    def B_matrix(self, xi, eta):
        """Calcula la matriz B (3x6) y det(J) en las coordenadas (xi, eta)."""
        _, dN_dxi_eta = self.shape_functions(xi, eta)
        
        # Coordenadas nodales reales (3x2)
        coords = np.array([[node.x, node.y] for node in self.nodes])

        # Matriz Jacobiana (2x2): J = dN_dxi_eta @ coords
        J = dN_dxi_eta @ coords
        detJ = np.linalg.det(J)
        
        if detJ <= 0:
            raise ValueError(f"Determinante del Jacobiano invalido ({detJ}). Revisa la orientacion de los nodos.")

        invJ = np.linalg.inv(J)

        # Derivadas de N respecto a las coordenadas reales (x, y): (2x3)
        dN_dxy = invJ @ dN_dxi_eta

        # Construccion de la matriz B (3x6)
        B = np.zeros((3, 6))
        for i in range(3):
            dN_dx = dN_dxy[0, i]
            dN_dy = dN_dxy[1, i]

            B[0, 2*i]     = dN_dx
            B[1, 2*i + 1] = dN_dy
            B[2, 2*i]     = dN_dy
            B[2, 2*i + 1] = dN_dx

        return B, detJ

    def stiffness_matrix(self):
        """Integra Ke para T3 usando 1 punto de Gauss triangular en el centroide."""
        C = self.material.C_matrix()  # Matriz constitutiva (3x3)

        # Centroide y peso para triangulo isoparametrico
        xi_g = 1.0 / 3.0
        eta_g = 1.0 / 3.0
        w_g = 0.5  # Peso de integracion

        # Evalua B y detJ en el centroide
        B, detJ = self.B_matrix(xi_g, eta_g)

        weight_total = w_g * detJ * self.thickness
        Ke = B.T @ C @ B * weight_total

        return Ke

    def global_dofs(self):
        """Mapea los grados de libertad globales del elemento T3 (6 DOFs)."""
        dofs = []
        for nid in self.node_global_ids:
            dofs.extend(self.mesh.node_global_dofs(nid))
        return dofs