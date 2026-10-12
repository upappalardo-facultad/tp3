import numpy as np

class T3:
    dofs_per_node = 2
    n_nodes = 3

    def __init__(self, global_id, node_global_ids, mesh, material, thickness=1.0):
        self.global_id = global_id
        self.node_global_ids = node_global_ids
        self.mesh = mesh
        self.material = material
        self.thickness = thickness

        # Coordenadas nodales
        self.coords = np.array([
            [mesh.node(nid).x, mesh.node(nid).y] for nid in node_global_ids
        ])

        x1, y1 = self.coords[0]
        x2, y2 = self.coords[1]
        x3, y3 = self.coords[2]

        # Cálculo del área 2A
        self.two_area = (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
        self.area = self.two_area / 2.0

        if self.area <= 0:
            raise ValueError(f"Área nula o negativa en el elemento T3 id {self.global_id}")

        # Coeficientes beta y gamma
        self.beta  = np.array([y2 - y3, y3 - y1, y1 - y2])
        self.gamma = np.array([x3 - x2, x1 - x3, x2 - x1])

    def B_matrix(self):
        """Matriz B constante para T3 de tamaño (3, 6)"""
        B = np.zeros((3, 6))
        for i in range(3):
            dN_dx = self.beta[i] / self.two_area
            dN_dy = self.gamma[i] / self.two_area

            B[0, 2*i]   = dN_dx
            B[1, 2*i+1] = dN_dy
            B[2, 2*i]   = dN_dy
            B[2, 2*i+1] = dN_dx

        return B

    def stiffness_matrix(self):
        """Al ser B constante: Ke = B^T * D * B * Area * Espesor"""
        C = self.material.C_matrix()
        B = self.B_matrix()
        Ke = B.T @ C @ B * self.area * self.thickness
        return Ke

    def global_dofs(self):
        dofs = []
        for nid in self.node_global_ids:
            dofs.extend(self.mesh.node_global_dofs(nid))
        return dofs