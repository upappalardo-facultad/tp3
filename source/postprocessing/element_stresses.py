import numpy as np

def compute_element_stresses(mesh, elements, u, plane_state="plane_strain"):
    """
    Calcula el tensor de tensiones (sig_xx, sig_yy, tau_xy) y la tensión equivalente de Von Mises.
    """
    num_elements = len(elements)
    
    sig_xx = np.zeros(num_elements)
    sig_yy = np.zeros(num_elements)
    tau_xy = np.zeros(num_elements)
    von_mises = np.zeros(num_elements)

    for i, elem in enumerate(elements):
        # 1. Extraer desplazamientos nodales del elemento (2 dofs por nodo)
        u_e = np.zeros(len(elem.node_global_ids) * 2)
        for local_idx, nid in enumerate(elem.node_global_ids):
            # Asumiendo que mesh devuelve dofs [dof_x, dof_y] para el nid
            dofs = mesh.node_global_dofs(nid)
            u_e[2 * local_idx]     = u[dofs[0]]
            u_e[2 * local_idx + 1] = u[dofs[1]]

        # 2. Matriz Constitutiva C de este elemento
        C = elem.material.C_matrix(plane_state=plane_state)

        # 3. Matriz B en el centro del elemento (xi=0, eta=0) o según el elemento
        if hasattr(elem, 'B_matrix'):
            # Para Q4, Q8, T6 (devuelve B, detJ o solo B)
            res = elem.B_matrix(0.0, 0.0)
            B = res[0] if isinstance(res, tuple) else res
        elif hasattr(elem, 'B'):
            # Para T3 / CST
            B = elem.B
        else:
            raise AttributeError(f"El elemento {type(elem).__name__} no define B_matrix o B.")

        # 4. Tensiones: sigma = C * B * u_e
        eps = B @ u_e
        sigma = C @ eps

        sig_xx[i] = sigma[0]
        sig_yy[i] = sigma[1]
        tau_xy[i] = sigma[2]

        # 5. Tensión de Von Mises (tomando en cuenta sig_zz si es Deformación Plana)
        if plane_state.lower() == "plane_strain":
            sig_zz = elem.material.nu * (sigma[0] + sigma[1])
            vm = np.sqrt(0.5 * ((sigma[0] - sigma[1])**2 + (sigma[1] - sig_zz)**2 + (sig_zz - sigma[0])**2 + 6.0 * sigma[2]**2))
        else:
            vm = np.sqrt(sigma[0]**2 - sigma[0] * sigma[1] + sigma[1]**2 + 3.0 * sigma[2]**2)

        von_mises[i] = vm

    return {
        "sig_xx": sig_xx,
        "sig_yy": sig_yy,
        "tau_xy": tau_xy,
        "von_mises": von_mises
    }