import numpy as np
from source.mesh.mesh import Mesh
from source.materials.linear_elastic import LinearElastic
from source.elements.t3 import T3
from source.systems.dense_matrix import DenseMatrix
from source.systems.boundary_conditions import apply_dirichlet_bc
from source.solvers.linear import solve
from source.output.paraview import write_results
from source.postprocessing.element_stresses import compute_element_stresses

# 1. Cargar Malla T3
mesh = Mesh()
mesh.read("tp3/meshes/patcht3.mesh")
mesh.dofs_per_node = T3.dofs_per_node

# 2. Material
material = LinearElastic(E=210e9, nu=0.3)

# 3. Instanciar los elementos T3
elements = []
for e in range(mesh.number_of_elements()):
    elem_type = mesh.element_types[e]
    if elem_type not in ["T3", "TRI3", "CTRIA3"]:
        raise ValueError(f"Se esperaba elemento T3 pero se encontró {elem_type}")
    
    element = T3(
        global_id=mesh.element_global_ids[e],
        node_global_ids=mesh.connectivity[e],
        mesh=mesh,
        material=material,
        thickness=1.0
    )
    elements.append(element)

# 4. Ensamblado de K
num_dofs = mesh.number_of_dofs()
K = DenseMatrix(num_dofs)
for element in elements:
    K.scatter_local_to_global(element)

f = np.zeros(num_dofs)

# 5. Campo Analítico u(x, y)
a0, a1, a2 = 0.001, 0.002, -0.0015
b0, b1, b2 = -0.0005, 0.001, 0.0025

def exact_disp(x, y):
    ux = a0 + a1 * x + a2 * y
    uy = b0 + b1 * x + b2 * y
    return ux, uy

# 6. Nodos de Contorno Exterior
# Nodo interno libre: 4 (1-based)
internal_nodes_1based = [4]
all_nodes_1based = list(range(1, mesh.number_of_nodes() + 1))
boundary_node_ids_1based = [nid for nid in all_nodes_1based if nid not in internal_nodes_1based]

boundary_node_ids = [nid - 1 for nid in boundary_node_ids_1based]  # A base 0

for nid in boundary_node_ids:
    node = mesh.node(nid)
    ux_val, uy_val = exact_disp(node.x, node.y)
    dofs = mesh.node_global_dofs(nid)
    
    apply_dirichlet_bc(K, f, dofs[0], ux_val)
    apply_dirichlet_bc(K, f, dofs[1], uy_val)

# 7. Resolver K * u = f
u = solve(K, f)

# 8. Calcular Tensiones
stresses = compute_element_stresses(mesh, elements, u, plane_state="plane_stress")

# 9. Verificación en Consola
print("=== RESULTADOS PATCH TEST T3 ===")
print(f"Tensión sig_xx media : {stresses['sig_xx'].mean():.6e} | Desv. Estándar: {stresses['sig_xx'].std():.6e}")
print(f"Tensión sig_yy media : {stresses['sig_yy'].mean():.6e} | Desv. Estándar: {stresses['sig_yy'].std():.6e}")
print(f"Tensión tau_xy media : {stresses['tau_xy'].mean():.6e} | Desv. Estándar: {stresses['tau_xy'].std():.6e}")

# 10. Exportar a ParaView
write_results("tp3/results/patch_t3_results.vtu", mesh, elements, u, stresses=stresses)