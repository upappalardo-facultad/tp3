import numpy as np
from source.mesh.mesh import Mesh
from source.materials.linear_elastic import LinearElastic
from source.elements.q8 import Q8
from source.systems.dense_matrix import DenseMatrix
from source.systems.boundary_conditions import apply_dirichlet_bc
from source.solvers.linear import solve
from source.output.paraview import write_results
from source.postprocessing.element_stresses import compute_element_stresses

# 1. Cargar Malla desde el archivo .mesh del Patch Test Q8
mesh = Mesh()
mesh.read("tp3/meshes/patchq8.mesh")
mesh.dofs_per_node = Q8.dofs_per_node

# 2. Material (Tensión/Deformación Plana configurada en la ley constitutiva o el postproceso)
material = LinearElastic(E=210e9, nu=0.3)

# 3. Instanciar los elementos Q8
elements = []
for e in range(mesh.number_of_elements()):
    elem_type = mesh.element_types[e]
    if elem_type not in ["Q8", "QUAD8", "CQUAD8"]:
        raise ValueError(f"Se esperaba elemento Q8 pero se encontró {elem_type}")
    
    element = Q8(
        global_id=mesh.element_global_ids[e],
        node_global_ids=mesh.connectivity[e],
        mesh=mesh,
        material=material,
        thickness=1.0
    )
    elements.append(element)

# 4. Ensamblado de la Matriz de Rigidez Global K
num_dofs = mesh.number_of_dofs()
K = DenseMatrix(num_dofs)
for element in elements:
    K.scatter_local_to_global(element)

f = np.zeros(num_dofs)

# 5. Definición del Campo de Desplazamientos Analítico u(x, y)
a0, a1, a2 = 0.001, 0.002, -0.0015
b0, b1, b2 = -0.0005, 0.001, 0.0025

def exact_disp(x, y):
    ux = a0 + a1 * x + a2 * y
    uy = b0 + b1 * x + b2 * y
    return ux, uy

# 6. Identificación Automática de Nodos de Contorno Exterior
boundary_node_ids = []

for nid in range(mesh.number_of_nodes()):
    node = mesh.node(nid)
    # Si el nodo está en cualquiera de los 4 bordes exteriores del rectángulo [0, 100] x [200, 300]
    if (np.isclose(node.x, 0.0) or np.isclose(node.x, 100.0) or 
        np.isclose(node.y, 200.0) or np.isclose(node.y, 300.0)):
        boundary_node_ids.append(nid)

for nid in boundary_node_ids:
    node = mesh.node(nid)
    ux_val, uy_val = exact_disp(node.x, node.y)
    dofs = mesh.node_global_dofs(nid)
    
    apply_dirichlet_bc(K, f, dofs[0], ux_val)
    apply_dirichlet_bc(K, f, dofs[1], uy_val)

# 7. Resolver K * u = f
u = solve(K, f)

# 8. Calcular Tensiones especificando el estado plano
stresses = compute_element_stresses(mesh, elements, u, plane_state="plane_strain")

# 9. Verificación en Consola
print("=== RESULTADOS PATCH TEST Q8 ===")
print(f"Tensión sig_xx media : {stresses['sig_xx'].mean():.6e} | Desv. Estándar: {stresses['sig_xx'].std():.6e}")
print(f"Tensión sig_yy media : {stresses['sig_yy'].mean():.6e} | Desv. Estándar: {stresses['sig_yy'].std():.6e}")
print(f"Tensión tau_xy media : {stresses['tau_xy'].mean():.6e} | Desv. Estándar: {stresses['tau_xy'].std():.6e}")

# 10. Exportar a ParaView pasando el diccionario de tensiones
write_results("tp3/results/patch_q8_results.vtu", mesh, elements, u, stresses=stresses)