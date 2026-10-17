import numpy as np
from source.mesh.mesh import Mesh
from source.materials.linear_elastic import LinearElastic
from source.elements.q4 import Q4
from source.elements.t3 import T3
from source.elements.q8 import Q8
from source.elements.t6 import T6
from source.systems.dense_matrix import DenseMatrix
from source.systems.boundary_conditions import apply_dirichlet_bc
from source.solvers.linear import solve
from source.output.paraview import write_results
from source.postprocessing.element_stresses import compute_element_stresses

# ==============================================================================
# 1. PARÁMETROS OFICIALES DE LA CÁTEDRA (EN UNIDADES S.I. - METROS, PASCALES, KG)
# ==============================================================================
E = 21e9              # Módulo de Elasticidad: 21 GPa = 21e9 Pa
nu = 0.20             # Coeficiente de Poisson
rho_concrete = 2400.0 # Densidad del hormigón [kg/m3]
g = 9.81              # Gravitación [m/s2]
thickness = 1.0       # Espesor unitario para estado plano [m]

rho_water = 1000.0    # Densidad del agua [kg/m3]
height_dam = 30.0     # Altura total de la presa [m]
borde_libre = 2.0     # Borde libre d = 2m
water_level = height_dam - borde_libre  # Cota del agua = 28.0 m

PLANE_STATE = "plane_strain"  # Estado Plano de Deformación

# ==============================================================================
# 2. CARGA DE MALLA Y CONVERSIÓN DE MILÍMETROS A METROS
# ==============================================================================
mesh = Mesh()
mesh.read("tp3/meshes/presa_opt.mesh") 
#tp3/meshes/presa_cos_q8t6.mesh asi funciona el q8t6
#tp3/meshes/presa_q4t3_cos.mesh asi funnciona el q4t3

mesh.dofs_per_node = 2

# CONVERSIÓN A METROS: 
for node in mesh.nodes:
        node.x /= 1000.0
        node.y /= 1000.0

material = LinearElastic(E=E, nu=nu)

# ==============================================================================
# 3. INSTANCIACIÓN DE ELEMENTOS
# ==============================================================================
elements = []
for e in range(mesh.number_of_elements()):
    elem_type = mesh.element_types[e]
    global_id = mesh.element_global_ids[e]
    connectivity = mesh.connectivity[e]

    if elem_type in ["Q4", "QUAD4", "CQUAD4", "CPLSTN4"]:
        element = Q4(global_id, connectivity, mesh, material, thickness)
    elif elem_type in ["T3", "TRI3", "CTRIA3", "CPLSTN3"]:
        element = T3(global_id, connectivity, mesh, material, thickness)
    elif elem_type in ["Q8", "QUAD8", "CQUAD8", "CPLSTN8"]:
        element = Q8(global_id, connectivity, mesh, material, thickness)
    elif elem_type in ["T6", "TRI6", "CTRIA6", "LST", "CPLSTN6"]:
        element = T6(global_id, connectivity, mesh, material, thickness)
    else:
        raise ValueError(f"Tipo de elemento {elem_type} no reconocido.")

    elements.append(element)

num_dofs = mesh.number_of_dofs()
K = DenseMatrix(num_dofs)
f = np.zeros(num_dofs)

# ==============================================================================
# 4. ENSAMBLADO DE K Y CARGA DE PESO PROPIO (GRAVEDAD)
# ==============================================================================
for element in elements:
    K.scatter_local_to_global(element)

    n_elem_nodes = len(element.node_global_ids)
    coords = np.array([[mesh.nodes[nid].x, mesh.nodes[nid].y] for nid in element.node_global_ids])
    
    if n_elem_nodes in [3, 6]:  # T3 / T6
        area = 0.5 * np.abs(np.cross(coords[1]-coords[0], coords[2]-coords[0]))
    else:  # Q4 / Q8
        area = 0.5 * np.abs(np.cross(coords[2]-coords[0], coords[3]-coords[1]))

    element_weight = area * thickness * rho_concrete * g
    force_per_node_y = -element_weight / n_elem_nodes

    for nid in element.node_global_ids:
        dofs = mesh.node_global_dofs(nid)
        f[dofs[1]] += force_per_node_y

# ==============================================================================
# 5. CARGA HIDROSTÁTICA DEL AGUA (x ≈ 0, y <= 28.0 m)
# ==============================================================================
upstream_nodes = []
for nid in range(mesh.number_of_nodes()):
    node = mesh.node(nid)
    if np.isclose(node.x, 0.0) and node.y <= water_level + 1e-3:
        upstream_nodes.append((nid, node.y))

upstream_nodes.sort(key=lambda item: item[1])

for i in range(len(upstream_nodes) - 1):
    nid1, y1 = upstream_nodes[i]
    nid2, y2 = upstream_nodes[i + 1]

    dy = abs(y2 - y1)
    p1 = rho_water * g * (water_level - y1)
    p2 = rho_water * g * (water_level - y2)

    F_segment = 0.5 * (p1 + p2) * dy * thickness

    dofs1 = mesh.node_global_dofs(nid1)
    dofs2 = mesh.node_global_dofs(nid2)

    f[dofs1[0]] += F_segment / 2.0
    f[dofs2[0]] += F_segment / 2.0

# ==============================================================================
# 6. CONDICIONES DE BORDE (EMPOTRAMIENTO EN LA BASE y = 0.0 m)
# ==============================================================================
for nid in range(mesh.number_of_nodes()):
    node = mesh.node(nid)
    if np.isclose(node.y, 0.0):
        dofs = mesh.node_global_dofs(nid)
        apply_dirichlet_bc(K, f, dofs[0], 0.0)
        apply_dirichlet_bc(K, f, dofs[1], 0.0)

# ==============================================================================
# 7. RESOLUCIÓN
# ==============================================================================
print("Resolviendo el sistema de ecuaciones de la presa...")
u = solve(K, f)
print("¡Sistema resuelto exitosamente!")

# ==============================================================================
# 8. POSTPROCESAMIENTO Y VTK
# ==============================================================================
stresses = compute_element_stresses(mesh, elements, u, plane_state=PLANE_STATE)

output_vtu = "tp3/results/presa_opt.vtu" 
write_results(output_vtu, mesh, elements, u, stresses=stresses)

print(f"\n=== RESULTADOS CORREGIDOS DE LA PRESA (METROS Y PASCALES) ===")
print(f"Desplazamiento horizontal máx (Ux) : {np.max(u[0::2]) * 1000.0:.4f} mm")
print(f"Desplazamiento vertical máx   (Uy) : {np.min(u[1::2]) * 1000.0:.4f} mm")
print(f"Tensión Von Mises Máxima            : {np.max(stresses['von_mises']) / 1e6:.2f} MPa")
print(f"Archivo generado para ParaView     : {output_vtu}")