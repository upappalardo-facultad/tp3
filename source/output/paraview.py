import meshio
import numpy as np

def write_results(filename, mesh, elements, u, stresses=None):
    """
    Exporta resultados de desplazamientos y tensiones a formato .vtu de ParaView.
    Soporta mallas mixtas (múltiples bloques de elementos en meshio).
    """
    # 1. Coordenadas de los nodos
    points = np.array([[node.x, node.y, 0.0] for node in mesh.nodes])
    
    # 2. Desplazamientos nodales (3D para VTK)
    displacement = np.zeros((mesh.number_of_nodes(), 3))
    for i in range(mesh.number_of_nodes()):
        dofs = mesh.node_global_dofs(i)
        displacement[i, 0] = u[dofs[0]]
        displacement[i, 1] = u[dofs[1]]

    # 3. Estructuración de celdas por tipo de elemento
    cells_dict = {}
    
    for i, connectivity in enumerate(mesh.connectivity):
        elem_type = mesh.element_types[i]
        
        if elem_type in ["Q4", "QUAD4", "CQUAD4", "CPLSTN4"]:
            key = "quad"
        elif elem_type in ["T3", "TRI3", "CTRIA3", "CPLSTN3"]:
            key = "triangle"
        elif elem_type in ["Q8", "QUAD8", "CQUAD8", "CPLSTN8"]:
            key = "quad8"
        elif elem_type in ["T6", "TRI6", "CTRIA6", "LST", "CPLSTN6"]:
            key = "triangle6"
        else:
            continue

        if key not in cells_dict:
            cells_dict[key] = []
        
        # Conectividad en base 0
        cells_dict[key].append(connectivity)

    cells = [(key, cells_dict[key]) for key in cells_dict]

    # 4. Asignación de datos por celdas (Cell Data) por bloque
    cell_data = {}
    if stresses is not None:
        # Preparamos las listas por componente para cada bloque de celdas
        for field in ["sig_xx", "sig_yy", "tau_xy", "von_mises"]:
            if field in stresses:
                cell_data[field] = []
                
                # Recorremos cada bloque de celdas para agrupar sus tensiones asociadas
                elem_offset = 0
                for key, elem_list in cells:
                    n_elem_in_block = len(elem_list)
                    block_stresses = stresses[field][elem_offset : elem_offset + n_elem_in_block]
                    cell_data[field].append(block_stresses)
                    elem_offset += n_elem_in_block

    # 5. Crear e escribir la malla en formato VTU
    output_mesh = meshio.Mesh(
        points=points,
        cells=cells,
        point_data={"displacement": displacement},
        cell_data=cell_data
    )
    output_mesh.write(filename)