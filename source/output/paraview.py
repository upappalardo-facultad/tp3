import os
import meshio


def element_cells(mesh):
    """
    Convierte la conectividad de la malla al formato de celdas de meshio
    soportando Q4 ('quad'), T3 ('triangle'), Q8 ('quad8') y T6 ('triangle6').
    """
    quads = []
    triangles = []
    quads8 = []
    triangles6 = []

    for i, connectivity in enumerate(mesh.connectivity):
        elem_type = mesh.element_types[i]
        
        # mesh.connectivity YA VIENE en base 0 desde mesh.py. NO restar 1.
        conn = connectivity 
        
        # Elementos lineales (1er Orden)
        if elem_type in ["Q4", "QUAD4", "CQUAD4", "CPLSTN4", "CPLSTS4"]:
            quads.append(conn)
        elif elem_type in ["T3", "TRI3", "CTRIA3", "CPLSTN3", "CPLSTS3"]:
            triangles.append(conn)
            
        # Elementos cuadráticos (2do Orden)
        elif elem_type in ["Q8", "QUAD8", "CQUAD8", "CPLSTN8", "CPLSTS8"]:
            quads8.append(conn)
        elif elem_type in ["T6", "TRI6", "CTRIA6", "LST", "CPLSTN6", "CPLSTS6"]:
            triangles6.append(conn)

    cells = []
    if quads:
        cells.append(("quad", quads))
    if triangles:
        cells.append(("triangle", triangles))
    if quads8:
        cells.append(("quad8", quads8))
    if triangles6:
        cells.append(("triangle6", triangles6))

    return cells

def write(filename, points, cells, displacement, stresses=None):
    directory = os.path.dirname(filename)

    if directory != "":
        os.makedirs(directory, exist_ok=True)

    cell_data = {}
    if stresses is not None:
        # Registra cada componente del diccionario de tensiones (sig_xx, sig_yy, tau_xy, von_mises)
        for key, value in stresses.items():
            cell_data[key] = [value]

    output_mesh = meshio.Mesh(
        points=points,
        cells=cells,
        point_data={"displacement": displacement},
        cell_data=cell_data,
    )

    output_mesh.write(filename)


def write_results(filename, mesh, elements, u, stresses=None):
    points = []
    displacement = []

    for node in mesh.nodes:
        dofs = mesh.node_global_dofs(node.global_id)

        points.append([node.x, node.y, 0.0])
        displacement.append([u[dofs[0]], u[dofs[1]], 0.0])

    cells = element_cells(mesh)
    write(filename, points, cells, displacement, stresses)