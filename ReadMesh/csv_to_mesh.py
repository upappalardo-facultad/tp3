import ast
import os
import pandas as pd


def convert_csv_to_single_mesh(
    nodes_csv_path, elements_csv_path, output_mesh_path, dim=2
):
    # 1. Cargar Nodos
    df_nodes = pd.read_csv(nodes_csv_path)
    df_nodes.columns = df_nodes.columns.str.strip()

    # Preservar el orden original y mapear IDs a una secuencia contigua 1..N
    node_ids = df_nodes["nid"].astype(int).tolist()
    node_map = {old_id: new_id + 1 for new_id, old_id in enumerate(node_ids)}

    total_nodes = len(df_nodes)

    # 2. Cargar Elementos
    df_elements = pd.read_csv(elements_csv_path)
    df_elements.columns = df_elements.columns.str.strip()

    # Función para parsear la conectividad
    def parse_nodes(val):
        if isinstance(val, str):
            val = val.strip()
            if val.startswith("[") and val.endswith("]"):
                return ast.literal_eval(val)
            return [int(x) for x in val.replace(",", " ").split()]
        elif isinstance(val, (list, tuple)):
            return [int(x) for x in val]
        return [int(val)]

    df_elements["nodes_list"] = df_elements["nodes"].apply(parse_nodes)

    # 3. Procesar y mapear los tipos de elemento a las etiquetas de tu parser (Q4, Q8, T3, T6)
    parsed_elements = []

    for _, row in df_elements.iterrows():
        elem_type_str = str(row["type"]).strip().upper()
        conn = [node_map[int(n)] for n in row["nodes_list"]]
        num_nodes = len(conn)

        # Mapeo según nombre o número de nodos
        if (
            num_nodes == 4
            or "QUAD4" in elem_type_str
            or "CPLSTN4" in elem_type_str
            or "CPLSTS4" in elem_type_str
        ):
            label = "Q4"
        elif (
            num_nodes == 8
            or "QUAD8" in elem_type_str
            or "CPLSTN8" in elem_type_str
            or "CPLSTS8" in elem_type_str
        ):
            label = "Q8"
        elif (
            num_nodes == 3
            or "TRI3" in elem_type_str
            or "CTRIA3" in elem_type_str
            or "CPLSTN3" in elem_type_str
            or "CPLSTS3" in elem_type_str
        ):
            label = "T3"
        elif (
            num_nodes == 6
            or "TRI6" in elem_type_str
            or "CTRIA6" in elem_type_str
            or "LST" in elem_type_str
            or "CPLSTN6" in elem_type_str
            or "CPLSTS6" in elem_type_str
        ):
            label = "T6"
        else:
            label = elem_type_str

        parsed_elements.append((label, conn))

    # 4. Escribir el archivo .mesh único
    os.makedirs(os.path.dirname(output_mesh_path) or ".", exist_ok=True)

    with open(output_mesh_path, "w") as f:
        # Encabezado general
        f.write("# n_nodes n_elements dim\n")
        f.write(f"{total_nodes} {len(parsed_elements)} {dim}\n\n")

        # Nodos
        f.write("# node_number x y\n")
        for new_id, (_, row) in enumerate(df_nodes.iterrows(), start=1):
            x = float(row["x"])
            y = float(row["y"])
            f.write(f"{new_id} {x:.6f} {y:.6f}\n")

        # Elementos (mezcla de Q4, T3, Q8, T6, etc.)
        f.write("\n# element_number element_type connectivity\n")
        for elem_idx, (elem_type_label, conn) in enumerate(
            parsed_elements, start=1
        ):
            conn_str = " ".join(map(str, conn))
            f.write(f"{elem_idx} {elem_type_label} {conn_str}\n")

    print(
        f"Malla unificada generada exitosamente en: {output_mesh_path}"
    )
    print(
        f"Total de nodos: {total_nodes} | Total de elementos: {len(parsed_elements)}"
    )


# --- Ejemplo de Uso ---
if __name__ == "__main__":
    nodes_csv = "C:/Users/mange/Documents/GitHub/ReadMesh/mesh_dat_files/ presa_opt_nodes.csv"
    elements_csv = "C:/Users/mange/Documents/GitHub/ReadMesh/mesh_dat_files/ presa_opt_elements.csv"
    output_mesh = "C:/Users/mange/Documents/GitHub/tp3/meshes/presa_opt.mesh"

    convert_csv_to_single_mesh(
        nodes_csv_path=nodes_csv,
        elements_csv_path=elements_csv,
        output_mesh_path=output_mesh,
        dim=2,
    )