import math
import random
import numpy as np
import plotly.graph_objects as go


def generar_espacio_perturbado(
    num_particulas=80, ancho_total=100, alto_total=100, profundo_total=100
):
    volumen = ancho_total * alto_total * profundo_total
    s = (volumen / num_particulas) ** (1 / 3)

    cols = max(1, round(ancho_total / s))
    rows = max(1, round(alto_total / s))
    layers = max(1, round(profundo_total / s))

    espaciado_x = ancho_total / cols
    espaciado_y = alto_total / rows
    espaciado_z = profundo_total / layers

    puntos_todos = []
    puntos_interiores = []

    # Capa de puntos fantasma (-1 a +1) para cerrar correctamente las celdas en los bordes
    for i in range(-1, cols + 1):
        for j in range(-1, rows + 1):
            for k in range(-1, layers + 1):
                base_x = i * espaciado_x + (espaciado_x / 2)
                base_y = j * espaciado_y + (espaciado_y / 2)
                base_z = k * espaciado_z + (espaciado_z / 2)

                pert_x = random.uniform(-0.3, 0.3) * espaciado_x
                pert_y = random.uniform(-0.3, 0.3) * espaciado_y
                pert_z = random.uniform(-0.3, 0.3) * espaciado_z

                x = base_x + pert_x
                y = base_y + pert_y
                z = base_z + pert_z

                pt = (x, y, z)
                puntos_todos.append(pt)

                # Restringimos las semillas interiores dentro del dominio [0, dimension]
                if 0 <= i < cols and 0 <= j < rows and 0 <= k < layers:
                    x_in = max(0.01, min(ancho_total - 0.01, x))
                    y_in = max(0.01, min(alto_total - 0.01, y))
                    z_in = max(0.01, min(profundo_total - 0.01, z))
                    puntos_interiores.append((x_in, y_in, z_in))

    return puntos_todos, puntos_interiores


def esfera_circunscrita(tetra):
    A, B, C, D = (
        np.array(tetra[0]),
        np.array(tetra[1]),
        np.array(tetra[2]),
        np.array(tetra[3]),
    )

    AB = B - A
    AC = C - A
    AD = D - A

    len_AB_sq = np.dot(AB, AB)
    len_AC_sq = np.dot(AC, AC)
    len_AD_sq = np.dot(AD, AD)

    cross_AC_AD = np.cross(AC, AD)
    cross_AD_AB = np.cross(AD, AB)
    cross_AB_AC = np.cross(AB, AC)

    denom = 2 * np.dot(AB, cross_AC_AD)

    if abs(denom) < 1e-10:
        return None, None

    numerador = (
        len_AB_sq * cross_AC_AD
        + len_AC_sq * cross_AD_AB
        + len_AD_sq * cross_AB_AC
    )

    center = A + numerador / denom
    radius_squared = np.sum((A - center) ** 2)

    return center, radius_squared


def punto_circuncirculo(punto, tetra):
    center, radius_squared = esfera_circunscrita(tetra)
    if center is None:
        return False
    distancia_cuadrada = np.sum((np.array(punto) - center) ** 2)
    return distancia_cuadrada <= radius_squared


def punto_nuevo(punto, lista, bad_lista):
    for tetra in lista:
        if punto_circuncirculo(punto, tetra):
            bad_lista.append(tetra)
    for tetra in bad_lista:
        if tetra in lista:
            lista.remove(tetra)


def caras_borde(bad_lista):
    caras_conteo = []
    for tetra in bad_lista:
        A, B, C, D = tetra
        tetra_caras = [(A, B, C), (A, B, D), (A, C, D), (B, C, D)]
        for cara in tetra_caras:
            cara_norm = tuple(
                sorted(cara, key=lambda pt: (pt[0], pt[1], pt[2]))
            )
            encontrada = False
            for item in caras_conteo:
                if item[0] == cara_norm:
                    item[1] += 1
                    encontrada = True
                    break
            if not encontrada:
                caras_conteo.append([cara_norm, 1])

    borde = []
    for item in caras_conteo:
        cara, conteo = item
        if conteo == 1:
            borde.append(cara)
    return borde


def completar_triangulacion(punto_nuevo, caras_borde, lista_principal):
    for cara in caras_borde:
        pA, pB, pC = cara
        nuevo_tetraedro = (pA, pB, pC, punto_nuevo)
        lista_principal.append(nuevo_tetraedro)


def delaunay_3d_con_params(points, ancho=100, alto=100, profundo=100):
    # Súper-tetraedro escalado dinámicamente según el tamaño del volumen
    limite = max(ancho, alto, profundo) * 50
    supa_tetra = (
        (-limite, -limite, -limite),
        (limite * 2, -limite, -limite),
        (0, limite * 2, -limite),
        (0, 0, limite * 2),
    )
    lista_principal = [supa_tetra]

    for p in points:
        bad_lista = []
        punto_nuevo(p, lista_principal, bad_lista)
        borde = caras_borde(bad_lista)
        completar_triangulacion(p, borde, lista_principal)

    tetraedros_limpios = []
    for tetra in lista_principal:
        comparte_super = any(vertice in supa_tetra for vertice in tetra)
        if not comparte_super:
            tetraedros_limpios.append(tetra)

    return tetraedros_limpios


def ordenar_puntos_cara_3d(puntos_cara, p_origen, p_vecino):
    if len(puntos_cara) < 3:
        return puntos_cara

    arr_puntos = np.array(puntos_cara)
    centroide = np.mean(arr_puntos, axis=0)

    v = np.array(p_vecino) - np.array(p_origen)
    norm_v = np.linalg.norm(v)
    if norm_v == 0:
        return puntos_cara
    v = v / norm_v

    arbitrario = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(v, arbitrario)) > 0.9:
        arbitrario = np.array([0.0, 1.0, 0.0])

    e1 = np.cross(v, arbitrario)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(v, e1)

    angulos = []
    for pt in arr_puntos:
        diff = pt - centroide
        x = np.dot(diff, e1)
        y = np.dot(diff, e2)
        angulos.append(math.atan2(y, x))

    pares = sorted(zip(angulos, puntos_cara), key=lambda x: x[0])
    return [p[1] for p in pares]


def voronoi_3d_puro(
    rango_caras=(4, 100),
    ancho=100,
    alto=100,
    profundo=100,
    num_particulas=80,
):
    cota_inf, cota_sup = rango_caras
    puntos_todos, puntos_interiores = generar_espacio_perturbado(
        num_particulas, ancho, alto, profundo
    )
    tetraedros = delaunay_3d_con_params(puntos_todos, ancho, alto, profundo)
    poliedros = []

    for p in puntos_interiores:
        vecinos_dict = {}
        for tetra in tetraedros:
            if p in tetra:
                center, _ = esfera_circunscrita(tetra)
                if center is not None:
                    for vertice in tetra:
                        if vertice != p:
                            if vertice not in vecinos_dict:
                                vecinos_dict[vertice] = []
                            vecinos_dict[vertice].append(tuple(center))

        caras_poliedro = []
        for vecino, puntos_cara in vecinos_dict.items():
            puntos_cara = list(set(puntos_cara))
            if len(puntos_cara) >= 3:
                cara_ordenada = ordenar_puntos_cara_3d(puntos_cara, p, vecino)
                caras_poliedro.append(cara_ordenada)

        if cota_inf <= len(caras_poliedro) <= cota_sup:
            # Se guardan las celdas de Voronoi puras sin realizar contracción
            poliedros.append(caras_poliedro)

    return poliedros, puntos_interiores


def visualizar_voronoi_solido(poliedros, puntos_semilla):
    fig = go.Figure()
    edge_x, edge_y, edge_z = [], [], []

    for poliedro in poliedros:
        r, g, b = (
            random.randint(80, 240),
            random.randint(80, 240),
            random.randint(80, 240),
        )
        color = f"rgb({r}, {g}, {b})"

        x_all, y_all, z_all = [], [], []
        i_indices, j_indices, k_indices = [], [], []
        vertex_offset = 0

        for cara in poliedro:
            if len(cara) < 3:
                continue

            face_indices = []
            for i, v in enumerate(cara):
                x_all.append(v[0])
                y_all.append(v[1])
                z_all.append(v[2])
                face_indices.append(vertex_offset)
                vertex_offset += 1

                v_siguiente = cara[(i + 1) % len(cara)]
                edge_x.extend([v[0], v_siguiente[0], None])
                edge_y.extend([v[1], v_siguiente[1], None])
                edge_z.extend([v[2], v_siguiente[2], None])

            for idx_cara in range(1, len(face_indices) - 1):
                i_indices.append(face_indices[0])
                j_indices.append(face_indices[idx_cara])
                k_indices.append(face_indices[idx_cara + 1])

        if x_all:
            fig.add_trace(
                go.Mesh3d(
                    x=x_all,
                    y=y_all,
                    z=z_all,
                    i=i_indices,
                    j=j_indices,
                    k=k_indices,
                    color=color,
                    opacity=0.6,
                    flatshading=True,
                    lighting=dict(
                        ambient=0.6, diffuse=0.8, roughness=0.8, specular=0.1
                    ),
                    hoverinfo="none",
                )
            )

    fig.add_trace(
        go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            line=dict(color="rgb(30, 30, 30)", width=3),
            hoverinfo="none",
            showlegend=False,
        )
    )

    px = [p[0] for p in puntos_semilla]
    py = [p[1] for p in puntos_semilla]
    pz = [p[2] for p in puntos_semilla]
    fig.add_trace(
        go.Scatter3d(
            x=px,
            y=py,
            z=pz,
            mode="markers",
            marker=dict(size=4, color="black", symbol="circle"),
            name="Semillas",
        )
    )

    fig.update_layout(
        title=dict(
            text="Espacio de Voronoi Continuo (Sin Contracción)",
            font=dict(color="black", size=20),
        ),
        paper_bgcolor="rgb(245, 245, 245)",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="rgb(245, 245, 245)",
        ),
        margin=dict(l=0, r=0, b=0, t=50),
    )

    fig.show()


# ==============================================================================
# TEST DE PRUEBA
# ==============================================================================
if __name__ == "__main__":
    ANCHO, ALTO, PROFUNDO = 100, 100, 100
    NUM_PARTICULAS = 40

    print("--- INICIANDO TEST DE GENERACIÓN VORONOI ---")
    print(f"Dimensiones del espacio: {ANCHO}x{ALTO}x{PROFUNDO} mm")
    print(f"Semillas solicitadas: {NUM_PARTICULAS}")

    poliedros_generados, semillas_interiores = voronoi_3d_puro(
        rango_caras=(4, 100),
        ancho=ANCHO,
        alto=ALTO,
        profundo=PROFUNDO,
        num_particulas=NUM_PARTICULAS,
    )

    print(
        f"✓ Generación exitosa: {len(poliedros_generados)} celdas de Voronoi continuas creadas."
    )
    print("Abriendo visor 3D interactivo...")

    visualizar_voronoi_solido(poliedros_generados, semillas_interiores)
