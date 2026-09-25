import json
import math
import random
import numpy as np
import plotly.graph_objects as go

# Importamos el generador del Script 2
from Particulas_objetos import voronoi_3d_puro


def calcular_volumen_poliedro(poliedro, semilla):
    volumen_total = 0.0
    S = np.array(semilla)
    for cara in poliedro:
        if len(cara) < 3:
            continue
        V0 = np.array(cara[0])
        for j in range(1, len(cara) - 1):
            V1 = np.array(cara[j])
            V2 = np.array(cara[j + 1])
            mat = np.array([V0 - S, V1 - S, V2 - S])
            volumen_total += abs(np.linalg.det(mat)) / 6.0
    return volumen_total


def contraer_poliedro(poliedro, semilla, factor):
    S = np.array(semilla)
    poliedro_contraido = []
    for cara in poliedro:
        cara_c = []
        for pt in cara:
            P = np.array(pt)
            nuevo_P = S + factor * (P - S)
            cara_c.append(tuple(nuevo_P))
        poliedro_contraido.append(cara_c)
    return poliedro_contraido


def ejecutar_pipeline_simulacion(ruta_json="config_simulacion.json"):
    # 1. Leer parámetros exactos del espacio del usuario
    with open(ruta_json, "r") as f:
        config = json.load(f)

    dims = config["dimensiones"]
    ancho = dims["ancho"]
    alto = dims["alto"]
    profundo = dims["profundo"]

    clases_config = config["clases"]
    total_estimado = config["estimaciones"]["total_particulas_estimadas"]

    print("--- 1. CONFIGURACIÓN DEL ESPACIO ---")
    print(
        f"Dimensiones requeridas: {ancho} x {alto} x {profundo} mm (Volumen: {dims['volumen_total']:.1f} mm³)"
    )
    print(f"Total de partículas a generar: {total_estimado}")

    colores = ["#3498db", "#e67e22", "#2ecc71", "#9b59b6"]
    for i, c in enumerate(clases_config):
        c["color"] = colores[i % len(colores)]

    # 2. Generar el espacio con las dimensiones exactas solicitadas por el usuario
    print("\n--- 2. GENERANDO VORONOI CONTINUO ---")
    poliedros, semillas = voronoi_3d_puro(
        rango_caras=(3, 100),
        ancho=ancho,
        alto=alto,
        profundo=profundo,
        num_particulas=total_estimado,
    )
    print(f"✓ Voronoi generado ({len(poliedros)} celdas creadas).")

    # 3. Clasificación probabilística y contracción
    print("\n--- 3. CLASIFICACIÓN Y CONTRACCIÓN ---")
    particulas_finales = []

    for poliedro, semilla in zip(poliedros, semillas):
        num_caras = len(poliedro)
        vol_inicial = calcular_volumen_poliedro(poliedro, semilla)

        # Filtrar clases válidas según número de caras y volumen mínimo
        clases_elegibles = [
            c
            for c in clases_config
            if c["rango_caras"][0] <= num_caras <= c["rango_caras"][1]
            and vol_inicial >= c["volumen_minimo"]
        ]

        if not clases_elegibles:
            continue

        # Selección ponderada por la fracción volumétrica requerida
        pesos = [c["fraccion_volumen"] for c in clases_elegibles]
        clase_asignada = random.choices(clases_elegibles, weights=pesos, k=1)[0]

        # Factor de escala f = (V_objetivo / V_inicial)^(1/3)
        v_obj = clase_asignada["volumen_objetivo"]
        factor_f = (v_obj / vol_inicial) ** (1 / 3)
        factor_f = min(1.0, factor_f)

        poliedro_contraido = contraer_poliedro(poliedro, semilla, factor_f)

        particulas_finales.append(
            {
                "clase": clase_asignada["nombre"],
                "color": clase_asignada["color"],
                "semilla": semilla,
                "poliedro": poliedro_contraido,
                "volumen_inicial": vol_inicial,
                "volumen_final": vol_inicial * (factor_f**3),
                "factor_escala": factor_f,
                "num_caras": num_caras,
            }
        )

    return particulas_finales, (ancho, alto, profundo)


def visualizar_particulas(particulas, dimensiones):
    ancho, alto, profundo = dimensiones
    fig = go.Figure()
    edge_x, edge_y, edge_z = [], [], []

    for p in particulas:
        color = p["color"]
        poliedro = p["poliedro"]
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

                v_sig = cara[(i + 1) % len(cara)]
                edge_x.extend([v[0], v_sig[0], None])
                edge_y.extend([v[1], v_sig[1], None])
                edge_z.extend([v[2], v_sig[2], None])

            for idx in range(1, len(face_indices) - 1):
                i_indices.append(face_indices[0])
                j_indices.append(face_indices[idx])
                k_indices.append(face_indices[idx + 1])

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
                    name=f"Clase: {p['clase']}",
                    opacity=0.85,
                    flatshading=True,
                )
            )

    fig.add_trace(
        go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            line=dict(color="rgb(20, 20, 20)", width=2),
            showlegend=False,
        )
    )

    fig.update_layout(
        title=f"Espacio de Partículas ({ancho}x{alto}x{profundo} mm)",
        paper_bgcolor="rgb(245, 245, 245)",
        scene=dict(
            xaxis=dict(range=[0, ancho]),
            yaxis=dict(range=[0, alto]),
            zaxis=dict(range=[0, profundo]),
            bgcolor="rgb(245, 245, 245)",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    fig.show()


if __name__ == "__main__":
    particulas, dims = ejecutar_pipeline_simulacion("config_simulacion.json")

    resumen = {}
    for p in particulas:
        resumen[p["clase"]] = resumen.get(p["clase"], 0) + 1

    print("\n--- PARTICULAS RESULTANTES ---")
    for clase, cant in resumen.items():
        print(f"  • {clase}: {cant} partículas")

    visualizar_particulas(particulas, dims)