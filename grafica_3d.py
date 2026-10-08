"""Visualización 3D de las partículas generadas."""

import plotly.graph_objects as go


def visualizar_particulas(particulas, dimensiones):
    ancho, alto, profundo = dimensiones
    fig = go.Figure()
    edge_x, edge_y, edge_z = [], [], []

    for particula in particulas:
        poliedro = particula["poliedro"]
        x, y, z = [], [], []
        indices_i, indices_j, indices_k = [], [], []
        indice_vertice = 0

        for cara in poliedro:
            if len(cara) < 3:
                continue
            indices_cara = []
            for indice, vertice in enumerate(cara):
                x.append(vertice[0])
                y.append(vertice[1])
                z.append(vertice[2])
                indices_cara.append(indice_vertice)
                indice_vertice += 1

                siguiente = cara[(indice + 1) % len(cara)]
                edge_x.extend([vertice[0], siguiente[0], None])
                edge_y.extend([vertice[1], siguiente[1], None])
                edge_z.extend([vertice[2], siguiente[2], None])

            for indice in range(1, len(indices_cara) - 1):
                indices_i.append(indices_cara[0])
                indices_j.append(indices_cara[indice])
                indices_k.append(indices_cara[indice + 1])

        if x:
            fig.add_trace(
                go.Mesh3d(
                    x=x,
                    y=y,
                    z=z,
                    i=indices_i,
                    j=indices_j,
                    k=indices_k,
                    color=particula["color"],
                    name=f"Clase: {particula['clase']}",
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
        title=f"Espacio de partículas ({ancho} × {alto} × {profundo} mm)",
        paper_bgcolor="rgb(245, 245, 245)",
        scene=dict(
            xaxis=dict(title="X (mm)", range=[0, ancho]),
            yaxis=dict(title="Y (mm)", range=[0, alto]),
            zaxis=dict(title="Z (mm)", range=[0, profundo]),
            bgcolor="rgb(245, 245, 245)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    fig.show()
