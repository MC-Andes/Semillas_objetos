import math
import random
import json
import numpy as np


def cargar_parametros_json(ruta_json):
    """Lee las dimensiones y el total global de partículas desde el JSON."""
    with open(ruta_json, "r", encoding="utf-8") as archivo:
        config = json.load(archivo)
    try:
        dimensiones = config["dimensiones"]
        ancho = float(dimensiones["ancho"])
        alto = float(dimensiones["alto"])
        profundo = float(dimensiones["profundo"])
        num_particulas = config["total_particulas"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"El JSON no contiene dimensiones o el total global válido: {error}") from error

    if (
        isinstance(num_particulas, bool)
        or not isinstance(num_particulas, int)
        or min(ancho, alto, profundo) <= 0
        or num_particulas <= 0
    ):
        raise ValueError("El JSON debe solicitar dimensiones y cantidad de partículas mayores que cero.")
    volumen_calculado = ancho * alto * profundo
    volumen_declarado = dimensiones.get("volumen_total")
    if volumen_declarado is not None and not math.isclose(
        float(volumen_declarado), volumen_calculado, rel_tol=1e-6, abs_tol=1e-6
    ):
        raise ValueError("El volumen_total del JSON no coincide con sus dimensiones.")
    estimacion = config.get("estimaciones", {}).get("total_particulas_estimadas")
    if estimacion is not None and estimacion != num_particulas:
        raise ValueError("El total global no coincide con el total registrado en estimaciones.")
    return ancho, alto, profundo, num_particulas


def generar_espacio_perturbado(
    num_particulas,
    ancho_total,
    alto_total,
    profundo_total,
    rango_perturbacion=0.3,
):
    """Crea semillas en una malla regular y las perturba dentro de la caja.

    ``rango_perturbacion`` es una fracción del espaciado local por eje:
    0.3 permite un desplazamiento de hasta ±30 % del espaciado.
    """
    if num_particulas <= 0:
        raise ValueError("num_particulas debe ser mayor que cero.")
    if min(ancho_total, alto_total, profundo_total) <= 0:
        raise ValueError("Las dimensiones de la caja deben ser mayores que cero.")
    if not 0 <= rango_perturbacion < 0.5:
        raise ValueError("rango_perturbacion debe estar entre 0 y 0.5.")

    volumen = ancho_total * alto_total * profundo_total
    s = (volumen / num_particulas) ** (1 / 3)

    cols = max(1, math.ceil(ancho_total / s))
    rows = max(1, math.ceil(alto_total / s))
    layers = max(1, math.ceil(profundo_total / s))

    espaciado_x = ancho_total / cols
    espaciado_y = alto_total / rows
    espaciado_z = profundo_total / layers

    # Seleccionar exactamente el número solicitado de celdas del mallado.
    indices_malla = np.linspace(0, cols * rows * layers - 1, num_particulas)
    indices_malla = np.unique(np.rint(indices_malla).astype(int))
    # La malla tiene al menos num_particulas celdas, y linspace proporciona
    # exactamente esa cantidad de índices distintos.
    puntos = []
    for indice in indices_malla:
        i = int(indice // (rows * layers))
        resto = int(indice % (rows * layers))
        j = resto // layers
        k = resto % layers
        base_x = (i + 0.5) * espaciado_x
        base_y = (j + 0.5) * espaciado_y
        base_z = (k + 0.5) * espaciado_z
        x = base_x + random.uniform(-rango_perturbacion, rango_perturbacion) * espaciado_x
        y = base_y + random.uniform(-rango_perturbacion, rango_perturbacion) * espaciado_y
        z = base_z + random.uniform(-rango_perturbacion, rango_perturbacion) * espaciado_z
        puntos.append((x, y, z))

    if len(puntos) != num_particulas:
        raise RuntimeError("No fue posible crear exactamente el número de semillas solicitado.")

    # Todas las semillas están dentro de la caja; no se crean puntos fantasma externos.
    return puntos, puntos


def _recortar_cara_plano(cara, normal, limite, tolerancia=1e-9):
    """Recorta un polígono para conservar los puntos con dot(normal, p) <= limite."""
    if not cara:
        return [], []
    recortada = []
    intersecciones = []
    for a, b in zip(cara, cara[1:] + cara[:1]):
        da = float(np.dot(normal, a) - limite)
        db = float(np.dot(normal, b) - limite)
        dentro_a = da <= tolerancia
        dentro_b = db <= tolerancia
        if dentro_a:
            recortada.append(tuple(a))
        if dentro_a != dentro_b:
            t = da / (da - db)
            punto = tuple(np.asarray(a) + t * (np.asarray(b) - np.asarray(a)))
            recortada.append(punto)
            intersecciones.append(punto)
    return recortada, intersecciones


def _recortar_poliedro_plano(caras, normal, limite):
    """Recorta un poliedro convexo y agrega la nueva cara de corte."""
    nuevas_caras = []
    corte = []
    for cara in caras:
        recortada, cruces = _recortar_cara_plano(cara, normal, limite)
        if len(recortada) >= 3:
            nuevas_caras.append(recortada)
        corte.extend(cruces)

    unicos = []
    for punto in corte:
        if not any(np.linalg.norm(np.asarray(punto) - np.asarray(q)) < 1e-8 for q in unicos):
            unicos.append(punto)
    if len(unicos) >= 3:
        centro = np.mean(unicos, axis=0)
        n = np.asarray(normal, dtype=float)
        n_norm = np.linalg.norm(n)
        if n_norm > 0:
            n /= n_norm
            eje = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(n, eje)) > 0.9:
                eje = np.array([0.0, 1.0, 0.0])
            u = np.cross(n, eje)
            u /= np.linalg.norm(u)
            v = np.cross(n, u)
            unicos.sort(key=lambda p: math.atan2(np.dot(np.asarray(p) - centro, v), np.dot(np.asarray(p) - centro, u)))
            nuevas_caras.append(unicos)
    return nuevas_caras


def _celda_voronoi_recortada(
    semilla, semillas, ancho, alto, profundo, competidores=None
):
    """Interseca la caja con los semiespacios de Voronoi de las otras semillas."""
    x, y, z = semilla
    caras = [
        [(0, 0, 0), (0, alto, 0), (0, alto, profundo), (0, 0, profundo)],
        [(ancho, 0, 0), (ancho, 0, profundo), (ancho, alto, profundo), (ancho, alto, 0)],
        [(0, 0, 0), (0, 0, profundo), (ancho, 0, profundo), (ancho, 0, 0)],
        [(0, alto, 0), (ancho, alto, 0), (ancho, alto, profundo), (0, alto, profundo)],
        [(0, 0, 0), (ancho, 0, 0), (ancho, alto, 0), (0, alto, 0)],
        [(0, 0, profundo), (0, alto, profundo), (ancho, alto, profundo), (ancho, 0, profundo)],
    ]
    p = np.asarray(semilla, dtype=float)
    sitios = semillas if competidores is None else sorted(competidores)
    for otra in sitios:
        if otra == semilla:
            continue
        q = np.asarray(otra, dtype=float)
        normal = 2.0 * (q - p)
        limite = float(np.dot(q, q) - np.dot(p, p))
        caras = _recortar_poliedro_plano(caras, normal, limite)
        if not caras:
            break
    # El redondeo numérico de intersecciones puede producir excesos microscópicos.
    limites = np.array([ancho, alto, profundo], dtype=float)
    return [
        [tuple(np.clip(np.asarray(p, dtype=float), 0.0, limites)) for p in cara]
        for cara in caras
    ]


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


def delaunay_3d_con_params(points, ancho, alto, profundo):
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


def voronoi_3d_puro(ruta_json="config_simulacion.json", rango_perturbacion=0.3):
    ancho, alto, profundo, num_particulas = cargar_parametros_json(ruta_json)
    puntos_todos, puntos_interiores = generar_espacio_perturbado(
        num_particulas, ancho, alto, profundo, rango_perturbacion
    )
    # Bowyer–Watson identifica los vecinos de Delaunay de cada semilla.
    tetraedros = delaunay_3d_con_params(puntos_todos, ancho, alto, profundo)
    vecinos = {p: set() for p in puntos_interiores}
    for tetra in tetraedros:
        for punto in tetra:
            if punto in vecinos:
                vecinos[punto].update(otro for otro in tetra if otro != punto)

    poliedros = []
    for p in puntos_interiores:
        candidatos = vecinos[p] or {otro for otro in puntos_interiores if otro != p}
        celda = _celda_voronoi_recortada(
            p, puntos_interiores, ancho, alto, profundo, candidatos
        )

        # Completar candidatos si una triangulación degenerada omitió algún
        # vecino: todo vértice debe estar del lado de Voronoi de esta semilla.
        while True:
            vertices = [np.asarray(v, dtype=float) for cara in celda for v in cara]
            faltantes = []
            p_arr = np.asarray(p, dtype=float)
            for otro in puntos_interiores:
                if otro == p or otro in candidatos:
                    continue
                q = np.asarray(otro, dtype=float)
                normal = 2.0 * (q - p_arr)
                limite = float(np.dot(q, q) - np.dot(p_arr, p_arr))
                if any(np.dot(normal, v) > limite + 1e-7 for v in vertices):
                    faltantes.append(otro)
            if not faltantes:
                break
            candidatos.update(faltantes)
            celda = _celda_voronoi_recortada(
                p, puntos_interiores, ancho, alto, profundo, candidatos
            )
        poliedros.append(celda)

    return poliedros, puntos_interiores

