import json
import math
import random
import numpy as np

# Importamos el generador del Script 2
from Particulas_objetos import voronoi_3d_puro
from utilidades_particulas import repartir_cantidades


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


def _vistas_unicas(puntos, tolerancia=1e-8):
    unicos = []
    for punto in puntos:
        p = np.asarray(punto, dtype=float)
        if not any(np.linalg.norm(p - np.asarray(q)) <= tolerancia for q in unicos):
            unicos.append(tuple(p))
    return unicos


def _cortar_poliedro(poliedro, normal, limite):
    """Conserva normal·p <= limite y agrega la cara plana del corte."""
    nuevas_caras = []
    intersecciones = []
    normal = np.asarray(normal, dtype=float)
    for cara in poliedro:
        recortada = []
        for a, b in zip(cara, cara[1:] + cara[:1]):
            a = np.asarray(a, dtype=float)
            b = np.asarray(b, dtype=float)
            da = float(np.dot(normal, a) - limite)
            db = float(np.dot(normal, b) - limite)
            dentro_a, dentro_b = da <= 1e-9, db <= 1e-9
            if dentro_a:
                recortada.append(tuple(a))
            if dentro_a != dentro_b:
                punto = a + (da / (da - db)) * (b - a)
                recortada.append(tuple(punto))
                intersecciones.append(tuple(punto))
        recortada = _vistas_unicas(recortada)
        if len(recortada) >= 3:
            nuevas_caras.append(recortada)

    corte = _vistas_unicas(intersecciones)
    if len(corte) >= 3:
        centro = np.mean(corte, axis=0)
        n = normal / np.linalg.norm(normal)
        eje = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(n, eje)) > 0.9:
            eje = np.array([0.0, 1.0, 0.0])
        u = np.cross(n, eje)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        corte.sort(key=lambda p: math.atan2(np.dot(np.asarray(p) - centro, v), np.dot(np.asarray(p) - centro, u)))
        nuevas_caras.append(corte)
    return nuevas_caras


def _angulos_diedros(poliedro):
    """Calcula ángulos interiores entre caras que comparten una arista."""
    if not poliedro:
        return []
    vertices = _vistas_unicas([p for cara in poliedro for p in cara])
    centro = np.mean(vertices, axis=0)
    normales = []
    claves = []
    for cara in poliedro:
        arr = np.asarray(cara, dtype=float)
        normal = None
        for i in range(1, len(arr) - 1):
            candidato = np.cross(arr[i] - arr[0], arr[i + 1] - arr[0])
            norma = np.linalg.norm(candidato)
            if norma > 1e-10:
                normal = candidato / norma
                break
        if normal is None:
            normales.append(None)
        else:
            if np.dot(normal, np.mean(arr, axis=0) - centro) < 0:
                normal = -normal
            normales.append(normal)
        claves.append({tuple(np.round(p, 7)) for p in cara})
    angulos = []
    for i in range(len(poliedro)):
        if normales[i] is None:
            continue
        for j in range(i + 1, len(poliedro)):
            if normales[j] is None or len(claves[i] & claves[j]) < 2:
                continue
            coseno = float(np.clip(np.dot(normales[i], normales[j]), -1.0, 1.0))
            angulos.append(180.0 - math.degrees(math.acos(coseno)))
    return angulos


def _punto_segmento_mas_cercano(p, a, b):
    ab = b - a
    denominador = float(np.dot(ab, ab))
    t = 0.0 if denominador < 1e-15 else float(np.clip(np.dot(p - a, ab) / denominador, 0.0, 1.0))
    return a + t * ab


def _punto_triangulo_mas_cercano(p, a, b, c):
    normal = np.cross(b - a, c - a)
    norma2 = float(np.dot(normal, normal))
    if norma2 > 1e-15:
        proyeccion = p - normal * (np.dot(p - a, normal) / norma2)
        v0, v1, v2 = b - a, c - a, proyeccion - a
        d00, d01, d11 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v1, v1)
        d20, d21 = np.dot(v2, v0), np.dot(v2, v1)
        denominador = d00 * d11 - d01 * d01
        if abs(denominador) > 1e-15:
            v = (d11 * d20 - d01 * d21) / denominador
            w = (d00 * d21 - d01 * d20) / denominador
            if v >= 0 and w >= 0 and v + w <= 1:
                return proyeccion
    candidatos = [
        _punto_segmento_mas_cercano(p, a, b),
        _punto_segmento_mas_cercano(p, b, c),
        _punto_segmento_mas_cercano(p, c, a),
    ]
    return min(candidatos, key=lambda q: np.linalg.norm(p - q))


def _segmentos_mas_cercanos(p1, q1, p2, q2):
    u, v, w = q1 - p1, q2 - p2, p1 - p2
    a, b, c = np.dot(u, u), np.dot(u, v), np.dot(v, v)
    d, e = np.dot(u, w), np.dot(v, w)
    denominador = a * c - b * b
    s_num, s_den = denominador, denominador
    t_num, t_den = denominador, denominador
    if denominador < 1e-15:
        s_num, s_den, t_num, t_den = 0.0, 1.0, e, c
    else:
        s_num, t_num = b * e - c * d, a * e - b * d
        if s_num < 0:
            s_num, t_num, t_den = 0.0, e, c
        elif s_num > s_den:
            s_num, t_num, t_den = s_den, e + b, c
    if t_num < 0:
        t_num = 0.0
        s_num = min(max(-d, 0.0), a)
        s_den = a if s_num else 1.0
    elif t_num > t_den:
        t_num = t_den
        s_num = min(max(b - d, 0.0), a)
        s_den = a if s_num else 1.0
    s = 0.0 if abs(s_num) < 1e-15 else s_num / s_den
    t = 0.0 if abs(t_num) < 1e-15 else t_num / t_den
    return p1 + s * u, p2 + t * v


def _distancia_superficies(poliedro_a, poliedro_b):
    """Devuelve distancia y puntos más cercanos de dos mallas poligonales."""
    vertices_a = [np.asarray(p, dtype=float) for p in _vistas_unicas(
        [p for cara in poliedro_a for p in cara]
    )]
    vertices_b = [np.asarray(p, dtype=float) for p in _vistas_unicas(
        [p for cara in poliedro_b for p in cara]
    )]
    triangulos_a = []
    triangulos_b = []
    aristas_a, aristas_b = set(), set()
    for poliedro, triangulos, aristas in (
        (poliedro_a, triangulos_a, aristas_a),
        (poliedro_b, triangulos_b, aristas_b),
    ):
        for cara in poliedro:
            arr = [np.asarray(p, dtype=float) for p in cara]
            for i in range(1, len(arr) - 1):
                triangulos.append((arr[0], arr[i], arr[i + 1]))
            for a, b in zip(arr, arr[1:] + arr[:1]):
                ka, kb = tuple(np.round(a, 8)), tuple(np.round(b, 8))
                aristas.add(tuple(sorted((ka, kb))))
    map_a = {tuple(np.round(p, 8)): p for p in vertices_a}
    map_b = {tuple(np.round(p, 8)): p for p in vertices_b}
    aristas_a = [(map_a[a], map_a[b]) for a, b in aristas_a]
    aristas_b = [(map_b[a], map_b[b]) for a, b in aristas_b]
    mejor = (float("inf"), None, None)
    for p in vertices_a:
        for tri in triangulos_b:
            q = _punto_triangulo_mas_cercano(p, *tri)
            d = float(np.linalg.norm(p - q))
            if d < mejor[0]:
                mejor = (d, p, q)
    for p in vertices_b:
        for tri in triangulos_a:
            q = _punto_triangulo_mas_cercano(p, *tri)
            d = float(np.linalg.norm(p - q))
            if d < mejor[0]:
                mejor = (d, q, p)
    for a0, a1 in aristas_a:
        for b0, b1 in aristas_b:
            p, q = _segmentos_mas_cercanos(a0, a1, b0, b1)
            d = float(np.linalg.norm(p - q))
            if d < mejor[0]:
                mejor = (d, p, q)
    return mejor


def _corregir_colisiones_finales(particulas, dimensiones, max_iteraciones=100):
    """Revisa distancias reales entre mallas y separa pares que aún colisionen."""
    dimensiones = np.asarray(dimensiones, dtype=float)
    movidas = set()
    for _ in range(max_iteraciones):
        hubo_colision = False
        for i, a in enumerate(particulas):
            for j in range(i + 1, len(particulas)):
                b = particulas[j]
                distancia, punto_a, punto_b = _distancia_superficies(
                    a["poliedro"], b["poliedro"]
                )
                requerida = (a["dimension_lineal"] + b["dimension_lineal"]) / 4.0
                if distancia + 1e-7 >= requerida:
                    continue
                hubo_colision = True
                direccion = np.asarray(punto_b) - np.asarray(punto_a)
                norma = np.linalg.norm(direccion)
                if norma < 1e-10:
                    direccion = np.asarray(b["semilla"]) - np.asarray(a["semilla"])
                    norma = np.linalg.norm(direccion)
                if norma < 1e-10:
                    direccion = np.array([1.0, 0.0, 0.0])
                    norma = 1.0
                direccion /= norma
                paso = requerida - distancia + 1e-6
                opciones = [
                    [(j, direccion * paso)],
                    [(i, -direccion * paso)],
                    [(i, -direccion * paso / 2), (j, direccion * paso / 2)],
                ]
                corregida = False
                for movimientos in opciones:
                    candidatas = []
                    posible = True
                    for indice, delta in movimientos:
                        item = particulas[indice]
                        holgura_pared = item["separacion_minima_mm"]
                        nuevo_poliedro = [
                            [tuple(np.asarray(p) + delta) for p in cara]
                            for cara in item["poliedro"]
                        ]
                        vertices = [np.asarray(p) for cara in nuevo_poliedro for p in cara]
                        vertices_array = np.asarray(vertices)
                        ajuste_pared = np.zeros(3)
                        for eje in range(3):
                            minimo, maximo = vertices_array[:, eje].min(), vertices_array[:, eje].max()
                            if minimo < holgura_pared:
                                ajuste_pared[eje] += holgura_pared - minimo
                            if maximo + ajuste_pared[eje] > dimensiones[eje] - holgura_pared:
                                ajuste_pared[eje] -= (
                                    maximo + ajuste_pared[eje]
                                    - (dimensiones[eje] - holgura_pared)
                                )
                        if np.linalg.norm(ajuste_pared) > 0:
                            delta = delta + ajuste_pared
                            nuevo_poliedro = [
                                [tuple(np.asarray(p) + delta) for p in cara]
                                for cara in item["poliedro"]
                            ]
                            vertices = [np.asarray(p) for cara in nuevo_poliedro for p in cara]
                        if any(
                            np.any(p < holgura_pared - 1e-7)
                            or np.any(p > dimensiones - holgura_pared + 1e-7)
                            for p in vertices
                        ):
                            posible = False
                            break
                        candidatas.append((indice, nuevo_poliedro, np.asarray(item["semilla"]) + delta))
                    if not posible:
                        continue
                    for indice, nuevo_poliedro, nueva_semilla in candidatas:
                        particulas[indice]["poliedro"] = nuevo_poliedro
                        particulas[indice]["semilla"] = nueva_semilla
                        movidas.add(indice)
                    corregida = True
                    break
                if not corregida:
                    raise ValueError(
                        "Se detectó una colisión, pero no hay espacio para mover "
                        "las partículas sin invadir una pared."
                    )
        if not hubo_colision:
            break
    else:
        raise RuntimeError("No convergió la corrección final de colisiones.")

    # Última comprobación, incluida la holgura hasta las paredes para las movidas.
    distancias_minimas = [float("inf")] * len(particulas)
    for i, a in enumerate(particulas):
        vertices_a = [np.asarray(p) for cara in a["poliedro"] for p in cara]
        if any(np.any(p < -1e-7) or np.any(p > dimensiones + 1e-7) for p in vertices_a):
            raise RuntimeError("La corrección final dejó una partícula fuera de la caja.")
        if i in movidas and any(
            min(np.min([p[eje] for p in vertices_a]), dimensiones[eje] - np.max([p[eje] for p in vertices_a]))
            < a["separacion_minima_mm"] - 1e-6
            for eje in range(3)
        ):
            raise RuntimeError("La partícula movida no conserva la holgura solicitada a las paredes.")
        for indice_vecino, b in enumerate(particulas[i + 1:], start=i + 1):
            distancia, _, _ = _distancia_superficies(a["poliedro"], b["poliedro"])
            requerida = (a["dimension_lineal"] + b["dimension_lineal"]) / 4.0
            if distancia < requerida - 1e-6:
                raise RuntimeError("Quedó una colisión tras la pasada final.")
            distancias_minimas[i] = min(distancias_minimas[i], distancia)
            distancias_minimas[indice_vecino] = min(
                distancias_minimas[indice_vecino], distancia
            )
        minimos = np.min(np.asarray(vertices_a), axis=0)
        maximos = np.max(np.asarray(vertices_a), axis=0)
        a["distancia_pared_mas_cercana_mm"] = float(
            min(np.min(minimos), np.min(dimensiones - maximos))
        )
    for indice, item in enumerate(particulas):
        item["distancia_particula_mas_cercana_mm"] = (
            None if math.isinf(distancias_minimas[indice]) else distancias_minimas[indice]
        )


def _cumple_geometria(poliedro, clase):
    ncaras = len(poliedro)
    if not clase["rango_caras"][0] <= ncaras <= clase["rango_caras"][1]:
        return False
    amin, amax = clase.get("rango_angulos", [0.0, 180.0])
    angulos = _angulos_diedros(poliedro)
    return bool(angulos) and all(amin - 1e-6 <= a <= amax + 1e-6 for a in angulos)


def ajustar_geometria_particula(poliedro, clase, max_cortes=120, punto_interior=None):
    """Aplica cortes aleatorios hasta cumplir caras y ángulos o agotar intentos."""
    actual = poliedro
    amin, amax = clase.get("rango_angulos", [0.0, 180.0])
    for _ in range(max_cortes):
        if _cumple_geometria(actual, clase):
            return actual
        vertices = _vistas_unicas([p for cara in actual for p in cara])
        if len(vertices) < 4:
            break
        centro = np.asarray(
            punto_interior if punto_interior is not None else np.mean(vertices, axis=0),
            dtype=float,
        )
        caras_actuales = len(actual)
        objetivo_min, objetivo_max = clase["rango_caras"]
        candidatos = []
        for _ in range(16):
            normal = np.random.normal(size=3)
            norma = np.linalg.norm(normal)
            if norma < 1e-10:
                continue
            normal /= norma
            proyecciones = np.asarray(vertices) @ normal
            proyeccion_interior = float(np.dot(centro, normal))
            if proyeccion_interior > (float(proyecciones.min()) + float(proyecciones.max())) / 2:
                normal = -normal
                proyecciones = -proyecciones
                proyeccion_interior = -proyeccion_interior
            minimo, maximo = float(proyecciones.min()), float(proyecciones.max())
            if maximo - minimo < 1e-9:
                continue
            # Desplaza cortes hacia esquinas cuando sobran caras; en otro caso
            # usa planos interiores para añadir nuevas caras.
            if caras_actuales > objetivo_max:
                limite = maximo - random.uniform(0.03, 0.35) * (maximo - minimo)
            else:
                limite = random.uniform(minimo + 0.12 * (maximo - minimo), maximo - 0.12 * (maximo - minimo))
            limite = max(limite, proyeccion_interior + 1e-6 * (maximo - minimo))
            if limite >= maximo - 1e-8 * (maximo - minimo):
                continue
            candidato = _cortar_poliedro(actual, normal, limite)
            if len(candidato) < 4:
                continue
            ncaras = len(candidato)
            exceso_caras = max(0, objetivo_min - ncaras) + max(0, ncaras - objetivo_max)
            angulos = _angulos_diedros(candidato)
            violaciones = sum(a < amin or a > amax for a in angulos)
            candidatos.append((exceso_caras * 1000 + violaciones, candidato))
        if not candidatos:
            break
        candidatos.sort(key=lambda elemento: elemento[0])
        actual = candidatos[0][1]
    if not _cumple_geometria(actual, clase):
        raise ValueError(
            f"No fue posible ajustar una partícula para la clase {clase['nombre']} "
            "con los rangos de caras y ángulos solicitados."
        )
    return actual


def _ajustar_posiciones_en_caja(preparadas, dimensiones, max_iteraciones=2000):
    """Traslada partículas fuera de la caja hacia adentro y resuelve colisiones."""
    dimensiones = np.asarray(dimensiones, dtype=float)
    posiciones = []
    limites_inferiores = []
    limites_superiores = []

    for item in preparadas:
        semilla = np.asarray(item["semilla"], dtype=float)
        factor = item["factor_escala"]
        offsets = np.asarray([
            factor * (np.asarray(p, dtype=float) - semilla)
            for cara in item["poliedro"] for p in cara
        ])
        minimo = offsets.min(axis=0)
        maximo = offsets.max(axis=0)
        espacio = item["dimension_lineal"] / 2.0
        posicion = semilla.copy()
        limite_inferior = -minimo
        limite_superior = dimensiones - maximo

        # Si cruza una pared, desplazar perpendicularmente hasta dejar la
        # separación completa. Las correcciones de esquina se aplican por eje.
        for eje in range(3):
            if posicion[eje] + minimo[eje] < 0:
                limite_inferior[eje] = espacio - minimo[eje]
                posicion[eje] = limite_inferior[eje]
            if posicion[eje] + maximo[eje] > dimensiones[eje]:
                limite_superior[eje] = dimensiones[eje] - espacio - maximo[eje]
                posicion[eje] = limite_superior[eje]
        if np.any(limite_inferior > limite_superior + 1e-9):
            raise ValueError(
                f"La partícula {item['clase']['nombre']} no cabe con la holgura "
                "de pared solicitada."
            )
        posicion = np.minimum(np.maximum(posicion, limite_inferior), limite_superior)
        posiciones.append(posicion)
        limites_inferiores.append(limite_inferior)
        limites_superiores.append(limite_superior)

    # El movimiento hacia una pared puede acercar dos partículas. Separarlas
    # como cuerpos rígidos y volver a respetar los límites de la caja.
    for _ in range(max_iteraciones):
        hubo_ajuste = False
        for i, a in enumerate(preparadas):
            for j in range(i + 1, len(preparadas)):
                b = preparadas[j]
                delta = posiciones[j] - posiciones[i]
                distancia = float(np.linalg.norm(delta))
                radio_a = a["radio_base"] * a["factor_escala"]
                radio_b = b["radio_base"] * b["factor_escala"]
                holgura = (a["dimension_lineal"] + b["dimension_lineal"]) / 4.0
                requerida = radio_a + radio_b + holgura
                if distancia + 1e-8 >= requerida:
                    continue
                if distancia < 1e-12:
                    direccion = np.array([1.0, 0.0, 0.0])
                else:
                    direccion = delta / distancia
                desplazamiento = (requerida - distancia) / 2.0
                posicion_anterior_i = posiciones[i].copy()
                posicion_anterior_j = posiciones[j].copy()
                posiciones[i] -= direccion * desplazamiento
                posiciones[j] += direccion * desplazamiento
                posiciones[i] = np.minimum(
                    np.maximum(posiciones[i], limites_inferiores[i]), limites_superiores[i]
                )
                posiciones[j] = np.minimum(
                    np.maximum(posiciones[j], limites_inferiores[j]), limites_superiores[j]
                )
                hubo_ajuste |= (
                    np.linalg.norm(posiciones[i] - posicion_anterior_i) > 1e-10
                    or np.linalg.norm(posiciones[j] - posicion_anterior_j) > 1e-10
                )
        if not hubo_ajuste:
            break

    for i, a in enumerate(preparadas):
        a["desplazamiento"] = posiciones[i] - np.asarray(a["semilla"], dtype=float)
        a["posicion_final"] = posiciones[i]
        for j in range(i + 1, len(preparadas)):
            b = preparadas[j]
            distancia = float(np.linalg.norm(posiciones[i] - posiciones[j]))
            radio_a = a["radio_base"] * a["factor_escala"]
            radio_b = b["radio_base"] * b["factor_escala"]
            holgura = (a["dimension_lineal"] + b["dimension_lineal"]) / 4.0
            if distancia - radio_a - radio_b < holgura - 1e-6:
                raise ValueError(
                    "No se pueden conservar simultáneamente la separación entre "
                    "partículas y la holgura requerida respecto a las paredes."
                )


def _ajustar_tamanos_y_separacion(preparadas):
    """Escala dentro de tolerancia y garantiza holgura usando esferas envolventes."""
    for item in preparadas:
        clase = item["clase"]
        volumen = float(clase["volumen_objetivo"])
        tolerancia = float(clase.get("tolerancia_tamano_porcentaje", 0.0)) / 100.0
        objetivo = (volumen / item["volumen_ajustado"]) ** (1.0 / 3.0)
        minimo = objetivo * (1.0 - tolerancia) ** (1.0 / 3.0)
        maximo = objetivo
        if minimo > maximo + 1e-9:
            raise ValueError(
                f"La partícula {clase['nombre']} no cabe dentro de la caja "
                "con el volumen mínimo permitido por su tolerancia."
            )
        item["factor_minimo"] = minimo
        item["factor_maximo"] = maximo
        item["radio_base"] = max(
            np.linalg.norm(np.asarray(p, dtype=float) - np.asarray(item["semilla"], dtype=float))
            for cara in item["poliedro"] for p in cara
        )
        item["dimension_lineal"] = volumen ** (1.0 / 3.0)

    progreso = 1.0
    for i, a in enumerate(preparadas):
        for b in preparadas[i + 1:]:
            distancia = float(np.linalg.norm(np.asarray(a["semilla"]) - np.asarray(b["semilla"])))
            separacion_requerida = (a["dimension_lineal"] + b["dimension_lineal"]) / 4.0
            disponible = distancia - separacion_requerida
            radio_minimo = (
                a["factor_minimo"] * a["radio_base"]
                + b["factor_minimo"] * b["radio_base"]
            )
            if disponible < radio_minimo - 1e-8:
                # Dejar el problema de posición al solver de traslación; el
                # tamaño usa el mínimo permitido si los centros iniciales chocan.
                progreso = 0.0
                continue
            crecimiento = (
                (a["factor_maximo"] - a["factor_minimo"]) * a["radio_base"]
                + (b["factor_maximo"] - b["factor_minimo"]) * b["radio_base"]
            )
            if crecimiento > 0:
                progreso = min(progreso, (disponible - radio_minimo) / crecimiento)

    progreso = max(0.0, min(1.0, progreso))
    for item in preparadas:
        item["factor_escala"] = item["factor_minimo"] + progreso * (
            item["factor_maximo"] - item["factor_minimo"]
        )

def ejecutar_pipeline_simulacion(ruta_json="config_simulacion.json"):
    # 1. Leer parámetros exactos del espacio del usuario
    with open(ruta_json, "r") as f:
        config = json.load(f)

    dims = config["dimensiones"]
    ancho = dims["ancho"]
    alto = dims["alto"]
    profundo = dims["profundo"]
    volumen_dominio = float(ancho) * float(alto) * float(profundo)
    if min(ancho, alto, profundo) <= 0:
        raise ValueError("Las dimensiones del volumen en el JSON deben ser positivas.")
    if "volumen_total" in dims and not math.isclose(
        float(dims["volumen_total"]), volumen_dominio, rel_tol=1e-6, abs_tol=1e-6
    ):
        raise ValueError("El volumen_total del JSON no coincide con sus dimensiones.")

    clases_config = config["clases"]
    total_estimado = config["total_particulas"]
    if isinstance(total_estimado, bool) or not isinstance(total_estimado, int):
        raise ValueError("El total global de partículas en el JSON debe ser un entero.")
    if total_estimado < 1:
        raise ValueError("El total global de partículas debe ser mayor que cero.")
    if not clases_config:
        raise ValueError("El JSON debe incluir al menos una clase de partículas.")
    nombres_clase = [str(c["nombre"]).strip() for c in clases_config]
    if len(set(nombres_clase)) != len(nombres_clase):
        raise ValueError("Cada clase del JSON debe tener un nombre único.")
    total_declarado = config.get("estimaciones", {}).get("total_particulas_estimadas")
    if total_declarado is not None and total_declarado != total_estimado:
        raise ValueError("El total estimado en el JSON no coincide con el total global solicitado.")
    suma_fracciones = sum(float(c["fraccion_aparicion"]) for c in clases_config)
    if not math.isclose(suma_fracciones, 1.0, rel_tol=1e-3):
        raise ValueError("Los porcentajes de aparición del JSON deben sumar 100 %.")
    conteos_esperados = repartir_cantidades(
        total_estimado, [float(c["fraccion_aparicion"]) for c in clases_config]
    )
    detalle_estimado = config.get("estimaciones", {}).get("detalle_clases", {})
    for clase, conteo in zip(clases_config, conteos_esperados):
        estimado_clase = detalle_estimado.get(clase["nombre"], {}).get("cantidad_asignada")
        if estimado_clase is not None and estimado_clase != conteo:
            raise ValueError(
                f"El conteo derivado de la clase {clase['nombre']} no coincide con estimaciones."
            )
    volumen_objetivo_total = sum(
        cantidad * float(clase["volumen_objetivo"])
        for clase, cantidad in zip(clases_config, conteos_esperados)
    )

    volumen_ponderado = total_estimado * sum(
        (float(c["fraccion_aparicion"]) / suma_fracciones) * float(c["volumen_objetivo"])
        for c in clases_config
    )
    volumen_margen = volumen_ponderado * 1.15
    if volumen_margen > volumen_dominio + 1e-6:
        raise ValueError(
            "El filtro de ocupación falla: el volumen ponderado con margen del "
            f"15 % ({volumen_margen:.6g} mm³) supera la caja ({volumen_dominio:.6g} mm³)."
        )
    if volumen_objetivo_total <= 0:
        raise ValueError("El volumen objetivo total de partículas debe ser positivo.")

    print("--- 1. CONFIGURACIÓN DEL ESPACIO ---")
    print(
        f"Dimensiones requeridas: {ancho} x {alto} x {profundo} mm (Volumen: {dims['volumen_total']:.1f} mm³)"
    )
    print(f"Total de partículas a generar: {total_estimado}")
    print(f"Volumen ponderado total de partículas: {volumen_ponderado:.3f} mm³")
    print(f"Volumen ponderado con margen 15 %: {volumen_margen:.3f} mm³")
    print(
        "Volumen libre nominal en la caja: "
        f"{max(0.0, ancho * alto * profundo - volumen_objetivo_total):.3f} mm³"
    )

    for c in clases_config:
        c.setdefault("color", "#3498db")

    # 2. Generar el espacio con las dimensiones exactas solicitadas por el usuario
    print("\n--- 2. GENERANDO VORONOI CONTINUO ---")
    poliedros, semillas = voronoi_3d_puro(ruta_json=ruta_json)
    print(f"✓ Voronoi generado ({len(poliedros)} celdas creadas).")

    if len(poliedros) != total_estimado or len(semillas) != total_estimado:
        raise RuntimeError(
            f"El generador devolvió {len(poliedros)} partículas; se solicitaron {total_estimado}."
        )

    # Repartir el total global según los porcentajes y emparejar celdas con tamaños afines.
    celdas = [
        (poliedro, semilla, calcular_volumen_poliedro(poliedro, semilla))
        for poliedro, semilla in zip(poliedros, semillas)
    ]
    celdas.sort(key=lambda elemento: elemento[2])
    restantes = {
        c["nombre"]: conteo for c, conteo in zip(clases_config, conteos_esperados)
    }
    clases_por_nombre = {c["nombre"]: c for c in clases_config}
    asignaciones = []
    for _ in range(total_estimado):
        elegibles = [clases_por_nombre[n] for n, cantidad in restantes.items() if cantidad > 0]
        elegida = random.choices(
            elegibles, weights=[c["fraccion_aparicion"] for c in elegibles], k=1
        )[0]
        asignaciones.append(elegida)
        restantes[elegida["nombre"]] -= 1

    # 3. Ajuste de geometría y tamaños
    print("\n--- 3. CLASIFICACIÓN, CORTES Y SEPARACIÓN ---")
    particulas_finales = []
    preparadas = []

    for (poliedro, semilla, vol_inicial), clase_asignada in zip(celdas, asignaciones):
        poliedro_ajustado = ajustar_geometria_particula(
            poliedro, clase_asignada, punto_interior=semilla
        )
        vol_ajustado = calcular_volumen_poliedro(poliedro_ajustado, semilla)
        if vol_ajustado <= 0:
            raise ValueError(f"La partícula de la clase {clase_asignada['nombre']} tiene volumen cero.")
        preparadas.append({
            "poliedro": poliedro_ajustado,
            "semilla": semilla,
            "clase": clase_asignada,
            "volumen_inicial": vol_inicial,
            "volumen_ajustado": vol_ajustado,
        })

    _ajustar_tamanos_y_separacion(preparadas)
    _ajustar_posiciones_en_caja(preparadas, (ancho, alto, profundo))
    for item in preparadas:
        poliedro = item["poliedro"]
        semilla_original = np.asarray(item["semilla"], dtype=float)
        semilla = np.asarray(item["posicion_final"], dtype=float)
        clase_asignada = item["clase"]
        vol_inicial = item["volumen_inicial"]
        v_obj = float(clase_asignada["volumen_objetivo"])
        factor_f = item["factor_escala"]
        poliedro_final = [
            [tuple(semilla + factor_f * (np.asarray(p) - semilla_original)) for p in cara]
            for cara in poliedro
        ]
        puntos_finales = [np.asarray(p) for cara in poliedro_final for p in cara]
        limites = np.array([ancho, alto, profundo], dtype=float)
        if any(np.any(p < -1e-7) or np.any(p > limites + 1e-7) for p in puntos_finales):
            raise ValueError(
                f"El ajuste de tamaño de una partícula de {clase_asignada['nombre']} "
                "la extiende fuera de la caja."
            )
        num_caras = len(poliedro_final)
        volumen_final = calcular_volumen_poliedro(poliedro_final, semilla)
        tolerancia = float(clase_asignada.get("tolerancia_tamano_porcentaje", 0.0)) / 100.0
        minimo_volumen = v_obj * (1.0 - tolerancia)
        maximo_volumen = v_obj
        if not minimo_volumen - 1e-6 <= volumen_final <= maximo_volumen + 1e-6:
            raise RuntimeError(
                f"El volumen final de una partícula de {clase_asignada['nombre']} "
                "quedó fuera de la tolerancia solicitada."
            )
        if not _cumple_geometria(poliedro_final, clase_asignada):
            raise RuntimeError(
                f"La partícula de {clase_asignada['nombre']} no cumple sus caras/ángulos."
            )

        particulas_finales.append(
            {
                "clase": clase_asignada["nombre"],
                "color": clase_asignada["color"],
                "semilla": semilla,
                "desplazamiento_pared_mm": item["desplazamiento"].tolist(),
                "poliedro": poliedro_final,
                "volumen_inicial": vol_inicial,
                "volumen_final": volumen_final,
                "factor_escala": factor_f,
                "num_caras": num_caras,
                "angulos_diedros": _angulos_diedros(poliedro_final),
                "separacion_minima_mm": item["dimension_lineal"] / 2.0,
                "dimension_lineal": item["dimension_lineal"],
            }
        )

    _corregir_colisiones_finales(particulas_finales, (ancho, alto, profundo))

    return particulas_finales, (ancho, alto, profundo)


