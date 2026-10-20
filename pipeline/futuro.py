"""
MKTBOARD.AR · fase 2 · FUTURO

Seis preguntas a 3 meses (91 días corridos), cada una con tres resultados posibles, y tres fuentes:
  mercado  → lo que descuentan los precios. Cuando el mercado da un valor esperado y no una distribución
             (no hay opciones sobre dólar, tasas ni riesgo país), se pasa a probabilidad con un supuesto explícito.
  historia → tasas base: qué pasó en todos los períodos de 3 meses y en los parecidos a hoy, con la cantidad de
             casos independientes (ventanas que no se pisan), que es lo que define el margen de error.
  vos      → lo carga el tablero y queda en tu navegador. No pasa por acá.

El tablero calcula las probabilidades con los umbrales que elijas a partir de los cuantiles que se guardan acá,
así que cambiar un umbral no exige volver a correr el proceso.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

H = 91                                  # horizonte en días corridos
Q = np.arange(0, 101, 1)                # cuantiles que se guardan (0 a 100)
INICIO_BANDAS = "2025-04-14"            # salida del cepo para personas y bandas cambiarias
DIV_SPX = 0.013                         # rendimiento por dividendos aproximado del S&P 500 (para el VIX)
MES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


# ════════════════════════════════════════════════════════════════════
# Herramientas
# ════════════════════════════════════════════════════════════════════
def limpia(s: pd.Series | None) -> pd.Series:
    if s is None:
        return pd.Series(dtype=float)
    s = pd.to_numeric(s, errors="coerce").dropna().sort_index()
    return s[~s.index.duplicated(keep="last")]


def a_futuro(s: pd.Series, h: int = H, modo: str = "pct") -> pd.Series:
    """Para cada fecha t con dato h días después: cambio entre t y t+h (en % o en puntos)."""
    s = limpia(s)
    if len(s) < 10:
        return pd.Series(dtype=float)
    destino = s.index + pd.Timedelta(days=h)
    ok = destino <= s.index[-1]
    if not ok.any():
        return pd.Series(dtype=float)
    v1 = s.reindex(destino[ok], method="ffill").values
    v0 = s[ok].values
    x = (v1 / v0 - 1) * 100 if modo == "pct" else v1 - v0
    return pd.Series(x, index=s.index[ok])


def cambio_previo(s: pd.Series, dias: int = 30, modo: str = "pct") -> pd.Series:
    """Cambio de los `dias` anteriores a cada fecha."""
    s = limpia(s)
    if len(s) < 10:
        return pd.Series(dtype=float)
    antes = s.reindex(s.index - pd.Timedelta(days=dias), method="ffill").values
    x = (s.values / antes - 1) * 100 if modo == "pct" else s.values - antes
    return pd.Series(x, index=s.index)


def casos_indep(fechas, h: int = H) -> int:
    """Cuántas ventanas de h días que no se pisan entran en esas fechas (el n que importa para el margen)."""
    n, ult = 0, None
    for f in fechas:
        if ult is None or (f - ult).days >= h:
            n, ult = n + 1, f
    return n


def resumen(x: pd.Series, criterio: str | None = None, minimo: int = 30, h: int = H):
    x = limpia(x)
    if len(x) < minimo:
        return None
    return {"cuantiles": [round(float(v), 4) for v in np.percentile(x.values, Q)],
            "n_obs": int(len(x)), "n_indep": casos_indep(list(x.index), h),
            "desde": str(x.index[0].date()), "hasta": str(x.index[-1].date()),
            "media": round(float(x.mean()), 4), "desvio": round(float(x.std()), 4), "criterio": criterio}


def tramo(v, corte):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return "sube" if v >= corte else ("baja" if v <= -corte else "quieto")


def alinear(cond: pd.Series, idx: pd.Index) -> pd.Series:
    """Lleva una serie diaria a otras fechas (último dato disponible)."""
    cond = limpia(cond)
    if not len(cond):
        return pd.Series(index=idx, dtype=float)
    return cond.reindex(cond.index.union(idx)).sort_index().ffill().reindex(idx)


def ultimos(s: pd.Series, dias: int = 400, dec: int = 4):
    s = limpia(s)
    if not len(s):
        return []
    s = s.loc[s.index[-1] - pd.Timedelta(days=dias):]
    return [[str(i.date()), round(float(v), dec)] for i, v in s.items()]


def ef(y_bono_equiv: float) -> float:
    """Tasa del Tesoro de EE.UU. (base semestral, en %) → efectiva anual (decimal)."""
    return (1 + y_bono_equiv / 200) ** 2 - 1


def mes_txt(p: pd.Period) -> str:
    return f"{MES[p.month - 1]}-{str(p.year)[2:]}"


def mes_cer(d: pd.Timestamp) -> pd.Period:
    """El CER del día d incorpora el IPC del mes anterior si d ≥ 16; si no, el de dos meses antes (metodología BCRA)."""
    return (pd.Period(d, "M") - (1 if d.day >= 16 else 2))


def habiles_atras(d: pd.Timestamp, n: int) -> pd.Timestamp:
    while n > 0:
        d -= pd.Timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


# ════════════════════════════════════════════════════════════════════
# Lo que descuenta el mercado
# ════════════════════════════════════════════════════════════════════
def curva_pesos(mk: dict, liq: pd.Timestamp):
    # las letras de menos de 14 días tienen tasas ruidosas (poco volumen, efecto de la liquidación): afuera
    curva = [dict(x) for x in (mk.get("lecap_curva") or []) if x.get("tem") is not None and x.get("dias", 0) >= 14]
    curva.sort(key=lambda x: x["dias"])
    for x in curva:
        x["venc"] = str((liq + pd.Timedelta(days=int(x["dias"]))).date())

    def tem_en(d):
        if not curva:
            return None
        if d <= curva[0]["dias"]:
            return curva[0]["tem"]
        if d >= curva[-1]["dias"]:
            return curva[-1]["tem"] if d - curva[-1]["dias"] <= 45 else None
        for a, b in zip(curva, curva[1:]):
            if a["dias"] <= d <= b["dias"]:
                return a["tem"] + (b["tem"] - a["tem"]) * (d - a["dias"]) / (b["dias"] - a["dias"])
        return None

    def crec(d):                         # crecimiento acumulado de $1 hasta el día d
        t = tem_en(d)
        return None if t is None else (1 + t) ** (d / 30)

    fwd = []
    for a, b in zip(curva, curva[1:]):
        if b["dias"] - a["dias"] < 10:
            continue
        g = ((1 + b["tem"]) ** (b["dias"] / 30) / (1 + a["tem"]) ** (a["dias"] / 30)) ** (30 / (b["dias"] - a["dias"])) - 1
        fwd.append({"desde": a["venc"], "hasta": b["venc"], "dias_desde": a["dias"], "dias_hasta": b["dias"], "tem": g})
    hoy_1m = tem_en(30)
    g91, g121 = crec(H), crec(H + 30)
    fwd_1m_3m = (g121 / g91 - 1) if (g91 and g121) else None
    return curva, fwd, tem_en, hoy_1m, fwd_1m_3m


def inflacion_implicita(px: dict, cfg: dict, cer_adel: pd.Series, liq: pd.Timestamp, tem_en):
    """LECAP contra BONCER cero cupón: qué inflación hace que den lo mismo.
    El CER ya publicado (llega hasta el 15 del mes que viene) se descuenta: lo que queda es inflación futura."""
    cer = limpia(cer_adel)
    if not len(cer) or not px or not cfg:
        return {"bonos": [], "tramos": [], "error": "faltan CER, precios o flujos"}
    ref = habiles_atras(liq, 10)
    cer_ref = float(cer.asof(ref)) if ref >= cer.index[0] else None
    K = cer.index[-1]                                     # último día con CER publicado
    cer_k = float(cer.iloc[-1])
    bc = cfg.get("bonos_cer") or {}
    items = bc.items() if isinstance(bc, dict) else [(b.get("ticker"), b) for b in bc if isinstance(b, dict)]
    bonos = []
    for tk, b in items:
        if not tk or not isinstance(b, dict) or tk not in px:
            continue
        fl = b.get("flujos") or []
        cero = (not fl) or (len(fl) == 1 and not (float(fl[0].get("tasa_interes") or 0) > 0))
        try:
            em, v = float(b.get("cer_emision")), pd.Timestamp(str(b.get("vencimiento"))[:10])
        except (TypeError, ValueError):
            continue
        if not cero or not em > 0 or cer_ref is None:
            continue
        d = (v - liq).days
        t = tem_en(d) if d >= 20 else None
        if t is None:
            continue
        vt = 100 * cer_ref / em                           # valor técnico al CER de referencia
        real = vt / px[tk]                                # retorno real bruto hasta el vencimiento
        nominal = (1 + t) ** (d / 30)
        fin = habiles_atras(v, 10)                        # el último CER que cobra el bono
        crec_cer = nominal / real                         # cuánto tiene que subir el CER entre ref y fin para empatar
        conocido = cer_k / cer_ref if fin > K else float(cer.asof(fin)) / cer_ref
        dias_inc = (fin - K).days
        if dias_inc < 25:
            continue                                      # casi todo su CER ya está publicado: lo que queda es ruido de precio
        mensual = (crec_cer / conocido) ** (30.4 / dias_inc) - 1
        real_tem = real ** (30 / d) - 1
        if not (-0.03 < real_tem < 0.04 and -0.01 < mensual < 0.10):
            continue                                      # precio o flujo raro: se descarta antes que mostrar basura
        bonos.append({"tk": tk, "venc": str(v.date()), "dias": d, "precio": px[tk], "tem_lecap": t,
                      "real_tem": real_tem, "real_tea": real ** (365 / d) - 1,
                      "crec_inc": crec_cer / conocido, "dias_inc": dias_inc, "fin": str(fin.date()),
                      "infl_mensual": mensual * 100,
                      "meses": [str(mes_cer(K + pd.Timedelta(days=1))), str(mes_cer(fin))]})
    bonos.sort(key=lambda x: x["dias"])
    tramos = []
    for a, b in zip(bonos, bonos[1:]):
        di = b["dias_inc"] - a["dias_inc"]
        if di < 20:
            continue
        m = (b["crec_inc"] / a["crec_inc"]) ** (30.4 / di) - 1
        if -0.01 < m < 0.10:
            tramos.append({"de": a["tk"], "a": b["tk"], "infl_mensual": m * 100,
                           "meses": [str(mes_cer(pd.Timestamp(a["fin"]) + pd.Timedelta(days=1))), b["meses"][1]]})
    return {"bonos": bonos, "tramos": tramos, "cer_ref": cer_ref, "ref": str(ref.date()),
            "cer_publicado_hasta": str(K.date())}


def mes_en(rango, mes: str) -> bool:
    return rango and rango[0] <= mes <= rango[1]


def fed(ust: pd.DataFrame | None):
    if ust is None or not len(ust):
        return None
    u = ust.dropna(how="all").ffill().iloc[-1]
    if pd.isna(u.get("ust3m")) or pd.isna(u.get("ust1")):
        return None
    e3, e1 = ef(float(u["ust3m"])), ef(float(u["ust1"]))
    f = ((1 + e1) / (1 + e3) ** 0.25) ** (1 / 0.75) - 1           # tasa a 3 meses dentro de 9 meses
    return {"fecha": str(ust.dropna(how="all").index[-1].date()),
            "t3m": float(u["ust3m"]), "t6m": float(u.get("ust6m")) if pd.notna(u.get("ust6m")) else None,
            "t1a": float(u["ust1"]), "t2a": float(u["ust2"]) if pd.notna(u.get("ust2")) else None,
            "cambio_12m_pb": (f - e3) * 1e4,
            "nota": "Aproximado con la curva de letras del Tesoro: incluye una pequeña prima por plazo."}


# ════════════════════════════════════════════════════════════════════
# Episodios parecidos a hoy
# ════════════════════════════════════════════════════════════════════
def analogos(rp, adr, spx, vix, oro, may_largo, n: int = 8):
    cal = pd.bdate_range("2000-01-03", limpia(rp).index[-1] if len(limpia(rp)) else pd.Timestamp.today())
    f = pd.DataFrame(index=cal)
    f["rp_log"] = np.log(alinear(rp, cal))
    f["rp_1m"] = alinear(cambio_previo(rp, 30), cal)
    f["adr_1m"] = alinear(cambio_previo(adr, 30), cal)
    f["vix"] = alinear(vix, cal)
    f["spx_1m"] = alinear(cambio_previo(spx, 30), cal)
    f = f.dropna()
    if len(f) < 500:
        return None
    hoy = f.iloc[-1]
    z = (f - f.mean()) / f.std()
    dist = np.sqrt(((z - z.iloc[-1]) ** 2).sum(axis=1))
    cand = dist[dist.index <= f.index[-1] - pd.Timedelta(days=H + 5)].sort_values()
    elegidos = []
    for fecha, dd in cand.items():
        if all(abs((fecha - e).days) > 120 for e, _ in elegidos):
            elegidos.append((fecha, dd))
        if len(elegidos) >= n:
            break
    out_series = {"adr": (adr, "pct"), "spx": (spx, "pct"), "oro": (oro, "pct"), "rp": (rp, "pct"),
                  "dolar": (may_largo, "pct")}
    fut = {k: a_futuro(s, H, m) for k, (s, m) in out_series.items() if s is not None}
    eps = []
    for fecha, dd in sorted(elegidos):
        e = {"fecha": str(fecha.date()), "distancia": round(float(dd), 2),
             "rp": round(float(math.exp(f.loc[fecha, "rp_log"]))), "rp_1m": round(float(f.loc[fecha, "rp_1m"]), 1),
             "adr_1m": round(float(f.loc[fecha, "adr_1m"]), 1), "vix": round(float(f.loc[fecha, "vix"]), 1),
             "spx_1m": round(float(f.loc[fecha, "spx_1m"]), 1), "despues": {}}
        for k, x in fut.items():
            v = x.asof(fecha) if len(x) and fecha >= x.index[0] else None
            e["despues"][k] = None if v is None or pd.isna(v) else round(float(v), 1)
        eps.append(e)
    med = {k: (round(float(np.nanmedian([e["despues"][k] for e in eps if e["despues"].get(k) is not None])), 1)
               if any(e["despues"].get(k) is not None for e in eps) else None) for k in fut}
    return {"hoy": {"fecha": str(f.index[-1].date()), "rp": round(float(math.exp(hoy["rp_log"]))),
                    "rp_1m": round(float(hoy["rp_1m"]), 1), "adr_1m": round(float(hoy["adr_1m"]), 1),
                    "vix": round(float(hoy["vix"]), 1), "spx_1m": round(float(hoy["spx_1m"]), 1)},
            "rasgos": ["nivel del riesgo país (log)", "cambio del riesgo país en el mes", "ADRs argentinos en el mes",
                       "VIX", "S&P 500 en el mes"],
            "episodios": eps, "mediana": med,
            "nota": "Se buscan desde 2000 los días más parecidos a hoy en esos cinco rasgos (cada uno en desvíos de su "
                    "historia) y se toman 8 separados por al menos 4 meses. Lo que pasó después no es un pronóstico: "
                    "muestra el rango de desenlaces de situaciones parecidas."}


# ════════════════════════════════════════════════════════════════════
# Las seis preguntas
# ════════════════════════════════════════════════════════════════════
def armar(*, hoy, liq, mk, px, cfg, rp_largo, ipc, tamar, badlar_largo, may, may_largo, cer_adel, ust,
          spx_largo, vix_largo, oro_largo, adr, rem) -> dict:
    P = []
    avisos = []

    def seguro(nombre, f, *a, defecto=None):
        try:
            return f(*a)
        except Exception as e:  # noqa: BLE001
            avisos.append(f"{nombre}: {type(e).__name__}: {str(e)[:150]}")
            return defecto

    curva, fwd, tem_en, hoy_1m, fwd_1m_3m = seguro("Curva de LECAP", curva_pesos, mk or {}, liq,
                                                   defecto=([], [], lambda d: None, None, None))
    infl = seguro("Inflación implícita", inflacion_implicita, px or {}, cfg or {}, cer_adel, liq, tem_en,
                  defecto={"bonos": [], "tramos": [], "error": "falló el cálculo"})
    vence = hoy + pd.Timedelta(days=H)

    rp, may_, s, v = limpia(rp_largo), limpia(may), limpia(spx_largo), limpia(vix_largo)
    a = limpia(adr)

    # 1 · Riesgo país ------------------------------------------------
    try:
        if len(rp):
            x = a_futuro(rp)
            c = cambio_previo(rp, 30)
            c_hoy = float(c.iloc[-1])
            t = tramo(c_hoy, 10)
            m = alinear(c, x.index).apply(lambda v: tramo(v, 10) == t)
            crit = {"sube": f"el riesgo país había subido 10% o más en el mes previo (hoy {c_hoy:+.0f}%)",
                    "baja": f"el riesgo país había bajado 10% o más en el mes previo (hoy {c_hoy:+.0f}%)",
                    "quieto": f"el riesgo país se había movido menos de 10% en el mes previo (hoy {c_hoy:+.0f}%)"}[t]
            ref = None
            if mk.get("ao27_tir") is not None and mk.get("fwd_post_eleccion") is not None:
                ref = (f"TIR AO27 {mk['ao27_tir'] * 100:.1f}% · AO28 {mk['ao28_tir'] * 100:.1f}% · tasa en dólares que el "
                       f"mercado pide para después de la elección de 2027: {mk['fwd_post_eleccion'] * 100:.1f}%"
                       + (f" (spread {mk['spread_fwd_pb']:.0f} pb)" if mk.get("spread_fwd_pb") is not None else ""))
            P.append({"id": "rp", "nombre": "Riesgo país", "clase": "Renta fija en dólares", "tipo": "cambio",
                      "unidad": "%", "umbrales": [-15, 15], "dec": 0, "unidad_nivel": "pb",
                      "hoy": {"valor": float(rp.iloc[-1]), "fecha": str(rp.index[-1].date())},
                      "historia": {"todos": resumen(x, "todos los períodos de 3 meses desde " + str(rp.index[0].year)),
                                   "parecidos": resumen(x[m], crit, minimo=20)},
                      "mercado": None,
                      "mercado_nota": "No hay opciones sobre el riesgo país: el mercado da una tasa, no una probabilidad.",
                      "referencia": ref,
                      "resolucion": {"serie": ultimos(rp, dec=0),
                                     "regla": "Riesgo país 91 días después del registro contra el del día del registro."}})
    except Exception as e:  # noqa: BLE001  — una pregunta rota no tumba a las demás
        avisos.append(f"1 · Riesgo país: {type(e).__name__}: {str(e)[:150]}")

    # 2 · Dólar contra el techo de la banda -------------------------
    try:
        if len(may_):
            band = may_.loc[INICIO_BANDAS:]
            x = a_futuro(band)
            r = np.log(band).diff().dropna()
            sigma = float(r.std() * math.sqrt(252)) if len(r) > 60 else None
            fwd91 = None
            pts = [(0, float(may_.iloc[-1]))] + [(l["dias"], l["fx"]) for l in (mk.get("lelink") or [])]
            for (d0, f0), (d1, f1) in zip(pts, pts[1:]):
                if d0 <= H <= d1 and d1 > d0:
                    fwd91 = f0 + (f1 - f0) * (H - d0) / (d1 - d0)
            P.append({"id": "techo", "nombre": "Dólar mayorista contra el techo de la banda", "clase": "Dólar",
                      "tipo": "barrera", "unidad": "%", "umbrales": None, "dec": 0, "unidad_nivel": "$",
                      "hoy": {"valor": float(may_.iloc[-1]), "fecha": str(may_.index[-1].date())},
                      "historia": {"todos": resumen(x, f"régimen de bandas (desde {INICIO_BANDAS}): cambio del mayorista "
                                                       "en 3 meses", minimo=20), "parecidos": None},
                      "historia_nota": "Solo cuenta el régimen de bandas: muy pocos casos. Mide si terminó por encima "
                                       "del techo, no si lo tocó en el camino (por eso es un piso de la probabilidad).",
                      "mercado": ({"dist": "barrera", "spot": float(may_.iloc[-1]), "fwd": fwd91, "T": H / 365,
                                   "sigma": sigma} if (fwd91 and sigma) else None),
                      "mercado_nota": ("Dólar que descuentan las LELINK a 91 días y la volatilidad del mayorista desde "
                                       "la salida del cepo. Supone que el dólar se mueve al azar alrededor de esa "
                                       "tendencia: no anticipa saltos de régimen."),
                      "resolucion": {"serie": ultimos(may_, dec=2),
                                     "regla": "Si el mayorista tocó el techo (según el techo que cargaste) en algún "
                                              "cierre dentro de los 91 días."}})
            if not fwd91:
                avisos.append("Sin dólar implícito a 91 días (faltan precios de LELINK).")
    except Exception as e:  # noqa: BLE001  — una pregunta rota no tumba a las demás
        avisos.append(f"2 · Dólar contra el techo de la banda: {type(e).__name__}: {str(e)[:150]}")

    # 3 · Inflación mensual ------------------------------------------
    try:
        ipc_m = limpia(ipc)
        if len(ipc_m) >= 24:
            ipc_m.index = ipc_m.index.to_period("M")
            ipc_m = ipc_m[~ipc_m.index.duplicated(keep="last")]
            ipc_m = ipc_m.reindex(pd.period_range(ipc_m.index[0], ipc_m.index[-1], freq="M"))   # sin huecos: shift(3) = 3 meses
            v0, ult_mes = float(ipc_m.iloc[-1]), ipc_m.index[-1]
            fechas = lambda ser: ser.set_axis(ser.index.to_timestamp(how="end").normalize())
            objetivo = ult_mes + 3                                    # el IPC que va a estar publicado al vencimiento
            fut = ipc_m.shift(-3).dropna()
            x_all = fut.copy()
            tend = (ipc_m - ipc_m.shift(3)).reindex(fut.index)
            t_hoy = float(ipc_m.iloc[-1] - ipc_m.iloc[-4])
            banda = (ipc_m.reindex(fut.index) - v0).abs() <= 0.4
            tr = tramo(t_hoy, 0.2)
            par = banda & tend.apply(lambda v: tramo(v, 0.2) == tr)
            hist_t = resumen(fechas(x_all[banda]), f"meses con inflación entre {v0 - 0.4:.1f}% y {v0 + 0.4:.1f}% "
                                            f"(hoy {v0:.1f}%) · el IPC de 3 meses después", minimo=12, h=85)
            hist_p = resumen(fechas(x_all[par]), {"sube": "además, la inflación venía subiendo (más de 0,2 pp en 3 meses)",
                                          "baja": "además, la inflación venía bajando (más de 0,2 pp en 3 meses)",
                                          "quieto": "además, la inflación venía estable (±0,2 pp en 3 meses)"}[tr],
                             minimo=12, h=85)
            obj = str(objetivo)
            tramo_m = next((t for t in infl["tramos"] if mes_en(t["meses"], obj)), None) \
                or next((b for b in infl["bonos"] if mes_en(b["meses"], obj)), None)
            sig = max(0.15, float(hist_t["desvio"])) if hist_t else 0.4
            mercado = None
            if tramo_m:
                mercado = {"dist": "normal", "media": round(tramo_m["infl_mensual"], 3), "sigma": round(sig, 3),
                           "meses": [mes_txt(pd.Period(tramo_m["meses"][0])), mes_txt(pd.Period(tramo_m["meses"][1]))],
                           "fuente": f"{tramo_m.get('de', tramo_m.get('tk'))}" + (f" → {tramo_m['a']}" if tramo_m.get("a") else "")}
            rem_obj = None
            for r in (rem or {}).get("ipc", []) if rem else []:
                if r.get("tipo") == "mensual" and str(r.get("desde", ""))[:7] == obj:
                    rem_obj = {"mediana": r.get("mediana"), "p10": r.get("p10"), "p90": r.get("p90"),
                               "origen": rem.get("origen"), "publicado": rem.get("fecha")}
            P.append({"id": "ipc", "nombre": f"Inflación de {mes_txt(objetivo)}", "clase": "Renta fija en pesos",
                      "tipo": "nivel", "unidad": "%", "umbrales": [1.5, 2.0], "dec": 1, "unidad_nivel": "%",
                      "objetivo_mes": obj,
                      "hoy": {"valor": v0, "fecha": str(ult_mes)},
                      "historia": {"todos": hist_t, "parecidos": hist_p},
                      "mercado": mercado,
                      "mercado_nota": ("Inflación que iguala el rendimiento de LECAP y BONCER en los meses que cubre ese "
                                       "tramo; el margen es el error típico histórico, no lo pone el mercado."
                                       if mercado else "No hay un BONCER cuyo CER cubra ese mes."),
                      "rem": rem_obj,
                      "resolucion": {"serie": [[str(i), round(float(v), 2)] for i, v in ipc_m.dropna().iloc[-24:].items()],
                                     "regla": f"IPC de {mes_txt(objetivo)} publicado por el INDEC."}})
    except Exception as e:  # noqa: BLE001  — una pregunta rota no tumba a las demás
        avisos.append(f"3 · Inflación mensual: {type(e).__name__}: {str(e)[:150]}")

    # 4 · Tasa en pesos ----------------------------------------------
    try:
        tam, bad = limpia(tamar), limpia(badlar_largo)
        base_tasa = tam if len(tam) else bad
        if len(bad) > 500 and len(base_tasa):
            b0 = float(bad.iloc[-1])
            x = a_futuro(bad, H, "abs")
            nivel = alinear(bad, x.index)
            banda = (nivel >= b0 * 0.5) & (nivel <= b0 * 1.5)
            c = cambio_previo(bad, 30, "abs")
            c_hoy = float(c.iloc[-1])
            tr = tramo(c_hoy, 1.0)
            par = banda & alinear(c, x.index).apply(lambda v: tramo(v, 1.0) == tr)
            hist_t = resumen(x[banda], f"BADLAR entre {b0 * 0.5:.0f}% y {b0 * 1.5:.0f}% (hoy {b0:.1f}%), desde "
                                       f"{bad.index[0].year} · cambio en 3 meses, en puntos")
            hist_p = resumen(x[par], {"sube": f"además, la tasa había subido 1 pp o más en el mes previo (hoy {c_hoy:+.1f})",
                                      "baja": f"además, la tasa había bajado 1 pp o más en el mes previo (hoy {c_hoy:+.1f})",
                                      "quieto": f"además, la tasa se había movido menos de 1 pp en el mes previo (hoy {c_hoy:+.1f})"}[tr],
                             minimo=20)
            mercado = None
            if hoy_1m is not None and fwd_1m_3m is not None and hist_t:
                mercado = {"dist": "normal", "media": round((fwd_1m_3m - hoy_1m) * 365 / 30 * 100, 3),
                           "sigma": round(max(1.0, float(hist_t["desvio"])), 3),
                           "tem_hoy": hoy_1m, "tem_fwd": fwd_1m_3m}
            P.append({"id": "tasa", "nombre": "Tasa en pesos (TAMAR)" if len(tam) else "Tasa en pesos (BADLAR)",
                      "clase": "Renta fija en pesos", "tipo": "cambio", "unidad": "pp", "umbrales": [-3, 3], "dec": 1,
                      "unidad_nivel": "% TNA",
                      "hoy": {"valor": float(base_tasa.iloc[-1]), "fecha": str(base_tasa.index[-1].date())},
                      "historia": {"todos": hist_t, "parecidos": hist_p},
                      "historia_nota": "La historia larga es de la BADLAR (la TAMAR existe desde 2024); se mueven casi igual.",
                      "mercado": mercado,
                      "mercado_nota": ("Cuánto cambia la tasa a 1 mes que descuenta la curva de LECAP para dentro de 3 meses, "
                                       "pasada a TNA. El margen es el error típico histórico, no lo pone el mercado."
                                       if mercado else "Sin curva de LECAP suficiente."),
                      "resolucion": {"serie": ultimos(base_tasa, dec=2),
                                     "regla": "Tasa 91 días después contra la del día del registro, en puntos."}})
    except Exception as e:  # noqa: BLE001  — una pregunta rota no tumba a las demás
        avisos.append(f"4 · Tasa en pesos: {type(e).__name__}: {str(e)[:150]}")

    # 5 · Acciones argentinas en dólares -----------------------------
    try:
        if len(a) > 1000:
            x = a_futuro(a)
            t = tramo(float(cambio_previo(rp, 30).iloc[-1]), 10) if len(rp) else None
            crit5 = {"sube": "el riesgo país había subido 10% o más en el mes previo",
                     "baja": "el riesgo país había bajado 10% o más en el mes previo",
                     "quieto": "el riesgo país se había movido menos de 10% en el mes previo"}.get(t)
            par = alinear(cambio_previo(rp, 30), x.index).apply(lambda v: tramo(v, 10) == t) if t else None
            P.append({"id": "adr", "nombre": "Acciones argentinas en dólares (ADRs)", "clase": "Acciones AR",
                      "tipo": "cambio", "unidad": "%", "umbrales": [-15, 15], "dec": 1, "unidad_nivel": "índice",
                      "hoy": {"valor": float(a.iloc[-1]), "fecha": str(a.index[-1].date())},
                      "historia": {"todos": resumen(x, f"canasta equiponderada de ADRs desde {a.index[0].year}"),
                                   "parecidos": resumen(x[par], crit5, minimo=20) if par is not None else None},
                      "mercado": None,
                      "mercado_nota": "No hay opciones líquidas sobre el índice argentino.",
                      "resolucion": {"serie": ultimos(a), "regla": "Canasta de ADRs 91 días después contra el día del registro."}})
    except Exception as e:  # noqa: BLE001  — una pregunta rota no tumba a las demás
        avisos.append(f"5 · Acciones argentinas en dólares: {type(e).__name__}: {str(e)[:150]}")

    # 6 · S&P 500 ----------------------------------------------------
    try:
        if len(s) > 1000:
            s50 = s.loc["1950":]
            x = a_futuro(s50)
            v0 = float(v.iloc[-1]) if len(v) else None
            par = None
            if v0:
                nv = alinear(v, x.index)
                par = (nv >= v0 * 0.8) & (nv <= v0 * 1.25)
            r = ef(float(ust["ust3m"].dropna().iloc[-1])) if (ust is not None and "ust3m" in ust and len(ust["ust3m"].dropna())) else 0.04
            P.append({"id": "spx", "nombre": "S&P 500", "clase": "CEDEARs", "tipo": "cambio", "unidad": "%",
                      "umbrales": [-10, 10], "dec": 0, "unidad_nivel": "pts",
                      "hoy": {"valor": float(s.iloc[-1]), "fecha": str(s.index[-1].date())},
                      "historia": {"todos": resumen(x, "todos los períodos de 3 meses desde 1950"),
                                   "parecidos": resumen(x[par], f"VIX entre {v0 * 0.8:.0f} y {v0 * 1.25:.0f} (hoy {v0:.1f}), "
                                                                "desde 1990", minimo=20) if par is not None else None},
                      "mercado": ({"dist": "lognormal", "sigma": v0 / 100, "T": H / 365, "mu": r - DIV_SPX, "vix": v0}
                                  if v0 else None),
                      "mercado_nota": ("Distribución que implica el VIX (volatilidad de las opciones a 30 días, extendida a "
                                       "3 meses). Las opciones suelen cobrar de más por la protección: el mercado tiende a "
                                       "darle más probabilidad a los movimientos grandes que la historia."),
                      "resolucion": {"serie": ultimos(s, dec=2), "regla": "S&P 500 91 días después contra el día del registro."}})
    except Exception as e:  # noqa: BLE001  — una pregunta rota no tumba a las demás
        avisos.append(f"6 · S&P 500: {type(e).__name__}: {str(e)[:150]}")

    mercado = {
        "pesos": {"curva": curva, "forwards": fwd, "tem_1m": hoy_1m, "tem_1m_en_3m": fwd_1m_3m},
        "inflacion": infl,
        "dolar": {"spot": float(may_.iloc[-1]) if len(may_) else None, "lelink": mk.get("lelink") or [],
                  "fwd91": next((q["mercado"]["fwd"] for q in P if q["id"] == "techo" and q["mercado"]), None),
                  "sigma": next((q["mercado"]["sigma"] for q in P if q["id"] == "techo" and q["mercado"]), None)},
        "riesgo": {k: mk.get(k) for k in ("ao27_tir", "ao28_tir", "fwd_post_eleccion", "spread_fwd_pb", "spread_pre_pb")},
        "fed": seguro("Fed", fed, ust),
        "spx": {"nivel": float(s.iloc[-1]) if len(s) else None, "vix": float(v.iloc[-1]) if len(v) else None},
        "rem": {"ipc": [r for r in ((rem or {}).get("ipc") or []) if r.get("tipo") == "mensual"],
                "origen": (rem or {}).get("origen"), "publicado": (rem or {}).get("fecha")} if rem else None,
    }
    return {"horizonte_dias": H, "hoy": str(hoy.date()), "vence": str(vence.date()), "preguntas": P,
            "mercado": mercado,
            "analogos": seguro("Episodios parecidos", analogos, rp, a, s, v, limpia(oro_largo), limpia(may_largo)),
            "avisos": avisos}
