"""
MKTBOARD.AR · fase 3 · CARTERA

Series semanales (cierre del viernes) desde 2002 de las cinco clases, medidas en dólares. El tablero las pasa a
pesos reales con el dólar libre y el CER, y con eso calcula en tu navegador:
  · la caída máxima de cualquier composición (rebalanceando cada semana), en dólares y en pesos reales;
  · la base por paridad de riesgo con el tope de caída que elijas;
  · los escenarios a 3 meses (riesgo país × S&P 500) con lo que hizo cada clase en cada combinación.

Proxies (largos, para que entren 2002, 2008, 2018-19, 2020 y 2023):
  acc  Acciones AR       canasta equiponderada de ADRs argentinos (Yahoo)
  ced  CEDEARs           70% SPY + 30% oro (GLD; antes de 2004, futuros de oro)
  rfp  Renta fija pesos  pesos capitalizando a la BADLAR (proxy de plazo fijo y LECAP corta)
  rfu  Renta fija USD    bono soberano modelo: 8 años, cupón 7%, rendimiento = Treasury 10 años + riesgo país,
                         con piso de 30 (recupero), sin cupón cuando el riesgo país pasa 3.000 (default) y sin
                         ganancia de precio en las semanas de los canjes de 2005 y 2020 (cambia el índice, no el bono)
  usd  Dólar             billete: 0% en dólares
Todo es público. Tu cartera y tu bitácora nunca pasan por acá.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DESDE_C = "2002-03-01"
CUPON, PLAZO, PISO, RP_DEFAULT = 0.07, 8, 30.0, 3000.0


def limpia(s):
    if s is None:
        return pd.Series(dtype=float)
    s = pd.to_numeric(s, errors="coerce").dropna().sort_index()
    s = s[s > 0] if len(s) else s
    return s[~s.index.duplicated(keep="last")]


def sin_escalones(s: pd.Series, umbral: float = 0.5) -> tuple[pd.Series, int]:
    """Algunas series viejas vienen redondeadas (el blue de 2011-2012 está en pesos enteros: 4, 5, 6, 7) y quedan como
    escalones: semanas planas y saltos de 17-25% en un día. En los años en que más de la mitad de los días la serie no
    se mueve, se reemplaza por una curva suave que pasa por el promedio de cada mes (promediar valores redondeados
    recupera bastante bien el valor de fondo). Los años con datos buenos no se tocan."""
    s = limpia(s)
    if len(s) < 40:
        return s, 0
    quieta = (s.diff() == 0).groupby(s.index.year).mean()
    malos = set(quieta[quieta > umbral].index)
    if not malos:
        return s, 0
    mal = pd.Series(s.index.year.isin(list(malos)), index=s.index)
    anclas = {i: float(v) for i, v in s[~mal].items()}
    for _, mes in s[mal].groupby([s[mal].index.year, s[mal].index.month]):
        anclas[mes.index[len(mes) // 2]] = float(np.exp(np.log(mes).mean()))
    ancla = pd.Series(anclas).sort_index()
    ns = lambda idx: pd.DatetimeIndex(idx).as_unit("ns").asi8.astype(float)
    interp = np.exp(np.interp(ns(s.index), ns(ancla.index), np.log(ancla.values.astype(float))))
    out = s.copy()
    out[mal.values] = interp[mal.values]
    return out, int(mal.sum())


def encadenar(fuentes: list, cal: pd.DatetimeIndex, nombres: list | None = None, diag: dict | None = None) -> pd.Series:
    """Índice armado con los movimientos de la mejor fuente disponible (en orden de prioridad).
    Cada paso entre dos días con datos se mide con UNA sola fuente que tenga dato en los dos extremos: así una fuente
    con datos salteados no mete escalones y el cambio de fuente no mete saltos por diferencias de nivel."""
    nombres = nombres or [f"f{i}" for i in range(len(fuentes))]
    series, alisados = [], {}
    for n, s in zip(nombres, fuentes):
        x, k = sin_escalones(s)
        series.append(x)
        if k:
            alisados[n] = k
    usadas = [(n, s) for n, s in zip(nombres, series) if len(s) >= 2]
    if not usadas:
        return pd.Series(dtype=float)
    X = pd.concat({n: s for n, s in usadas}, axis=1).sort_index()
    X = X[X.notna().any(axis=1)]
    lr = np.log(X / X.shift(1))                        # NaN si a la fuente le falta alguno de los dos extremos
    paso = pd.Series(np.nan, index=X.index)
    elegida = pd.Series(None, index=X.index, dtype=object)
    for n, _ in usadas:                                # la primera fuente con los dos extremos manda
        libre = paso.isna() & lr[n].notna()
        paso[libre] = lr[n][libre]
        elegida[libre] = n
    sin = int(paso.iloc[1:].isna().sum())
    idx = np.exp(paso.fillna(0.0).cumsum())
    if diag is not None:
        e = elegida.dropna()
        diag["pasos_sin_fuente"] = sin
        diag["datos_alisados"] = alisados
        diag["por_anio"] = {str(y): {k: round(float(v), 3) for k, v in g.value_counts(normalize=True).items()}
                            for y, g in e.groupby(e.index.year)}
    return idx.reindex(cal.union(idx.index)).sort_index().ffill().reindex(cal)


# Cambios de composición del índice de riesgo país tras los canjes de deuda (los bonos en default salen y entran los
# nuevos): el salto del índice no es una ganancia para quien tenía los bonos, así que esas semanas no suman precio.
CANJES = [("2005-06-01", "2005-07-15"), ("2020-09-01", "2020-09-30")]


def precio_bono(y: np.ndarray) -> np.ndarray:
    y = np.maximum(y, 1e-4)
    p = CUPON * 100 * (1 - (1 + y) ** -PLAZO) / y + 100 * (1 + y) ** -PLAZO
    return np.maximum(p, PISO)


def armar(*, rp_largo, adr, spy, spx, oro_gld, oro_fut, tnx, badlar, may_largo, ccl_largo, blue_largo, cer_largo) -> dict:
    hoy = max(limpia(x).index[-1] for x in (rp_largo, adr, spx) if len(limpia(x)))
    cal = pd.bdate_range(DESDE_C, hoy)
    sem = pd.date_range(pd.Timestamp(DESDE_C) + pd.offsets.Week(weekday=4), hoy, freq="W-FRI")
    if sem[-1] < hoy:
        sem = sem.append(pd.DatetimeIndex([hoy]))           # la última semana, aunque no haya terminado
    a_sem = lambda s: s.reindex(cal.union(s.index)).sort_index().ffill().reindex(sem)
    notas = []

    # Dólar libre: CCL, si no hay blue, si no mayorista (antes del cepo eran casi lo mismo)
    diag_fx = {}
    fx_i = encadenar([ccl_largo, blue_largo, may_largo], cal, ["CCL", "blue", "mayorista"], diag_fx)
    ult = next((float(limpia(s).iloc[-1]) for s in (ccl_largo, blue_largo, may_largo) if len(limpia(s))), None)
    if not len(fx_i) or ult is None:
        raise ValueError("sin dólar para convertir")
    fx = a_sem(fx_i * (ult / float(fx_i.iloc[-1])))

    cer = limpia(cer_largo)
    if len(cer) < 500:
        raise ValueError("sin CER largo")
    cer_w = a_sem(cer)

    # Acciones AR: ADRs
    acc = a_sem(limpia(adr))

    # CEDEARs: 70% SPY (con dividendos) + 30% oro, rebalanceado cada semana
    eq = a_sem(encadenar([spy, spx], cal))
    oro = a_sem(encadenar([oro_gld, oro_fut], cal))
    if limpia(spy).empty:
        notas.append("CEDEARs: sin SPY, se usó el S&P 500 sin dividendos.")
    r_ced = 0.7 * eq.pct_change() + 0.3 * oro.pct_change()
    ced = (1 + r_ced.fillna(0)).cumprod()

    # Pesos a BADLAR, en pesos → a dólares con el dólar libre
    b = limpia(badlar)
    cal_c = pd.date_range(DESDE_C, hoy)
    tasa = b.reindex(cal_c.union(b.index)).sort_index().ffill().reindex(cal_c).bfill()
    pesos = (1 + tasa.shift(1).fillna(tasa.iloc[0]) / 100 / 365).cumprod()
    rfp = a_sem(pesos) / fx

    # Bono soberano modelo en dólares
    t = limpia(tnx)
    if len(t) and float(t.median()) > 20:                    # Yahoo a veces cotiza ^TNX ×10
        t = t / 10
    ust = a_sem(t) if len(t) else pd.Series(4.0, index=sem)
    ust = ust.fillna(4.0)
    rp = a_sem(limpia(rp_largo))
    y = (ust.values / 100) + rp.values / 1e4
    P = precio_bono(y)
    cup = np.where(rp.values[:-1] < RP_DEFAULT, CUPON * 100 / 52 / P[:-1], 0.0)
    precio = P[1:] / P[:-1] - 1
    for a_, b_ in CANJES:
        precio[(sem[1:] >= a_) & (sem[1:] <= b_)] = 0.0
    r_rfu = np.r_[0.0, precio + cup]
    rfu = pd.Series(np.cumprod(1 + np.nan_to_num(r_rfu)), index=sem)

    usd = pd.Series(1.0, index=sem)
    spx_w = a_sem(limpia(spx))

    df = pd.DataFrame({"acc": acc, "ced": ced, "rfp": rfp, "rfu": rfu, "usd": usd, "fx": fx, "cer": cer_w,
                       "rp": rp, "spx": spx_w}, index=sem)
    inicio = df.dropna().index[0] if len(df.dropna()) else None
    if inicio is None:
        raise ValueError("no hay una fecha con todas las series")
    df = df.loc[inicio:].ffill()
    for c in ("acc", "ced", "rfp", "rfu", "usd"):
        df[c] = df[c] / df[c].iloc[0] * 100

    def lista(c, dec=6):
        return [float(f"{v:.{dec}g}") for v in df[c].values]
    return {
        "desde": str(df.index[0].date()), "hasta": str(df.index[-1].date()), "semanas": [str(d.date()) for d in df.index],
        "usd": {c: lista(c) for c in ("acc", "ced", "rfp", "rfu", "usd")},
        "fx": lista("fx"), "cer": lista("cer"), "rp": lista("rp", 5), "spx": lista("spx"),
        "proxies": {
            "acc": "Canasta equiponderada de ADRs argentinos (Yahoo)",
            "ced": "70% SPY + 30% oro (GLD; antes de 2004, futuros de oro), rebalanceado cada semana",
            "rfp": "Pesos capitalizando a la BADLAR de bancos privados (un plazo fijo minorista paga algo menos)",
            "rfu": "Bono soberano modelo de 8 años y cupón 7%: rinde Treasury 10 años + riesgo país; piso de 30 "
                   "(lo que se recuperó en los defaults), sin cupón con riesgo país arriba de 3.000 y sin ganancia de "
                   "precio en los canjes de 2005 y 2020 (ahí cambia la composición del índice, no el valor del bono)",
            "usd": "Dólar billete: 0% en dólares",
        },
        "conversion": "En pesos reales: valor en dólares × dólar libre (CCL; antes, blue; antes de 2011, mayorista) ÷ CER.",
        "fx_fuentes": diag_fx,
        "notas": notas,
    }
