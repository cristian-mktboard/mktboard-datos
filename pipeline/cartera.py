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
                         con piso de 30 (recupero) y sin cupón cuando el riesgo país pasa 3.000 (default)
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


def encadenar(fuentes: list, cal: pd.DatetimeIndex) -> pd.Series:
    """Índice armado con los retornos de la mejor fuente disponible cada día (en orden de prioridad).
    Cada fuente se completa hacia adelante dentro de su propio rango, así no se cuentan dos veces los días."""
    rets = []
    for s in fuentes:
        s = limpia(s)
        if len(s) < 2:
            continue
        x = s.reindex(cal.union(s.index)).sort_index().ffill().reindex(cal)
        x[(cal < s.index[0]) | (cal > s.index[-1] + pd.Timedelta(days=7))] = np.nan
        rets.append(np.log(x).diff())
    if not rets:
        return pd.Series(dtype=float)
    r = rets[0]
    for x in rets[1:]:
        r = r.fillna(x)
    r = r.fillna(0.0)
    primero = min(limpia(s).index[0] for s in fuentes if len(limpia(s)) >= 2)
    r[cal <= primero] = 0.0
    return np.exp(r.cumsum())


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
    fx_i = encadenar([ccl_largo, blue_largo, may_largo], cal)
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
    r_rfu = np.r_[0.0, P[1:] / P[:-1] - 1 + cup]
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
                   "(lo que se recuperó en los defaults) y sin cupón con riesgo país arriba de 3.000",
            "usd": "Dólar billete: 0% en dólares",
        },
        "conversion": "En pesos reales: valor en dólares × dólar libre (CCL; antes, blue; antes de 2011, mayorista) ÷ CER.",
        "notas": notas,
    }
