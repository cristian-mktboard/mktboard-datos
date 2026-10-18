#!/usr/bin/env python3
"""
MKTBOARD.AR · proceso diario · fase 1 (contexto + impacto)

Qué hace, en orden:
  1. Baja datos públicos: argentinadatos, BCRA (API v4.0), data912, datos.gob.ar, FRED y Yahoo (yfinance).
  2. Arma el VECTOR DE ESTADO: cada variable con su nivel, cambios, percentil de 3 años y serie corta.
  3. Arma un ÍNDICE por clase de activo (acciones AR, CEDEARs, renta fija en pesos, renta fija en dólares, dólar)
     y mide su SENSIBILIDAD semanal a cada variable, en dólares (CCL) y en pesos reales (CER).
  4. Calcula el risk-off score y lo que descuenta el mercado (LECAP, LELINK, AO27/AO28).
  5. Escribe data/contexto.json (lo lee el tablero) y agrega una fila a data/historial.csv.

Cada fuente está aislada: si una falla, el resto sigue y el error queda registrado en "fuentes".
Solo datos de mercado. Nada personal.
"""
from __future__ import annotations

import io
import json
import math
import time
import pathlib
import datetime as dt
import warnings

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
VERSION = "1.1"
RAIZ = pathlib.Path(__file__).resolve().parents[1]
DATA = RAIZ / "data"
DATA.mkdir(exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (mktboard-ar; proceso diario de uso personal)"}
AHORA = pd.Timestamp.now(tz="America/Argentina/Buenos_Aires")
HOY = AHORA.normalize().tz_localize(None)
DESDE = (HOY - pd.DateOffset(years=4)).date().isoformat()
FUENTES: dict[str, dict] = {}

AD = "https://api.argentinadatos.com/v1"
BCRA = "https://api.bcra.gob.ar/estadisticas/v4.0/Monetarias"
D9 = "https://data912.com"
CFG_URL = "https://raw.githubusercontent.com/arisbdar/rendimientos-ar/main/public/config.json"

PANEL_LIDER = ["ALUA", "BBAR", "BMA", "BYMA", "CEPU", "COME", "CRES", "ECOG", "EDN", "GGAL", "LOMA",
               "METR", "MIRG", "PAMP", "SUPV", "TGNO4", "TGSU2", "TRAN", "TXAR", "VALO", "YPFD"]
CANASTA_CEDEAR = {"SPY": 0.70, "GLD": 0.30}
BONOS_USD = ["AL30D", "GD30D"]


# ════════════════════════════════════════════════════════════════════
# Utilidades de descarga
# ════════════════════════════════════════════════════════════════════
def http_get(url: str, intentos: int = 3, timeout: int = 45, bcra: bool = False) -> requests.Response:
    ultimo = None
    for i in range(intentos):
        try:
            try:
                r = requests.get(url, headers=UA, timeout=timeout)
            except requests.exceptions.SSLError:
                if not bcra:
                    raise
                # La API del BCRA tuvo problemas de cadena de certificados: datos públicos, se reintenta sin verificar.
                import urllib3
                urllib3.disable_warnings()
                r = requests.get(url, headers=UA, timeout=timeout, verify=False)
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001
            ultimo = e
            time.sleep(2 * (i + 1))
    raise ultimo  # type: ignore[misc]


def get_json(url: str, **kw):
    return http_get(url, **kw).json()


def a_serie(datos, fk: str, vk: str) -> pd.Series:
    d = {}
    for x in datos or []:
        v = x.get(vk)
        if v in (None, "", "."):
            continue
        try:
            d[pd.Timestamp(str(x.get(fk))[:10])] = float(v)
        except (TypeError, ValueError):
            continue
    s = pd.Series(d, dtype=float).sort_index()
    return s[~s.index.duplicated(keep="last")]


def ultima_fecha(obj):
    if isinstance(obj, pd.Series) and len(obj):
        return str(obj.index[-1].date())
    if isinstance(obj, pd.DataFrame) and len(obj):
        return str(obj.index[-1].date())
    if isinstance(obj, dict) and obj.get("fecha"):
        return str(obj["fecha"])
    return None


def fuente(nombre: str):
    """Aísla cada descarga: si falla, registra el error y devuelve None."""
    def deco(f):
        def envoltura(*a, **k):
            t0 = time.time()
            try:
                out = f(*a, **k)
                vacio = out is None or (hasattr(out, "__len__") and len(out) == 0)
                FUENTES[nombre] = {"ok": not vacio, "ultimo": ultima_fecha(out),
                                   "seg": round(time.time() - t0, 1), **({"error": "sin datos"} if vacio else {})}
                return None if vacio else out
            except Exception as e:  # noqa: BLE001
                FUENTES[nombre] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:180]}",
                                   "seg": round(time.time() - t0, 1)}
                return None
        return envoltura
    return deco


# ════════════════════════════════════════════════════════════════════
# Fuentes
# ════════════════════════════════════════════════════════════════════
@fuente("Riesgo país · argentinadatos")
def s_riesgo():
    return a_serie(get_json(f"{AD}/finanzas/indices/riesgo-pais"), "fecha", "valor").loc[DESDE:]


def _dolar(casa):
    return a_serie(get_json(f"{AD}/cotizaciones/dolares/{casa}"), "fecha", "venta").loc[DESDE:]


@fuente("Dólar CCL · argentinadatos")
def s_ccl():
    return _dolar("contadoconliqui")


@fuente("Dólar MEP · argentinadatos")
def s_mep():
    return _dolar("bolsa")


@fuente("Dólar oficial · argentinadatos")
def s_oficial():
    return _dolar("oficial")


@fuente("Inflación INDEC · argentinadatos")
def s_ipc():
    return a_serie(get_json(f"{AD}/finanzas/indices/inflacion"), "fecha", "valor")


@fuente("REM · argentinadatos (o el último guardado)")
def s_rem():
    """El endpoint a veces devuelve [] (pasó el 18/10/2026). Se guarda cada REM bueno en data/rem.json
    y, si la API viene vacía, se usa el último guardado avisando de qué informe es."""
    cache = DATA / "rem.json"
    try:
        nuevo = _rem_api()
        if nuevo["ipc"]:
            cache.write_text(json.dumps(nuevo, ensure_ascii=False, indent=1), encoding="utf-8")
            return {**nuevo, "origen": "api"}
    except Exception:  # noqa: BLE001
        pass
    if cache.exists():
        return {**json.loads(cache.read_text(encoding="utf-8")), "origen": "guardado"}
    raise RuntimeError("argentinadatos devolvió el REM vacío y todavía no hay uno guardado")


def _rem_api():
    arr = get_json(f"{AD}/finanzas/rem/ultimo")
    arr = arr if isinstance(arr, list) else (arr or {}).get("datos", [])
    filas = [x for x in arr if not x.get("muestra") or "todos" in str(x.get("muestra")).lower()]
    def pick(clave, excluir=None):
        out = []
        for x in filas:
            ind = str(x.get("indicador", "")).lower()
            if clave in ind and not (excluir and excluir in ind):
                out.append({"periodo": x.get("periodo"), "tipo": x.get("periodoTipo"),
                            "desde": str(x.get("periodoDesde") or "")[:10], "mediana": x.get("mediana"),
                            "p10": x.get("percentil10"), "p90": x.get("percentil90")})
        return out
    ipc = pick("nivel general", "núcleo")
    return {"informe": (filas[0] if filas else {}).get("informe"), "fecha": (filas[0] if filas else {}).get("fecha"),
            "ipc": ipc, "tc": pick("tipo de cambio")}


@fuente("Confianza en el Gobierno · argentinadatos")
def s_icg():
    return a_serie(get_json(f"{AD}/politica/indices/confianza-gobierno"), "fecha", "valor")


CATALOGO = None


def id_bcra(regex: str, por_defecto: int) -> int:
    global CATALOGO
    import re
    try:
        if CATALOGO is None:
            CATALOGO = get_json(BCRA, bcra=True).get("results", [])
        for v in CATALOGO:
            if re.search(regex, str(v.get("descripcion", "")), re.I):
                return int(v["idVariable"])
    except Exception:  # noqa: BLE001
        pass
    return por_defecto


def bcra_serie(idv: int, desde: str = DESDE) -> pd.Series:
    hasta = HOY.date().isoformat()
    filas, off = [], 0
    for _ in range(10):
        j = get_json(f"{BCRA}/{idv}?desde={desde}&hasta={hasta}&limit=3000&offset={off}", bcra=True)
        res = j.get("results", [])
        det = res[0]["detalle"] if res and isinstance(res[0], dict) and "detalle" in res[0] else res
        if filas and det and det[0] == filas[0]:
            break                                         # la API devolvió la misma página: no seguir
        filas += det
        if len(det) < 3000:
            break
        off += 3000
    return a_serie(filas, "fecha", "valor")


@fuente("Dólar mayorista · BCRA")
def s_mayorista():
    return bcra_serie(id_bcra(r"tipo de cambio mayorista", 5))


@fuente("CER · BCRA")
def s_cer():
    return bcra_serie(id_bcra(r"coeficiente de estabilizaci", 30))


@fuente("TAMAR bancos privados · BCRA")
def s_tamar():
    return bcra_serie(id_bcra(r"tamar de bancos privados", 44))


@fuente("BADLAR bancos privados · BCRA")
def s_badlar():
    return bcra_serie(id_bcra(r"badlar en pesos de bancos privados|badlar de bancos privados", 7))


@fuente("Reservas · BCRA")
def s_reservas():
    return bcra_serie(id_bcra(r"^reservas internacionales", 1))


@fuente("Compras netas de divisas · BCRA")
def s_compras():
    return bcra_serie(id_bcra(r"compras netas de divisas", 47))


@fuente("Préstamos al sector privado en pesos · BCRA")
def s_prestamos():
    return bcra_serie(117)  # 'Préstamos otorgados al sector privado' en moneda local


@fuente("EMAE · datos.gob.ar (INDEC)")
def s_emae():
    j = get_json("https://apis.datos.gob.ar/series/api/series/?ids=143.3_NO_PR_2004_A_21&limit=1000&format=json")
    s = pd.Series({pd.Timestamp(f): float(v) for f, v in j.get("data", []) if v is not None}).sort_index()
    return s


def d9_hist(tipo: str, tk: str) -> pd.Series:
    return a_serie(get_json(f"{D9}/historical/{tipo}/{tk}"), "date", "c").loc[DESDE:]


@fuente("Acciones panel líder · data912")
def s_panel():
    cols = {}
    for tk in PANEL_LIDER:
        try:
            s = d9_hist("stocks", tk)
            if len(s) > 200:
                cols[tk] = s
        except Exception:  # noqa: BLE001
            continue
    return pd.DataFrame(cols).sort_index() if cols else None


@fuente("CEDEARs SPY y GLD · data912")
def s_cedears():
    cols = {tk: d9_hist("cedears", tk) for tk in CANASTA_CEDEAR}
    return pd.DataFrame(cols).sort_index()


@fuente("Bonos AL30D y GD30D · data912")
def s_bonos_usd():
    cols = {}
    for tk in BONOS_USD:
        try:
            cols[tk] = d9_hist("bonds", tk)
        except Exception:  # noqa: BLE001
            continue
    return pd.DataFrame(cols).sort_index() if cols else None


def fred(serie_id: str) -> pd.Series:
    txt = http_get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={serie_id}", intentos=2, timeout=25).text
    df = pd.read_csv(io.StringIO(txt))
    fcol = df.columns[0]
    s = pd.to_numeric(df[serie_id], errors="coerce")
    s.index = pd.to_datetime(df[fcol])
    return s.dropna().loc[DESDE:]


def tesoro_eeuu() -> pd.DataFrame:
    """Curva diaria del Tesoro de EE.UU. (misma base que FRED/H.15), un CSV por año."""
    partes = []
    for y in range(int(DESDE[:4]), HOY.year + 1):
        url = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/"
               f"{y}/all?type=daily_treasury_yield_curve&field_tdr_date_value={y}&page&_format=csv")
        df = pd.read_csv(io.StringIO(http_get(url, intentos=2, timeout=30).text))
        df.index = pd.to_datetime(df["Date"], format="%m/%d/%Y")
        partes.append(df[["1 Yr", "2 Yr", "10 Yr"]].apply(pd.to_numeric, errors="coerce"))
    d = pd.concat(partes).sort_index()
    d.columns = ["ust1", "ust2", "ust10"]
    return d[~d.index.duplicated(keep="last")].dropna().loc[DESDE:]


@fuente("Treasuries 1, 2 y 10 años · FRED o Tesoro de EE.UU.")
def s_ust():
    try:
        return pd.DataFrame({"ust1": fred("DGS1"), "ust2": fred("DGS2"), "ust10": fred("DGS10")})
    except Exception:  # noqa: BLE001  — FRED suele cortar conexiones desde servidores en la nube
        return tesoro_eeuu()


def yahoo(tk: str) -> pd.Series:
    import yfinance as yf
    h = yf.Ticker(tk).history(period="5y", auto_adjust=True)["Close"]
    h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
    return h[h > 0].loc[DESDE:]


@fuente("S&P 500 · Yahoo")
def s_spx():
    return yahoo("^GSPC")


@fuente("VIX · Yahoo")
def s_vix():
    return yahoo("^VIX")


@fuente("Brasil (EWZ) · Yahoo")
def s_ewz():
    return yahoo("EWZ")


@fuente("Precios en vivo de letras y bonos · data912")
def s_vivo():
    px = {}
    for ep in ("arg_notes", "arg_bonds"):
        for x in get_json(f"{D9}/live/{ep}"):
            if x.get("symbol") and (x.get("c") or 0) > 0:
                px[x["symbol"]] = float(x["c"])
    return px


@fuente("Pagos finales de LECAP · rendimientos-ar")
def s_config():
    return get_json(CFG_URL)


# ════════════════════════════════════════════════════════════════════
# Cálculos auxiliares
# ════════════════════════════════════════════════════════════════════
def habil_siguiente(d: pd.Timestamp) -> pd.Timestamp:
    d = d + pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    return d


MES_COD = {"E": 1, "F": 2, "M": 3, "A": 4, "Y": 5, "J": 6, "L": 7, "G": 8, "S": 9, "O": 10, "N": 11, "D": 12}


def venc_ticker(tk: str):
    import re
    m = re.match(r"^[A-Z](\d{2})([EFMAYJLGSOND])(\d)$", tk)
    if not m:
        return None
    try:
        return pd.Timestamp(2020 + int(m.group(3)), MES_COD[m.group(2)], int(m.group(1)))
    except ValueError:
        return None


def mercado(px: dict | None, cfg: dict | None, may: pd.Series | None, ust: pd.DataFrame | None) -> dict:
    """Curva de LECAP, dólar implícito (LELINK) y riesgo país antes/después de la elección (AO27/AO28)."""
    out: dict = {}
    if not px:
        return out
    liq = habil_siguiente(HOY)
    curva = []
    lecaps = (cfg or {}).get("lecaps", [])
    if isinstance(lecaps, dict):
        lecaps = lecaps.get("letras", [])
    import re
    for l in lecaps if isinstance(lecaps, list) else []:
        if not isinstance(l, dict) or l.get("activo") is False:
            continue
        tk, pf, fv = l.get("ticker"), l.get("pago_final"), l.get("fecha_vencimiento")
        # solo LECAP/BONCAP cero cupón (S30N6, T15E7…); deja afuera TO26, TY30P y similares
        if not (tk and pf and fv) or tk not in px or not re.match(r"^[ST]\d{2}[EFMAYJLGSOND]\d$", tk):
            continue
        d = (pd.Timestamp(fv) - liq).days
        if d < 5:
            continue
        r = float(pf) / px[tk] - 1
        if not (-0.05 < r < 1.5):
            continue
        curva.append({"tk": tk, "dias": d, "tem": (1 + r) ** (30 / d) - 1})
    curva.sort(key=lambda x: x["dias"])
    out["lecap_curva"] = curva

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
    out["lecap_tem_90d"] = tem_en(90)

    lelink = []
    for tk, p in px.items():
        v = venc_ticker(tk)
        if not tk.startswith("D") or v is None or p < 1000:
            continue
        d = (v - liq).days
        t = tem_en(d) if d >= 3 else None
        if t is None:
            continue
        lelink.append({"tk": tk, "dias": d, "fx": p / 100 * (1 + t) ** (d / 30)})
    lelink.sort(key=lambda x: x["dias"])
    out["lelink"] = lelink
    spot = float(may.iloc[-1]) if may is not None and len(may) else None
    out["mayorista"] = spot
    if spot and lelink:
        pts = [(0, spot)] + [(x["dias"], x["fx"]) for x in lelink]
        fx90 = None
        for (d0, f0), (d1, f1) in zip(pts, pts[1:]):
            if d0 <= 90 <= d1:
                fx90 = f0 + (f1 - f0) * (90 - d0) / (d1 - d0)
        out["fx_impl_90d"] = fx90
        out["deval_impl_90d"] = (fx90 / spot - 1) if fx90 else None

    # AO27 / AO28: TIR con cupón mensual (flujos de rendimientos-ar), forward entre ambos y spread contra Treasuries
    sob = (cfg or {}).get("soberanos", {})
    def flujos(tk):
        b = sob.get(tk) or {}
        return [(pd.Timestamp(f["fecha"] if isinstance(f, dict) else f[0]),
                 float(f["monto"] if isinstance(f, dict) else f[1])) for f in b.get("flujos", [])]
    def tir(precio, fl):
        cf = [((f - liq).days / 365, m) for f, m in fl if f > liq]
        if not cf or not precio:
            return None
        y = 0.1
        for _ in range(200):
            fv = -precio + sum(m / (1 + y) ** t for t, m in cf)
            df = sum(-t * m / (1 + y) ** (t + 1) for t, m in cf)
            ny = y - fv / df
            if abs(ny - y) < 1e-10:
                y = ny
                break
            y = max(ny, -0.5)
        dur = sum(t * m / (1 + y) ** t for t, m in cf) / precio
        return y, dur
    a = tir(px.get("AO27D"), flujos("AO27"))
    b = tir(px.get("AO28D"), flujos("AO28"))
    if a and b and b[1] > a[1]:
        fwd = ((1 + b[0]) ** b[1] / (1 + a[0]) ** a[1]) ** (1 / (b[1] - a[1])) - 1
        out.update({"ao27_tir": a[0], "ao28_tir": b[0], "fwd_post_eleccion": fwd})
        if ust is not None and len(ust.dropna()):
            u = ust.dropna().iloc[-1]
            ef = lambda y: (1 + y / 200) ** 2 - 1  # H.15: base semestral → efectivo anual
            us = lambda D: ef(u["ust1"]) + (ef(u["ust2"]) - ef(u["ust1"])) * min(max(D - 1, 0), 1)
            fu = ((1 + us(b[1])) ** b[1] / (1 + us(a[1])) ** a[1]) ** (1 / (b[1] - a[1])) - 1
            out["spread_fwd_pb"] = (fwd - fu) * 1e4
            out["spread_pre_pb"] = (a[0] - us(a[1])) * 1e4
    return out


def percentil(s: pd.Series, años: int = 3):
    s = s.dropna()
    if len(s) < 12:
        return None
    v = s.loc[s.index[-1] - pd.DateOffset(years=años):]
    x = v.iloc[-1]
    return float(((v < x).mean() + 0.5 * (v == x).mean()) * 100)


def cambio(s: pd.Series, dias: int, modo: str):
    s = s.dropna()
    if len(s) < 3:
        return None
    ref = s.loc[: s.index[-1] - pd.Timedelta(days=dias)]
    if not len(ref):
        return None
    a, b = float(ref.iloc[-1]), float(s.iloc[-1])
    if modo == "pct":
        return (b / a - 1) * 100 if a else None
    return b - a


def serie_corta(s: pd.Series, mensual: bool = False):
    s = s.dropna()
    if not len(s):
        return []
    s = s.loc[s.index[-1] - pd.DateOffset(years=3):]
    if not mensual:
        s = s.resample("W-FRI").last().dropna()
    return [[str(i.date()), round(float(v), 4)] for i, v in s.items()]


# Flujos de AL30/GD30 por 100 de valor nominal original (amortización 4% en jul-24 y 8% semestral desde ene-25,
# cupón 0,75% anual hasta jul-27). Se completan con los del config de rendimientos-ar cuando están.
FLUJOS_AL30 = {"2023-07-09": 0.25, "2024-01-09": 0.375, "2024-07-09": 4.375, "2025-01-09": 8.36, "2025-07-09": 8.33,
               "2026-01-09": 8.30, "2026-07-09": 8.27, "2027-01-09": 8.24, "2027-07-09": 8.21}


def flujos_bono(cfg, tk_base: str) -> dict:
    fl = dict(FLUJOS_AL30)
    b = ((cfg or {}).get("soberanos") or {}).get(tk_base) or {}
    for f in b.get("flujos", []) or []:
        try:
            fecha = str(f["fecha"] if isinstance(f, dict) else f[0])[:10]
            fl[fecha] = float(f["monto"] if isinstance(f, dict) else f[1])
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    return fl


def retorno_total(precio: pd.Series, flujos: dict) -> pd.Series:
    """Retorno diario con cupones y amortizaciones: el pago se suma el día del corte (la mayor caída cerca del pago)."""
    p = precio.dropna()
    r = p.pct_change()
    for f, monto in flujos.items():
        d = pd.Timestamp(f)
        ventana = r.loc[d - pd.Timedelta(days=10): d + pd.Timedelta(days=3)]
        if not len(ventana):
            continue
        dia = ventana.idxmin()
        prev = p.shift(1).loc[dia]
        if prev and prev > 0:
            r.loc[dia] = r.loc[dia] + monto / prev
    return r


def indice_tamar(tamar: pd.Series, cal: pd.DatetimeIndex) -> pd.Series:
    """Pesos a tasa: capitaliza la TAMAR día a día (proxy de plazo fijo / LECAP corta)."""
    t = tamar.reindex(cal.union(tamar.index)).sort_index().ffill().reindex(cal)
    dias = pd.Series(cal, index=cal).diff().dt.days.fillna(0)
    fac = 1 + t.shift(1).fillna(t.iloc[0]) / 100 * dias / 365
    return fac.cumprod()


def ols(y: pd.Series, x: pd.Series):
    j = pd.concat([y, x], axis=1).dropna()
    if len(j) < 20 or j.iloc[:, 1].var() == 0:
        return None
    yy, xx = j.iloc[:, 0], j.iloc[:, 1]
    beta = float(np.cov(yy, xx)[0, 1] / xx.var())
    r = float(yy.corr(xx))
    n = len(j)
    se = math.sqrt(max(1 - r * r, 1e-12) / (n - 2)) * yy.std() / xx.std()
    return {"beta": beta, "r2": r * r, "n": n, "t": beta / se if se else None}


# ════════════════════════════════════════════════════════════════════
# Principal
# ════════════════════════════════════════════════════════════════════
def main():
    rp, ccl, mep, ofi = s_riesgo(), s_ccl(), s_mep(), s_oficial()
    ipc, rem, icg = s_ipc(), s_rem(), s_icg()
    may, cer, tamar, reservas, compras, prest = s_mayorista(), s_cer(), s_tamar(), s_reservas(), s_compras(), s_prestamos()
    badlar = s_badlar()
    # Para medir sensibilidades hace falta historia larga: se usa la tasa con más datos (BADLAR o TAMAR)
    cand = [(n, x) for n, x in (("BADLAR", badlar), ("TAMAR", tamar)) if x is not None and len(x)]
    tasa_nom, tasa_hist = max(cand, key=lambda c: len(c[1])) if cand else (None, None)
    emae = s_emae()
    panel, ceds, bonos = s_panel(), s_cedears(), s_bonos_usd()
    ust, spx, vix, ewz = s_ust(), s_spx(), s_vix(), s_ewz()
    vivo, cfg = s_vivo(), s_config()

    try:
        mk = mercado(vivo, cfg, may, ust)
        FUENTES["Cálculos de mercado (LECAP, LELINK, AO27/AO28)"] = {"ok": bool(mk.get("lecap_curva")), "ultimo": str(HOY.date()),
                                                                      **({} if mk.get("lecap_curva") else {"error": "sin curva de LECAP"})}
    except Exception as e:  # noqa: BLE001
        mk = {}
        FUENTES["Cálculos de mercado (LECAP, LELINK, AO27/AO28)"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:180]}"}

    # ── Vector de estado ─────────────────────────────────────────────
    V = []
    def var(id_, nombre, grupo, s, unidad, fav, fuente_txt, dec=1, modo_cambio="abs", mensual=False, nota=None, valor=None):
        if s is None or not len(s.dropna()):
            V.append({"id": id_, "nombre": nombre, "grupo": grupo, "unidad": unidad, "fav": fav, "fuente": fuente_txt,
                      "valor": valor, "sin_historia": True, "nota": nota})
            return
        s = s.dropna()
        V.append({"id": id_, "nombre": nombre, "grupo": grupo, "unidad": unidad, "fav": fav, "fuente": fuente_txt,
                  "dec": dec, "valor": round(float(s.iloc[-1]), 4), "fecha": str(s.index[-1].date()),
                  "cambio_1s": None if mensual else cambio(s, 7, modo_cambio),
                  "cambio_1m": cambio(s, 30 if not mensual else 20, modo_cambio),  # mensual: contra el dato anterior
                  "modo_cambio": modo_cambio, "pctl_3a": percentil(s), "mensual": mensual,
                  "serie": serie_corta(s, mensual), "nota": nota})

    var("rp", "Riesgo país", "Argentina · riesgo y dólar", rp, "pb", "baja", "argentinadatos (JP Morgan EMBI)", 0, "abs")
    var("mayorista", "Dólar mayorista", "Argentina · riesgo y dólar", may, "$", None, "BCRA", 0, "pct",
        nota="La distancia al techo de la banda se calcula en el tablero con el techo que cargás vos.")
    if ccl is not None and ofi is not None:
        brecha = (ccl / ofi.reindex(ccl.index).ffill() - 1) * 100
        var("brecha", "Brecha CCL / oficial", "Argentina · riesgo y dólar", brecha, "%", "baja", "argentinadatos", 1, "abs")
    if compras is not None and may is not None:
        usd = compras / may.reindex(compras.index).ffill()
        var("compras", "Compras del BCRA (prom. 10 ruedas)", "Argentina · riesgo y dólar", usd.rolling(10).mean(),
            "US$M/día", "sube", "BCRA", 1, "abs")
    var("reservas", "Reservas brutas", "Argentina · riesgo y dólar", reservas, "US$M", "sube", "BCRA", 0, "abs")

    var("tamar", "TAMAR bancos privados", "Pesos · tasa e inflación", tamar, "% TNA", None, "BCRA", 2, "abs",
        nota="Ni buena ni mala: alta favorece al que está en pesos y castiga a las acciones.")
    var("badlar", "BADLAR bancos privados", "Pesos · tasa e inflación", badlar, "% TNA", None, "BCRA", 2, "abs")
    tem90 = mk.get("lecap_tem_90d")
    V.append({"id": "lecap90", "nombre": "LECAP a 90 días", "grupo": "Pesos · tasa e inflación", "unidad": "% mensual",
              "fav": None, "fuente": "data912 + rendimientos-ar", "dec": 2,
              "valor": round(tem90 * 100, 3) if tem90 is not None else None, "sin_historia": True,
              "nota": "La historia se arma con los días que corre este proceso."})
    var("ipc", "Inflación mensual (IPC)", "Pesos · tasa e inflación", ipc, "%", "baja", "INDEC vía argentinadatos", 1, "abs",
        mensual=True)
    rem_prox = None
    if rem and rem.get("ipc"):
        mes = HOY.replace(day=1).date().isoformat()
        mens = [x for x in rem["ipc"] if x.get("tipo") == "mensual" and x.get("desde") and x.get("mediana") is not None]
        cand = [x for x in mens if x["desde"] >= mes]
        rem_prox = (sorted(cand, key=lambda x: x["desde"])[0] if cand
                    else (sorted(mens, key=lambda x: x["desde"])[-1] if mens else None))
    if rem_prox:
        V[-1]["rem"] = rem_prox
    infl_ref, expost = None, False
    if rem_prox and rem_prox.get("mediana") is not None:
        infl_ref = float(rem_prox["mediana"])
    elif ipc is not None and len(ipc):
        infl_ref, expost = float(ipc.iloc[-1]), True      # sin REM: contra el último dato publicado
    if tem90 is not None and infl_ref is not None:
        V.append({"id": "tasa_real",
                  "nombre": "Tasa real (LECAP − último IPC)" if expost else "Tasa real esperada (LECAP − REM)",
                  "grupo": "Pesos · tasa e inflación", "unidad": "pp/mes", "fav": None, "fuente": "cálculo propio",
                  "dec": 2, "valor": round(tem90 * 100 - infl_ref, 3), "sin_historia": True, "expost": expost,
                  "nota": ("Sin REM disponible: se compara con la inflación pasada, no con la esperada. "
                           if expost else (f"REM {rem_prox.get('periodo')}"
                                           + (" (informe guardado). " if rem and rem.get("origen") == "guardado" else ". ")))
                          + "Positiva: los pesos le ganan a la inflación."})
    if mk.get("deval_impl_90d") is not None:
        V.append({"id": "deval90", "nombre": "Devaluación implícita a 90 días", "grupo": "Pesos · tasa e inflación",
                  "unidad": "%", "fav": "baja", "fuente": "LELINK vs. LECAP (data912)", "dec": 1,
                  "valor": round(mk["deval_impl_90d"] * 100, 2), "sin_historia": True,
                  "nota": f"Dólar implícito a 90 días: ${mk['fx_impl_90d']:,.0f}"})

    if emae is not None and len(emae) > 13:
        emae = emae.asfreq("MS")
        var("emae", "Actividad (EMAE, interanual)", "Actividad y política", (emae / emae.shift(12) - 1) * 100, "%",
            "sube", "INDEC vía datos.gob.ar", 1, "abs", mensual=True)
    if prest is not None and cer is not None:
        pm = prest.resample("ME").mean()
        if len(pm) and prest.index[-1] < pm.index[-1]:
            pm = pm.iloc[:-1]                             # el mes en curso todavía no terminó
        cm = cer.resample("ME").mean().reindex(pm.index)
        real = pm / cm
        var("credito", "Crédito real en pesos (interanual)", "Actividad y política", (real / real.shift(12) - 1) * 100,
            "%", "sube", "BCRA (deflactado por CER)", 1, "abs", mensual=True)
    var("icg", "Confianza en el Gobierno (UTDT)", "Actividad y política", icg, "0–5", "sube", "UTDT vía argentinadatos",
        2, "abs", mensual=True)

    if spx is not None:
        var("spx", "S&P 500 (variación a 1 mes)", "Afuera", (spx / spx.shift(21) - 1) * 100, "%", "sube", "Yahoo", 1, "abs")
    if ust is not None:
        var("ust10", "Tasa a 10 años de EE.UU.", "Afuera", ust["ust10"], "%", "baja", "FRED", 2, "abs")
    var("vix", "Volatilidad (VIX)", "Afuera", vix, "pts", "baja", "Yahoo", 1, "abs")
    if ewz is not None:
        var("ewz", "Brasil (EWZ, variación a 1 mes)", "Afuera", (ewz / ewz.shift(21) - 1) * 100, "%", "sube", "Yahoo", 1, "abs")

    # ── Risk-off score (misma definición que analisis_riskoff.ipynb) ──
    riskoff = None
    if rp is not None and ccl is not None and ofi is not None and panel is not None:
        bancos = [t for t in ("GGAL", "BMA", "BBAR") if t in panel.columns]
        if bancos:
            f = rp.index.intersection(ccl.index).intersection(ofi.index)
            for t in bancos:
                f = f.intersection(panel[t].dropna().index)
            f = f.sort_values()
            if len(f) > 250:
                br = ccl.loc[f] / ofi.loc[f] - 1
                b = pd.concat([panel[t].loc[f] / ccl.loc[f] for t in bancos], axis=1)
                b = (b / b.iloc[0]).mean(axis=1)
                W = 90
                z = lambda s: (s - s.rolling(W).mean()) / s.rolling(W).std()
                rz, bz, ez = z(rp.loc[f]), z(br), z(b.pct_change(20))
                sc = (rz + bz - ez).dropna()
                if len(sc):
                    riskoff = {"valor": round(float(sc.iloc[-1]), 3), "fecha": str(sc.index[-1].date()),
                               "pctl": float((sc < sc.iloc[-1]).mean() * 100),
                               "riesgo_z": round(float(rz.loc[sc.index[-1]]), 3),
                               "brecha_z": round(float(bz.loc[sc.index[-1]]), 3),
                               "bancos_z": round(float(ez.loc[sc.index[-1]]), 3),
                               "serie": serie_corta(sc)}
                    var("riskoff", "Risk-off score", "Argentina · riesgo y dólar", sc, "score", "baja",
                        "cálculo propio", 2, "abs", nota="Describe el estado; no anticipa (ya medido).")

    # ── Índices de las clases de activo (en pesos, diarios) ──────────
    cal = pd.bdate_range(HOY - pd.DateOffset(years=3), HOY)
    def al_cal(s):
        return None if s is None else s.reindex(cal.union(s.index)).sort_index().ffill().reindex(cal)
    clases_px: dict[str, pd.Series] = {}
    if panel is not None:
        r = panel.reindex(cal.union(panel.index)).sort_index().ffill().reindex(cal).pct_change(fill_method=None)
        med = r.median(axis=1)
        r = r.where(r.sub(med, axis=0).abs() < 0.25)     # descarta saltos propios de un papel (dividendo en acciones, error)
        clases_px["acc"] = (1 + r.mean(axis=1).fillna(0)).cumprod()
    if ceds is not None and set(CANASTA_CEDEAR) <= set(ceds.columns):
        r = ceds.reindex(cal.union(ceds.index)).sort_index().ffill().reindex(cal).pct_change(fill_method=None)
        r = r.where(r.abs() < 0.35)                      # cambio de ratio del CEDEAR, no un retorno
        clases_px["ced"] = (1 + sum(w * r[t].fillna(0) for t, w in CANASTA_CEDEAR.items())).cumprod()
    if tasa_hist is not None:
        clases_px["rfp"] = indice_tamar(tasa_hist, cal)
    if bonos is not None and mep is not None and len(bonos.columns):
        rts = []
        for tk in bonos.columns:
            rt = retorno_total(bonos[tk], flujos_bono(cfg, tk[:-1]))      # AL30D → flujos de AL30
            rts.append(rt.reindex(cal).where(lambda x: x.abs() < 0.3))
        rb = pd.concat(rts, axis=1).mean(axis=1).fillna(0)
        clases_px["rfu"] = (1 + rb).cumprod() * al_cal(mep)  # retorno total en USD × MEP = pesos
    if mep is not None:
        clases_px["usd"] = al_cal(mep)
    CLASES = [
        {"id": "acc", "nombre": "Acciones AR", "proxy": "Panel líder equiponderado (21 papeles, data912)"},
        {"id": "ced", "nombre": "CEDEARs", "proxy": "Canasta 70% SPY / 30% GLD (CEDEARs, data912)"},
        {"id": "rfp", "nombre": "Renta fija en pesos", "proxy": f"Pesos capitalizando a {tasa_nom or 'tasa'} (proxy de plazo fijo y LECAP corta)"},
        {"id": "rfu", "nombre": "Renta fija en dólares", "proxy": "Promedio AL30D y GD30D (data912)"},
        {"id": "usd", "nombre": "Dólar", "proxy": "Dólar MEP"},
    ]

    # ── Factores (cambios semanales) ─────────────────────────────────
    sem = lambda s: s.resample("W-FRI").last()
    FACT = []
    def factor(id_, nombre, s, tipo, sentido, desc, unidad):
        if s is None or len(s.dropna()) < 200:
            return
        w = sem(s.dropna())
        ch = np.log(w).diff() if tipo == "log" else w.diff()
        FACT.append({"id": id_, "nombre": nombre, "cambios": ch.dropna(), "tipo": tipo, "sentido": sentido,
                     "desc": desc, "unidad": unidad, "nivel": float(s.dropna().iloc[-1])})
    factor("rp", "Sube el riesgo país", rp, "log", 1, "riesgo país", "%")
    factor("fx", "Sube el dólar mayorista", may, "log", 1, "dólar mayorista", "%")
    factor("tasa", f"Sube la tasa en pesos ({tasa_nom})", tasa_hist, "abs", 1, tasa_nom or "tasa", "pp")
    factor("spx", "Cae el S&P 500", spx, "log", -1, "S&P 500", "%")
    if ust is not None:
        factor("ust10", "Sube la tasa a 10 años de EE.UU.", ust["ust10"], "abs", 1, "Treasury 10 años", "pp")
    factor("vix", "Sube la volatilidad global (VIX)", vix, "abs", 1, "VIX", "pts")
    factor("ewz", "Cae Brasil (EWZ)", ewz, "log", -1, "EWZ", "%")

    sensib = {"ventana_semanas": 104, "unidades": {}, "factores": []}
    for F in FACT:
        c = F["cambios"].iloc[-104:]
        m4 = c.rolling(4).sum().dropna()
        shock = float(m4.std()) * F["sentido"] if len(m4) > 10 else None
        if shock is None:
            continue
        F["shock"] = shock
        if F["tipo"] == "log":
            txt = f"{(math.exp(shock) - 1) * 100:+.1f}%"
            if F["id"] == "rp":
                txt += f" (≈ {F['nivel'] * (math.exp(shock) - 1):+.0f} pb)"
        else:
            txt = f"{shock:+.2f} {F['unidad']}" if F["unidad"] != "pts" else f"{shock:+.1f} pts"
        sensib["factores"].append({"id": F["id"], "nombre": F["nombre"], "shock_txt": txt,
                                   "shock": shock, "tipo": F["tipo"], "unidad": F["unidad"], "nivel": F["nivel"],
                                   "desc": f"Movimiento típico de un mes del {F['desc']} (1 desvío de 4 semanas, últimos 2 años)"})
    deflactores = {"usd": al_cal(ccl), "real": al_cal(cer)}
    for unidad, defl in deflactores.items():
        if defl is None:
            continue
        mat = {}
        for C in CLASES:
            px = clases_px.get(C["id"])
            if px is None:
                continue
            serie = (px / defl).dropna()
            y = np.log(sem(serie)).diff().dropna()
            fila = {}
            for F in FACT:
                if "shock" not in F:
                    continue
                y2, x2 = y.iloc[-104:], F["cambios"]
                o = ols(y2, x2)
                if not o:
                    continue
                o1, o2 = ols(y2.iloc[:52], x2), ols(y2.iloc[52:], x2)
                estable = bool(o1 and o2 and np.sign(o1["beta"]) == np.sign(o2["beta"])
                               and 0.33 <= abs(o1["beta"]) / max(abs(o2["beta"]), 1e-9) <= 3)
                fila[F["id"]] = {"beta": round(o["beta"], 4), "r2": round(o["r2"], 3), "n": o["n"],
                                 "t": round(o["t"], 2) if o["t"] is not None else None,
                                 "impacto": round((math.exp(o["beta"] * F["shock"]) - 1) * 100, 2),
                                 "estable": estable}
            # retorno y volatilidad de la clase en esta unidad (último año)
            r52 = y.iloc[-52:]
            fila["_stats"] = {"ret_1a": round((math.exp(r52.sum()) - 1) * 100, 1) if len(r52) else None,
                              "vol_anual": round(float(r52.std() * math.sqrt(52) * 100), 1) if len(r52) > 10 else None}
            mat[C["id"]] = fila
        sensib["unidades"][unidad] = mat

    # ── Salida ───────────────────────────────────────────────────────
    out = {
        "version": VERSION,
        "generado": AHORA.isoformat(timespec="minutes"),
        "fuentes": FUENTES,          # primero: es lo que se mira cuando algo falla
        "variables": V,
        "clases": CLASES,
        "sensibilidades": sensib,
        "riskoff": riskoff,
        "mercado": {k: v for k, v in mk.items()},
        "rem": rem,
    }
    def limpio(o):
        if isinstance(o, float):
            return None if (math.isnan(o) or math.isinf(o)) else round(o, 6)
        if isinstance(o, dict):
            return {k: limpio(v) for k, v in o.items()}
        if isinstance(o, list):
            return [limpio(v) for v in o]
        if isinstance(o, (np.floating,)):
            return limpio(float(o))
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return o
    (DATA / "contexto.json").write_text(json.dumps(limpio(out), ensure_ascii=False), encoding="utf-8")
    (DATA / "fuentes.json").write_text(json.dumps(limpio({"generado": out["generado"], "fuentes": FUENTES}),
                                                  ensure_ascii=False, indent=1), encoding="utf-8")

    # Historial: una fila por día (si se corre dos veces el mismo día, se reemplaza)
    fila = {"fecha": str(HOY.date()), "generado": out["generado"]}
    for v in V:
        if v.get("valor") is not None and not v.get("expost"):   # no mezclar definiciones en la historia
            fila[v["id"]] = v["valor"]
    for k in ("lecap_tem_90d", "fx_impl_90d", "ao27_tir", "ao28_tir", "fwd_post_eleccion", "spread_fwd_pb"):
        if mk.get(k) is not None:
            fila[k] = round(float(mk[k]), 6)
    hp = DATA / "historial.csv"
    h = pd.read_csv(hp) if hp.exists() else pd.DataFrame()
    h = pd.concat([h[h.get("fecha", pd.Series(dtype=str)) != fila["fecha"]] if len(h) else h, pd.DataFrame([fila])],
                  ignore_index=True)
    h.to_csv(hp, index=False)

    ok = sum(1 for f in FUENTES.values() if f.get("ok"))
    print(f"MKTBOARD · {out['generado']} · fuentes OK {ok}/{len(FUENTES)} · variables {len(V)} · factores {len(FACT)}")
    for n, f in FUENTES.items():
        print(f"  {'OK ' if f.get('ok') else 'ERR'} {n}: {f.get('ultimo') or f.get('error')}")
    if ok < len(FUENTES) / 2:
        raise SystemExit("Fallaron más de la mitad de las fuentes: revisá el log.")


if __name__ == "__main__":
    main()
