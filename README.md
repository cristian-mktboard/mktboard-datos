# MKTBOARD.AR · datos diarios

Este repositorio corre **solo**, de lunes a viernes a las 19:00 de Buenos Aires: baja datos públicos del mercado
argentino y global, calcula el contexto y cómo pega en cada clase de activo, y guarda el resultado en
`data/contexto.json`. El tablero lee ese archivo.

**Qué hay acá:** solo datos de mercado y cálculos. 

## Qué esperar

- **Horario:** GitHub programa en hora UTC: `0 22 * * 1-5` = 19:00 de Buenos Aires, lunes a viernes. En momentos de
  mucha carga puede arrancar más tarde.
- **Si falla:** GitHub te manda un mail. El tablero sigue funcionando con los datos del último día bueno y te muestra
  la fecha.
- **Pausa automática:** en repositorios públicos, GitHub pausa la programación si pasan 60 días sin actividad. El
  guardado diario de datos cuenta como actividad; si igual se pausa, se reactiva desde **Actions** con un clic.
- **Costo:** cero. GitHub Actions es gratis en repositorios públicos.

## Qué calcula (fase 1)

| Bloque | Contenido |
|---|---|
| Vector de estado | ~19 variables (riesgo país, dólar, brecha, compras y reservas del BCRA, tasas, inflación y REM, LECAP, devaluación implícita, actividad, crédito, confianza, S&P 500, tasa de EE.UU., VIX, Brasil, risk-off) con nivel, cambio semanal y mensual, percentil de 3 años y serie corta |
| Sensibilidades | Cuánto se mueve cada clase (acciones AR, CEDEARs, renta fija en pesos, renta fija en dólares, dólar) ante el movimiento típico de un mes de cada variable. Regresión semanal, 2 años, en pesos reales y en dólares, con R², t y control de estabilidad |
| Mercado | Curva de LECAP, dólar implícito por LELINK, AO27/AO28 y riesgo país después de la elección |
| Historial | Una fila por día en `data/historial.csv`: la historia propia del sistema |

## Fuentes
argentinadatos.com · API del BCRA v4.0 · data912.com · datos.gob.ar (INDEC) · FRED (Reserva Federal de St. Louis) ·
Yahoo Finance (yfinance) · rendimientos-ar (pagos finales de LECAP y flujos de bonos).
