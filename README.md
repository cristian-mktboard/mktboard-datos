# MKTBOARD.AR · datos diarios

Este repositorio corre **solo**, de lunes a viernes a las 19:00 de Buenos Aires: baja datos públicos del mercado
argentino y global, calcula el contexto y cómo pega en cada clase de activo, y guarda el resultado en
`data/contexto.json`. El tablero (`mktboard-sistema.html`, en tu PC) lee ese archivo.

**Qué hay acá:** solo datos de mercado y cálculos. Tu cartera, tu bitácora y tus probabilidades quedan en tu
navegador y nunca se suben.

---

## Configuración desde cero (30–45 minutos, una sola vez)

### 1 · Crear la cuenta
1. Entrá a <https://github.com> → **Sign up**.
2. Mail, contraseña y un nombre de usuario (por ejemplo `cristian-mktboard`). Ese nombre va a aparecer en la
   dirección de tus datos.
3. Confirmá el mail que te llega.

### 2 · Crear el repositorio (la "carpeta")
1. Arriba a la derecha, el botón **+** → **New repository**.
2. **Repository name:** `mktboard-datos`.
3. Elegí **Public** (tiene que ser público para que el tablero pueda leer los datos sin una clave).
4. Tildá **Add a README file**.
5. **Create repository**.

### 3 · Subir el programa
1. Descomprimí el zip que te pasé. Adentro hay una carpeta `pipeline`, el archivo `requirements.txt` y este `README.md`.
2. En tu repositorio: **Add file** → **Upload files**.
3. Arrastrá la carpeta `pipeline`, el archivo `requirements.txt` y este `README.md` (reemplaza al que creó GitHub).
4. Abajo, **Commit changes**.

### 4 · Crear la programación diaria
GitHub esconde las carpetas que empiezan con punto, así que este archivo se crea a mano:
1. **Add file** → **Create new file**.
2. En el nombre escribí exactamente: `.github/workflows/diario.yml` (al escribir las barras se crean las carpetas).
3. Pegá el contenido del archivo `diario.yml` que está en el zip (carpeta `.github/workflows`).
4. **Commit changes**.

### 5 · Darle permiso para guardar los datos
1. En el repositorio: **Settings** → **Actions** → **General**.
2. Abajo de todo, **Workflow permissions** → **Read and write permissions** → **Save**.

### 6 · Correrlo la primera vez
1. Pestaña **Actions**. Si aparece un aviso para habilitar los workflows, aceptalo.
2. A la izquierda, **mktboard-diario** → botón **Run workflow** → **Run workflow**.
3. Tarda 3 a 6 minutos. **Verde** = salió bien. **Rojo** = entrá, abrí el paso "Correr el proceso", copiá el
   texto y pasámelo.
4. Volvé a la pestaña **Code**: tiene que aparecer la carpeta `data` con `contexto.json` e `historial.csv`.

### 7 · Conectar el tablero
1. Abrí `mktboard-sistema.html` en tu PC.
2. Pestaña **Datos** → escribí tu usuario y `mktboard-datos` → **Guardar y probar**.

---

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
