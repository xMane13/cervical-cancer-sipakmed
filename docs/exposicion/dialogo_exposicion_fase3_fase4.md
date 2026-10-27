# Diálogo de exposición — Fase 3 y Fase 4
*(≈3 minutos. Los bloques de código son para mostrar en pantalla mientras se habla, no para leerlos en voz alta.)*

---

## Enlace con lo anterior (≈15 s)

Ya vimos cómo evaluar de forma honesta. Ahora la pregunta cambia: si además de la foto le damos al
modelo las mismas medidas que usa un citotecnólogo al microscopio, ¿acierta más? El dataset trae los
contornos del núcleo y de la célula dibujados a mano, así que podemos calcularlas y probarlo.

---

## Fase 3 — Le damos las medidas del especialista (≈1:05)

Sacamos 26 medidas por célula: qué tan grande es el núcleo, cuánto ocupa dentro de la célula, qué tan
redondo es, qué tan oscuro. Y lo importante aquí no es el experimento, es cómo se compara.

**[Mostrar: la red con y sin medidas]**
```python
def crear_red(n_extra=0):
    base = models.efficientnet_b0(weights=...IMAGENET1K_V1)
    base.classifier = nn.Identity()          # nos quedamos con los 1280 números
    cabeza = nn.Sequential(nn.Linear(1280 + n_extra, 256), nn.ReLU(), nn.Dropout(0.3),
                           nn.Linear(256, len(CLASES)))
```

El error típico en los trabajos publicados es entrenar una red con medidas y compararla contra otra
red distinta sin medidas: cambian dos cosas a la vez y no sabes cuál causó la diferencia. Nosotros
usamos la misma red, la misma cabeza y los mismos pliegues. Ese `n_extra` es el único interruptor:
con cero recibe solo la foto, con veintiséis recibe también las medidas.

**[Mostrar: resultados]**
```
Solo medidas, sin ver la foto : 0.8505
Solo la foto                  : 0.9731
Foto + medidas                : 0.9671     McNemar p = 0.0075
```

El resultado va en contra de lo que esperábamos. No es que las medidas no aporten: **empeoran**, y la
prueba estadística lo confirma.

Pero ojo: el modelo que usa solo las medidas, sin ver una sola foto, llega a 0.85. O sea que las
medidas no son basura, tienen señal real. Entonces, ¿por qué no sirven?

---

## Fase 4 — Buscando el motivo (≈1:15)

La hipótesis era que la red ya calcula esas medidas por su cuenta, así que dárselas es repetirle algo
que ya sabe. Para comprobarlo hicimos algo al revés de lo intuitivo.

Cuando la red procesa una foto, antes de decidir la clase produce 1280 números que la resumen. En vez
de usarlos para adivinar la clase, los usamos para adivinar las medidas.

**[Mostrar: el sondeo]**
```python
mod = RidgeCV(alphas=ALFAS).fit((EMBED[tr] - mu) / sd, obj[tr])
pred[te] = mod.predict((EMBED[te] - mu) / sd)
```

Una regresión lineal sencilla, a propósito: con algo más potente podría calcular la medida por su
cuenta y ya no sabríamos si estaba dentro de la red o la inventó el sondeo.

**[Mostrar: R² por medida]**
```
Color e intensidad          Forma del núcleo
  área citoplasma  0.83       redondez        0.19
  núcleo/célula    0.66       área nuclear    0.16
  oscuridad        0.64       alargamiento    0.045
```

Y la respuesta salió mejor que la hipótesis. Dentro de la red está el color y la intensidad; la forma
no está. La red mira el núcleo, pero lo mide por su color, no por su geometría. Y en la fase 3 le
dimos justamente la geometría que le falta. Por eso no le sirvió: **lo que le falta no es lo que
necesita.**

Además pusimos un control, la misma red sin entrenar en células. Recupera las medidas mejor que la
nuestra en las doce, sin excepción: entrenarla para esta tarea le empobrece la descripción de la
forma, porque aprende a quedarse solo con lo que le sirve para separar las cinco clases.

---

## Cierre (≈15 s)

Un resultado negativo se puede reportar como "no funcionó" y ya. Preferimos explicar por qué, y
encontramos algo más interesante que la hipótesis original. Y queda una pregunta clara para lo que
sigue: si no decide por la forma del núcleo, ¿por dónde está mirando? Gracias.
