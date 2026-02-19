# Regresion lineal Simple y Multiple
En este módulo se explora el uso de herramientas de análisis de datos para modelar relaciones entre variables mediante regresión lineal simple y regresión lineal múltiple.

Se aplicarán conceptos de estadística inferencial y aprendizaje automático supervisado, haciendo énfasis en la interpretación de métricas como:

- Coeficientes de regresión

- Coeficiente de determinación (R²)

- Error estándar residual (RSE)

- Pruebas de significancia (p-values y F-statistic)
 
### Recursos

| Dataset | [Felicidad y GDP.csv](A1_2_Felicidad_y_GDP.csv) |
|---|---|
| **Variables Adicionales** | [Felicidad_con_mas_variables.csv](Felicidad_con_mas_variables.csv) |
| **Notebook** | [.ipynb](Tarea_2_IA_Regresion_lineal_y_multiple.ipynb) |


Todo esto sera con el fin de entender el comportamiento del conjunto de datos.

## *La Felicidad con relacion al GDP (Producto Interno Bruto) y otras variables*
Cuando Pensamos en la felicidad, probablemente se nos viene a la mente las cosas que nos producen felicidad o esa sensación de alegria, 
pero, naturalmente tambien es inevitable pensar si hay otras cosas involucradas a la feliciad que podrian influir en si alguien es feliz o no, 
como por ejemplo el factor socio-economico, siendo de tal manera que el tener una buena economia personal y estable nos permite tener libertad financiera, 
posiblemente una mejor calidad de vida y por consiguiente, la felicidad. ¿Pero, es esto acertado?; pues en este estudio abordaremos brevemente ese tema.

## Introducción:

Diversos estudios han abarcado el tema de la felicidad de manera estadística, haciendo preguntas y recopilando información de varios países con el objetivo
de cuantificar la calidad de vida de las personas a partir de ciertos factores económicos, sociales, etc. Una de las organizaciones con más información al respecto 
es la World Happiness Report (WHR), la cual ha recopilado información de más de 100 países alrededor del mundo, con el propósito de analizar qué variables influyen de
manera más significativa en el bienestar de la población.

Estos análisis consideran no solo el ingreso económico, sino también factores como el apoyo social, la esperanza de vida, la libertad de decisión y 
la percepción de corrupción. A partir de estos datos, es posible identificar patrones y relaciones que ayudan a entender las diferencias en los niveles de felicidad 
entre países.

Este estudio, aunque un poco más limitado en recursos, tiene el mismo objetivo, ver si existe alguna relación entre la felicidad y algunos factores socioeconómicos utilizando
conceptos de machine learning, data train y test, además de utilizar métodos matemáticos como la regresión lineal simple y múltiple.

## Metodologia

- *Exploración de Datos:* Se explorará la forma, dimensiones y tipos de variables presentes en los datasets utilizados en este estudio. Asimismo, en caso de ser necesario, se transformarán los datos para facilitar una mejor manipulación y análisis de los mismos.

- *Regresión lineal simple:* Con ayuda de Python se realizará una regresión lineal simple para analizar si existe alguna relación entre la variable de salida (Felicidad) y la variable de interés (GDP).

- *Visualización e interpretación de datos:* Con el fin de entender mejor el modelo de regresión lineal, se crearán gráficas que permitan observar
 de manera más clara la relación entre las variables y detectar posibles patrones o tendencias.

- *Propuesta de nuevas variables:* Se consultarán fuentes confiables y públicas para incorporar tres variables adicionales que puedan ayudar a
  describir de mejor manera el comportamiento del dataset.

- *Regresión lineal múltiple:* Se realizará una regresión lineal múltiple utilizando tanto las variables iniciales como las nuevas variables propuestas, con el objetivo de evaluar si el modelo mejora en la explicación de los datos. Para ello, se dividirán los datos en conjuntos de entrenamiento (*train*) y prueba (*test*) en aproximadamente in 80% de los datos originales.

- *Indicadores:* Con el fin de evaluar el desempeño de los modelos, se utilizarán métricas como el *F-statistic*, *p-value*, \(R^2\), *RSE*, entre otros,
   que permitirán medir qué tan bien el modelo explica la relación entre las variables.

- *Conclusiones:* Finalmente, se llegará a una conclusión con base en el análisis de los datos y los resultados obtenidos en los pasos previamente descritos.

## Conceptos Clave:
- **p-value (valor p):** es un valor que nos indica que tan probable es que la relación sea casualidad, si es muy pequeño, hay relación significativa entre las variables,
 si es muy alto significa que probablemente no hay relación significativa.
- **F-statistic:** es una forma de medir si las variables que tenemos describen o no el conjunto de datos con ellas, si es muy grande, el modelo indica que las variables lo describen bien, si es pequeño las variables no lo describen bien, es una forma de medir el modelo de forma global.
- **t-statistic:** nos ayuda a ver si una variable en específico que nos sirva o nos estorbe, si es muy pequeño la variable no influye mucho, si es muy grande la variable tiene un impacto significativo en el modelo.
- **RSE:** mide que tanto nos equivocamos en promedio, tiene las unidades de las variables de interés.
- **RSS:** es la cantidad total de error que comete el modelo.
- **R^2:** es un indicador que nos ayuda a saber que tanto de los datos podemos explicar con el modelo.

### Regresión lineal Simple
La regresión lineal simple es una técnica que se usa para describir la relación entre dos variables:

- Una variable que queremos explicar o predecir (variable dependiente)

- Una variable que creemos que influye en ella (variable independiente)

**Forma matematica:** Y= β₀ + β₁x + ε

Donde:

Y → Variable que queremos explicar (por ejemplo, felicidad)

X → Variable que usamos para explicarla (por ejemplo, PIB)

β₀ (beta cero) → Intercepto, es el valor de Y cuando X = 0

β₁ (beta uno) → Pendiente, indica cuánto cambia Y cuando X aumenta en una unidad

ε (epsilon) → Error, representa todo lo que el modelo no logra explicar

Basicamente es que tanto cambia Y significa a partir de la cantidad de x multiplicada por β₀, sumandole β₁

La idea es ajustar una línea recta que se acerque lo más posible a los datos.

### Regresión Lineal Multiple

La regresión lineal múltiple es una extensión de la regresión simple, pero ahora usamos varias variables para explicar una sola.

Esto es útil cuando un fenómeno depende de muchos factores al mismo tiempo (como la felicidad, que depende de salud, dinero, apoyo social, etc.).

**Forma matematica:**
Y = β₀ + β₁x1 + β₂x2 + β₃x3 + ... + βₖxk + ε

Donde:

Y → Variable que queremos explicar

X₁, X₂, X₃, …, Xₖ → Variables explicativas

β₀ → Intercepto

β₁, β₂, β₃, …, βₖ → Efecto individual de cada variable sobre Y

ε → Error del modelo

Es que tanto cambia Y tomando en cuentan multiples variables a la vez

NOTA: Si es cierto que estos indicadores pueden parecerse y confundirse, pero no hay porque alarmarse, solo son indicadores que nos ayudan a comprender como le va a nuestro modelo al trabajar con estos datos.

### Train - Test data

En el análisis de datos, el conjunto se divide comúnmente en datos de entrenamiento (train) y datos de prueba (test). Los datos de entrenamiento se utilizan para que el modelo aprenda las relaciones entre las variables y ajuste sus parámetros, mientras que los datos de prueba se reservan para evaluar qué tan bien funciona el modelo con información nueva que no ha visto antes. Esta división es importante porque permite comprobar si el modelo realmente entendió los patrones de los datos o si solo memorizó la información inicial, ayudando así a medir su capacidad de generalización.

# Procedimiento

[Exploracion de Datos](Exploracion_datos.md)

[Regresión Lineal Simple](Regresion_lineal_simple.md)

[Regresion Lineal Multiple / Conclusion](Regresion_lineal_multiple.md)

