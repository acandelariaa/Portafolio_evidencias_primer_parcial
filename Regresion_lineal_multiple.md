# Regresión Lineal Múltiple
En este apartado vamos a explorar distintas variables que podrían ayudarnos a describir mejor el comportamiento de nuestros datos.

### Otras variables de interés
Como vimos en el paso anterior que la variable de GDP solo explica en un 22.2% la relación de la felicidad, una buena estrategia es buscar otras variables que nos
puedan ayudar a explicar de mejor manera el comportamiento de la felicidad a partir de factores sociales, económicos, etc.

Para Este estudio, se optó por buscar variables en distintas fuentes, lo más confiable parece ser utilizar variables dentro del mismo WHR, las cuales fueran no
tanto economicas si no un poco mas subjetivas y relacionadas con el ambito social.

Las variables propuestas fueron:

- **Social Support:** mide que tanto las personas sienten que tienen en alguien en quien confiar en momentos difíciles, la escala va del 0 al 1.
- **Health Life Expectancy:** mide que tantos años se esperan vivir con buena salud, se mide en años.
- **Freedom of making choices:** mide que tan satisfechas se sienten las personas se sienten con su libertad de tomar decisiones, la escala va del 0 al 1

De modo que estaremos trabajando con estas variables para ver si podemos mejorar un poco la relación del modelo con respecto a estas variables.

*Helliwell, J. F., Huang, H., Wang, S., & Norton, M. (2020). Statistical Appendix for Chapter 2 
(World Happiness Report 2020). Sustainable Development Solutions Network. Recuperado de https://files.worldhappiness.report/WHR20_Statistical_Appendix_01.pdf*

Ya habiendo comentado lo anterior, procedamos a trabajar con los datos.

### Cargar los datos nuevos
NOTA: en este caso se trabajó con dos archivos .csv, con la diferencia que estas nuevas variables se agregaran a la hoja principal, utilizando 
como una especie de "merge" el cual fusionara las hojas el archivo, haciéndolas coincidir por medio del nombre del país, en caso de no tener ese país en la lista, 
no se utilizará.

>Python Code

```python
# cargar nueva base de datos
df_nuevo = pd.read_csv('/content/drive/MyDrive/Inteligencia_Artificial_1/Felicidad_con_mas_variables.csv')

# ver las nuevas variables
df_nuevo
```

>Output

|Pais|	Social support|	Healthy life expectancy|	Freedom to make life choices|
|--|--|--|--|
|Finland|	0.954330	|71.900825|	0.949172|
|Switzerland|	0.942847|	74.102448|	0.921337|
|Iceland	|0.974670	|73.000000	|0.948892|
|Norway	|0.952487	|73.200783|	0.955750|
|...	|...	|...	|...	|...|
|Central |African Republic	|0.319460	|45.200001	|0.640881|
|Rwanda	|0.540835	|61.098846	|0.900589|
|Zimbabwe|	0.763093	|55.617260	|0.711458|
|South Sudan|	0.553707|	51.000000	|0.451314|
|Afghanistan|	0.470367	|52.590000	|0.396573|
|153 rows × 4 columns||||

Bien!, tenemos buena cantidad de variables, ahora vamos a fusionar estos datos con los anteriores para ver si podemos obtener mejores indicadores.

### Fusionar Datos

>Python Code

```python
# unir las nuevas variables conforme al pais
df_unido = pd.merge(df, df_nuevo, on='Pais', how='inner')
df_unido
```

>Output

| País        | Felicidad (cantril ladder) | GDP (Mill USD) | log_GDP   | Social support | Healthy life expectancy | Freedom to make life choices |
|------------|----------------------------|----------------|-----------|----------------|-------------------------|------------------------------|
| Finland    | 7.8210                     | 271837.000000  | 12.512958 | 0.954330       | 71.900825               | 0.949172                     |
| Denmark    | 7.6362                     | 356085.000000  | 12.782925 | 0.955991       | 72.402504               | 0.951444                     |
| Iceland    | 7.5575                     | 21718.075725   | 9.985900  | 0.974670       | 73.000000               | 0.948892                     |
| Switzerland| 7.5116                     | 752248.000000  | 13.530821 | 0.942847       | 74.102448               | 0.921337                     |
| Netherlands| 7.4149                     | 913865.000000  | 13.725438 | 0.939139       | 72.300919               | 0.908548                     |
| ...        | ...                        | ...            | ...       | ...            | ...                     | ...                          |
| Botswana   | 3.4711                     | 14930.072799   | 9.611133  | 0.779122       | 58.924454               | 0.821328                     |
| Rwanda     | 3.2682                     | 10184.345442   | 9.228607  | 0.540835       | 61.098846               | 0.900589                     |
| Zimbabwe   | 2.9951                     | 18051.170799   | 9.800966  | 0.763093       | 55.617260               | 0.711458                     |
| Lebanon    | 2.9553                     | 25948.915861   | 10.163885 | 0.824338       | 67.106583               | 0.551358                     |
| Afghanistan| 2.4038                     | 20116.137326   | 9.909278  | 0.470367       | 52.590000               | 0.396573                     |
|136 rows × 7 columns|||||||


Observación: nótese como se redujo el tamaño original de las observaciones, muy seguramente debido 
a que algunos países no tenían ciertos datos, de modo que sé descartaron automáticamente, más específicamente se quitaron 5 observaciones, 
de modo que obtenemos 136 observaciones y 6 variables para trabajar.

# Crear Modelo de regresión lineal multiple

Empezaremos generando datos de entrenamiento y prueba, con una proporcion del 80% de los datos originales.



>Python Code



```python
# Genera datos de entrenamiento
train = df_unido.sample(frac = 0.8)
# Genera datos de validación
test = df_unido.drop(train.index)
# Imprime dimensiones de datos de entrenamiento
print("Train:", train.shape)
# Imprime dimensiones de datos de prueba
print("Test:",test.shape)
# Imprime primeras 5 filas de datos de entrenamiento
print(train.head())
```

>Output


```text
Train: (109, 7)
Test: (27, 7)
           Pais  Felicidad (cantril ladder)  GDP(Mill USD)    log_GDP  \
124        Togo                      4.1123   7.574637e+03   8.932561   
24    Singapore                      6.4802   3.452960e+05  12.752157   
11    Australia                      7.1621   1.327840e+06  14.099064   
52     Honduras                      6.0221   2.382784e+04  10.078610   
87   Bangladesh                      5.1555   3.739020e+05  12.831749   

     Social support  Healthy life expectancy  Freedom to make life choices  
124        0.551313                54.719898                      0.649829  
24         0.910269                76.804581                      0.926645  
11         0.944855                73.604538                      0.915432  
52         0.821870                67.198769                      0.870603  
87         0.687293                64.503067                      0.900625  
```


Ahora vamos a realizar la regresion lineal multiple con nuestros datos ya divididos


>Python Code


```python
#importamos la libreria necesaria
import statsmodels.api as sm

# Definimos la variable dependiente (felicidad)
Y = train['Felicidad (cantril ladder)']

# Quitamos variables independientes (todas menos felicidad y país)
X = train.drop(['Felicidad (cantril ladder)', 'Pais'], axis=1)

# Agregar intercepto
X = sm.add_constant(X)

# Ajustar modelo
model = sm.OLS(Y, X)
results = model.fit()

# Resumen
print(results.summary())

```


>Output


```text
                                OLS Regression Results                                
======================================================================================
Dep. Variable:     Felicidad (cantril ladder)   R-squared:                       0.732
Model:                                    OLS   Adj. R-squared:                  0.719
Method:                         Least Squares   F-statistic:                     56.25
Date:                        Sun, 25 Jan 2026   Prob (F-statistic):           6.53e-28
Time:                                00:17:24   Log-Likelihood:                -96.588
No. Observations:                         109   AIC:                             205.2
Df Residuals:                             103   BIC:                             221.3
Df Model:                                   5                                         
Covariance Type:                    nonrobust                                         
================================================================================================
                                   coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------------------
const                           -3.8290      0.620     -6.173      0.000      -5.059      -2.599
GDP(Mill USD)                 3.142e-08   3.26e-08      0.964      0.337   -3.32e-08    9.61e-08
log_GDP                          0.0148      0.043      0.347      0.729      -0.070       0.099
Social support                   4.1673      0.774      5.382      0.000       2.632       5.703
Healthy life expectancy          0.0672      0.013      5.087      0.000       0.041       0.093
Freedom to make life choices     1.8107      0.561      3.229      0.002       0.698       2.923
==============================================================================
Omnibus:                       19.044   Durbin-Watson:                   2.227
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               29.829
Skew:                          -0.797   Prob(JB):                     3.33e-07
Kurtosis:                       5.007   Cond. No.                     2.91e+07
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 2.91e+07. This might indicate that there are
strong multicollinearity or other numerical problems.
```



OBSERVACIONES: nótese como con este modelo de regresión lineal múltiple, nuestra R^2, tuvo un cambio significativo, de 0.222 a 0.719, es decir, ahora con estas nuevas variables nuestro modelo puede predecir casi un 72% en el entrenamiento del conjunto de datos, eso es un incremento de casi 50 puntos porcentuales.

Así mismo nótese que al agregar nuevas variables, otras dejaron de tener tanta relevancia para el modelo, como por ejemplo:

- log_GDP, p|t|= 0.729
- GDP    , p|t|= 0.337

Sabiendo que la variable de log_GDP es la medida del GDP, pero en escala logarítmica para una mejor visualización de los datos, dejo de importar casi en un 72%

En cambio, la variable original de GDP, dejo de ser relevante en un 33%, es decir, si ayuda a describir el comportamiento, pero no tanto como las nuevas variables,
las cuales obtuvieron p values de < 0.002.

Esto es un muy buen avance, significa que nuestro modelo va por buen camino, pero, como es de esperarse, aun hay que validar estos datos para ver 
si realmente es tan bueno como parece.


Como ya vimos que algunas variables parecen no aportar demasiado, vamos a quitarlas y ver como se comporta el modelo mediante los indicadores explicados previamente al inicio 
, este caso, quitaremos log_GDP por su pvalue de 73 %

### Quitar variables con poca importancia


>Python Code


```python
# Genera el nuevo elemento X, sin la variable log_GDP
XNew = X.drop('log_GDP', axis = 1)
# Define el nuevo modelo sin esa variable
modelNew = sm.OLS(Y,sm.add_constant(XNew))
# Ajusta el nuevo modelo
resultsNew = modelNew.fit()

yhatNew = resultsNew.predict(sm.add_constant(XNew))
RSSNew = sum((Y-yhatNew)**2)
EMSNew = (RSSNew - RSS) / 1
FNew = EMSNew / RMS
pvalNew = st.f.sf(FNew, 1, n-m-1)
t = np.sqrt(FNew)
print("New F =", FNew)
print("t-value =", t)
print("p-value =", pvalNew)
print("OLS's p-value =", results.pvalues.log_GDP)
```


>Output



```text
New F = 0.12033581824152356
t-value = 0.3468945347530335
p-value = 0.7293782896073862
OLS's p-value = 0.7293782896073646
```


Observación
La prueba F= 0.12 y el p-value de 0.729 muestra que log_GDP no es una variable significativa dentro del modelo múltiple, por lo que su exclusión es estadísticamente justificable. 
Esto sugiere que, una vez considerados factores sociales y de bienestar, el ingreso económico (medido en logaritmo) deja de ser un predictor importante de la felicidad 
entre países.


Ahora, ya casi acercandonos al final de este estudio, calculemos el RSE y R^2 para los datos en validacion (test)

### Calcular indicadores RSE y R^2

>Python Code


```python
# Quitar 'const' de la lista de columnas
cols_modelo = results.model.exog_names.copy()
cols_modelo.remove('const')

# Usar esas columnas en test
XTest = test[cols_modelo]

# Agregar constante
XTest = sm.add_constant(XTest)

# Ahora sí predecir
yhatTest = results.predict(XTest)

# Errores
RSSTest = sum((YTest - yhatTest)**2)
TSSTest = sum((YTest - np.mean(YTest))**2)

nTest = XTest.shape[0]
mTest = XTest.shape[1]   # sin constante

# Métricas
RSETest = np.sqrt(RSSTest / (nTest - mTest - 1))
R2Test = 1 - RSSTest / TSSTest

print("RSE test =", RSETest)
print("R^2 test =", R2Test)
```


>Output



```text
RSE test = 0.6422473740999439
R^2 test = 0.6253273782192916
```


Ya en el test obtuvimos un R^2 de 0.625, un poco mas bajo que el training, pero dentro de lo que cabe, sigue siendo razonable, 
con esto podemos decir que nuestro nuevo modelo multivariado explica el 62.5% del conjunto de datos, es un valor razonable para 
cuestiones relacionadas a la felicidad, las cuales tienden a ser relativas y subjetivas.

# Conclusiones

Con este estudio encontramos factores que podrían determinar en su mayoría la felicidad, más allá de eso vimos que al usar una regresión lineal simple, 
podemos describir el comportamiento de un conjunto de datos; sin embargo, no es tan preciso y nunca lo será como usar una regresión lineal múltiple, con la 
cuál al tener más variables explicativas se puede entender y modelar mejor el comportamiento de un conjunto de datos.

Para este estudio en específico, al principio, natural e intuitivamente, pensamos que el GDP (producto interno bruto), podría determinar la felicidad de las personas
de diversos países debido a su factor económico, en el primer modelo de regresión lineal simple, exploramos que podría ser una posibilidad, sin embargo, al realizar la 
regresión lineal múltiple, agregando, más variables, nos dimos cuenta de que eso no necesariamente es verdad, al agregar variables como la esperanza de una vida saludable,
la percepción del apoyo social y la percepción de la libertad, nos dimos cuenta por medio de estadística, que esos factores pasan a un segundo plano cuando se trata de 
hablar de lo que define la verdadera felicidad.

De modo que se concluye que la inclusión de nuevas variables sí mejora la precisión del modelo, ya que al incorporar factores sociales y de bienestar se logra explicar
una mayor parte de la variabilidad en la felicidad. Sin embargo, también se observa que no todas las variables añadidas aportan información relevante, como fue el caso de
log_GDP, cuyo efecto no resultó estadísticamente significativo. Esto indica que un buen modelo no solo depende de agregar más variables, sino de incluir aquellas que realmente
contribuyen a explicar el fenómeno estudiado.

Como trabajo futuro, sería conveniente incluir más factores relevantes y explorar modelos más complejos que permitan capturar mejor 
la naturaleza diversa y cambiante del bienestar humano.


# Referencias APA
- *Data sharing | The World Happiness Report. (s. f.). https://www.worldhappiness.report/data-sharing/*
- *Frequently asked questions | The World Happiness Report. (s. f.-b). https://www.worldhappiness.report/faq/*
- *Home | The World Happiness Report. (2025, 20 agosto). https://www.worldhappiness.report/*
- *Macrotrends. (s. f.). GDP by country. Recuperado de https://www.macrotrends.net/global-metrics/countries/ranking/gdp-gross-domestic-product*
