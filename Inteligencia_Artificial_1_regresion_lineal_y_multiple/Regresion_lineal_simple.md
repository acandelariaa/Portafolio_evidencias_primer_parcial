
# Regresión Lineal Simple
En este apartado se realizara una regresion lineal con el fin de poder analizar y visualizar los datos, asi mismo como el verificar cosas que puedan surgir en el camino.

### Transformacion de datos
Con el fin de manejar mejor los datos, agregaremos las escalas para poder tenerlas en cuenta al momento de hacer calculos, ademas de eso debido a la notacion cientifica del GDP dividiremos los datos entre 1e^6 (1,000,000) para tenerlo en Millones de dolares y no tener tanta dispersión de datos.

> Python Code


```python
# Renombrar columnas
df = df.rename(columns={'Felicidad': 'Felicidad (cantril ladder)', 'GDP': 'GDP(Mill USD)'})
df.head()

# Convertir a Millones de dolares para mejor interpretacion
df['GDP(Mill USD)'] = df['GDP(Mill USD)']/1000000
df.head()

# ver tamaño del dataset
df
```


>Output


|Pais	|Felicidad (cantril ladder)	|GDP(Mill USD)|
|--|--|--|
|Finland	|7.8210|	271837.000000|
|Denmark	|7.6362	|356085.000000|
|Iceland	|7.5575	|21718.075725|
|Switzerland	|7.5116	|752248.000000|
|Netherlands	|7.4149	|913865.000000|
|...	|...	|...	|...|
|Botswana	|3.4711	|14930.072799|
|Rwanda	|3.2682	|0184.345442|
|Zimbabwe	|2.9951	|18051.170799|
|Lebanon	|2.9553	|25948.915861|
|Afghanistan	|2.4038	|20116.137326|
|141 rows × 3 columns|||

Ahora podemos ver que el dataset esta conformado por 141 observaciones y 2 variables de interes Felicidad (en la escala de cantril) y GDP (en millones de dolares)

#### Modelo de regresion Lineal

Con los datos que tenemos, vamos a utilizar una regresion lineal simple para ver si podemos entender el comportamiento de los datos.

> Python Code


```python
# Definir variables
X = df["GDP(Mill USD)"]
Y = df["Felicidad (cantril ladder)"]

# Agregar intercepto
X = sm.add_constant(X)

# Ajustar modelo
modelo = sm.OLS(Y, X).fit()

print(modelo.summary())

# Graficar datos
plt.figure(figsize=(8,5))
plt.scatter(df["GDP(Mill USD)"], df["Felicidad (cantril ladder)"], label="Datos reales")
plt.plot(df["GDP(Mill USD)"], modelo.predict(X), color="red", label="Recta ajustada")
plt.xlabel("PIB (millones de USD)")
plt.ylabel("Felicidad (Ladder score)")
plt.title("Regresión lineal simple: Felicidad vs PIB")
plt.legend()
plt.grid(True)
plt.show()
```

>Output


```text
                                OLS Regression Results                                
======================================================================================
Dep. Variable:     Felicidad (cantril ladder)   R-squared:                       0.030
Model:                                    OLS   Adj. R-squared:                  0.023
Method:                         Least Squares   F-statistic:                     4.246
Date:                        Sat, 24 Jan 2026   Prob (F-statistic):             0.0412
Time:                                22:19:08   Log-Likelihood:                -210.63
No. Observations:                         141   AIC:                             425.3
Df Residuals:                             139   BIC:                             431.2
Df Model:                                   1                                         
Covariance Type:                    nonrobust                                         
=================================================================================
                    coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------
const             5.5099      0.095     58.247      0.000       5.323       5.697
GDP(Mill USD)  8.509e-08   4.13e-08      2.060      0.041    3.44e-09    1.67e-07
==============================================================================
Omnibus:                        1.731   Durbin-Watson:                   0.062
Prob(Omnibus):                  0.421   Jarque-Bera (JB):                1.798
Skew:                          -0.251   Prob(JB):                        0.407
Kurtosis:                       2.766   Cond. No.                     2.37e+06
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 2.37e+06. This might indicate that there are
strong multicollinearity or other numerical problems.
```

![Modeo sesgado](Felicidad_vs_pib.png)

Notese como el indicador R^2 sobre que tanto podemos explicar con este modelo, es de aproximadamente 3%, por el contrario significa que no podemos explicar el 97% de los datos que estan ahi con ese modelo. Al parecer el convertir los datos a millones de dolares no fue suficiente para poder visualizar el comportamiento de los datos, de modo que tenemos que probablemente tengamos que hacer otra transformacion de datos para poder ajustar la escala, probemos una escala logaritmica solamente aplicada a la variable de GPD.

### Transformar GDP a escala logaritmica


>Python Code


```python
df["log_GDP"] = np.log(df["GDP(Mill USD)"])

X = sm.add_constant(df["log_GDP"])
Y = df["Felicidad (cantril ladder)"]

modelo_log = sm.OLS(Y, X).fit()
print(modelo_log.summary())

# graficar datos
plt.figure(figsize=(8,5))
plt.scatter(df["log_GDP"], df["Felicidad (cantril ladder)"], label="Datos reales")
plt.plot(df["log_GDP"], modelo_log.predict(X), color="red", label="Recta ajustada")
plt.xlabel("log(PIB en millones USD)")
plt.ylabel("Felicidad")
plt.title("Regresión lineal simple: Felicidad vs log(PIB)")
plt.legend()
plt.grid(True)
plt.show()
```


>Output


```Text
                                OLS Regression Results                                
======================================================================================
Dep. Variable:     Felicidad (cantril ladder)   R-squared:                       0.222
Model:                                    OLS   Adj. R-squared:                  0.216
Method:                         Least Squares   F-statistic:                     39.59
Date:                        Sat, 24 Jan 2026   Prob (F-statistic):           3.83e-09
Time:                                22:06:04   Log-Likelihood:                -195.09
No. Observations:                         141   AIC:                             394.2
Df Residuals:                             139   BIC:                             400.1
Df Model:                                   1                                         
Covariance Type:                    nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          2.4664      0.498      4.948      0.000       1.481       3.452
log_GDP        0.2728      0.043      6.292      0.000       0.187       0.359
==============================================================================
Omnibus:                        2.648   Durbin-Watson:                   0.455
Prob(Omnibus):                  0.266   Jarque-Bera (JB):                2.523
Skew:                          -0.326   Prob(JB):                        0.283
Kurtosis:                       2.944   Cond. No.                         70.5
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```


![Datos logaritmicos](Felicidad_vs_pib_log.png)

¡Listo!, ahora podemos ver claramente una tendencia lineal en el conjunto de datos, nótese también como el indicador de R^2 subió a un 22.2%, lo cual sigue siendo bajo; sin embargo, ahora el modelo explica un poco más nuestro conjunto de datos.

Ya habiendo obtenido lo necesario, podemos ver los coeficientes $\beta_0$ y $\beta_1$, siendo de [ 2.466 , 0.272 ] respectivamente, de modo que nuestro modelo de regresión lineal simple se vería de la siguiente forma, hablando matemáticamente.

Y = 4.266 + 0.272*log(x), donde por cada unidad de x log(GDP), se aumentan 0.272 unidades en Y, siendo Y la felicidad, eso tomando en que tenemos una constante independiente de 4.266.

Este modelo funciona, pero, sin embargo, no nos dice mucho, ya que aún hay casi un 77.8% que no podemos explicar con él. Por otro lado, para el valor logarítmico de GDP, obtuvimos un p-value de prácticamente 0, de modo que nos indica que es un factor que probablemente puede estar muy relacionado con la felicidad.

De momento el GDP (producto interno bruto) parece estar fuertemente relacionado con la felicidad, sugiriendo que puede ser una variable de interés para este estudio, sin embargo, seguimos sin tener los datos suficientes para abarcar este tema de mejor manera.

En el siguiente módulo, exploraremos otras variables que nos podrían ayudar a describir mejor el comportamiento de los datos.
