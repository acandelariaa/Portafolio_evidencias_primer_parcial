# Exploracion y contexto de los datos
En este apartado se explorara el dataset asi como un poco del contexto de las variables que estan contenidas.


>Disclaimer: el lenguaje de programacion utilizado fue python, utilizando google colaboratory como herramienta.

### Carga de datos
Vamos a conectar los datos desde google drive
> Python Code

```python
# Conectar el drive e incluir librerias necesarias
from google.colab import drive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import drive
drive.mount('/content/drive')

# importar el dataset
df = pd.read_csv('/content/drive/MyDrive/Inteligencia_Artificial_1/A1.2 Felicidad y GDP.csv')

# Ver dataset
df.head(5)
```

> Output

|Pais|	Felicidad	|GDP|
|--|--|--|
|Finland|	7.8210|	2.718370e+11|
|Denmark	|7.6362	|3.560850e+11|
|Iceland	|7.5575	|2.171808e+10|
|Switzerland	|7.5116	|7.522480e+11|
|Netherlands	|7.4149	|9.138650e+11|

Nota: Indagando en internet, más específicamente en la página oficial de WHR, y en páginas con indicadores financieros como Macrotrends, notamos que las unidades para GPD con (USD) para cada pais en la lista y la felicidad fue determinada con una escala llamada *cantril ladder*, la cual básicamente es un número del 1 al 10 en donde las personas se imaginan en donde estan en la vida.

> "Our happiness ranking is based on a single life evaluation question called the Cantril Ladder:
Please imagine a ladder with steps numbered from 0 at the bottom to 10 at the top. The top of the ladder represents the best possible life for you and the bottom of the ladder represents the worst possible life for you. On which step of the ladder would you say you personally feel you stand at this time?"

*Frequently asked questions | The World Happiness Report. (s. f.-b).*

*Macrotrends. (s. f.). GDP by country.*
