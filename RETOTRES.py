import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from random import randint, choice
from sklearn.preprocessing import StandardScaler
import random
from sklearn.model_selection import cross_val_score, train_test_split

#Under y overfitting

# 1: ¿Cuales son las principales complicaciones de este planteamiento?
# Escribe tu respuesta a la pregunta 1 en esta celda de código:
'''
El conjunto de entrenamiento es demasiado pequeño y de poca complejidad.
'''


#Generemos 30 datos a partir de la función planteada en la ecuación 1
#Adicionalmente para complicarle un poco el trabajo al modelo agreguemos un poco de ruido
np.random.seed(0)
n_samples = 30

def true_fun(X):
    return np.cos(1.5 * np.pi * X)

X = np.sort(np.random.rand(n_samples))
y_sin_ruido = true_fun(X)
y = true_fun(X) + np.random.randn(n_samples) * 0.1

plt.plot(X, label='X')
plt.plot(y_sin_ruido, label='y (muestra de función original)')
plt.plot(y, label='y (muestra con ruido)')
plt.legend();
plt.show()


#Pregunta 2: ¿Cuántos grados son necesarios?
'''
Tres grados
'''

#A continuación te mostramos una manera de usar pipelines para explorar hiperparámetros.
#Los pipelines son muy útiles al momento de explorar hiperparámetros (en este caso el hiperparámetro que estamos explorando es el máximo grado de libertad en la transformación de la variable X necesario para estimar y)

degrees = [1, 4, 15]

plt.figure(figsize=(14, 5))
for i in range(len(degrees)):
    ax = plt.subplot(1, len(degrees), i + 1)
    plt.setp(ax, xticks=(), yticks=())

    polynomial_features = PolynomialFeatures(degree=degrees[i], include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),("linear_regression", linear_regression),])
    pipeline.fit(X[:, np.newaxis], y)

    X_test = np.linspace(0, 1, 100)
    plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Modelo creado")
    plt.plot(X_test, true_fun(X_test), label="Función original")
    plt.scatter(X, y, edgecolor="b", s=20, label="Muestra con ruido")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((0, 1))
    plt.ylim((-2, 2))
    plt.legend(loc="best")
    plt.title("Modelo usando desde X^1 hasta X^{}".format(degrees[i]))
plt.show()

#Pregunta 3:
#¿Como podemos modificar el código anterior para incluir el cálculo del error y
#sistematizar la selección del grado máximo polinomial en la transformación de X?

# Escribe tu respuesta a la pregunta 3 en esta celda de código:
'''
Ajustar los hiperparametros para mejorar la presicion
'''

#Pregunta 4: ¿Cuales son otros hiperparámetros que incrementan la complejidad de los modelos? en el caso de:
# Escribe tu respuesta a la pregunta 4 en esta celda de código:
'''
  Regresión logística:fit_intercept ,penalty
  Arbol de decision:     max_depth,min_samples_split,min_samples_leaf,max_leaf_nodes.
  K-medias: n_clusters, init,max_iter,tol,precomputed_distances, algorithm
  Redes neuronales:Choice,range,list,batch_size,number_of_hidden_layers,QUniform(min_value, max_value, q),
  QLogUniform(min_value, max_value, q),QNormal(mu, sigma, q),QLogNormal(mu, sigma, q),Uniform(min_value, max_value)
  LogUniform(min_value, max_value),Normal(mu, sigma),LogNormal(mu, sigma)
'''

####Reto sobre end-to-end machine learning model####


#2.Importar la informacion en un dataframe
df = pd.read_csv("insurance.csv")
print(df.head())

#3.seleccionar features y crea una regresión lineal,
# luego calcula el score para conocer el performance del modelo


df = df.drop(["region"], axis=1) #excluyendo region

#converting strings to bool
df['sex'] = df['sex'].map({'sex': {'female': True, 'male': False}})
df['smoker'] = df['smoker'].map({'smoker': {'yes': True, 'no': False}})


#llenando nan
df["sex"] = df["sex"].fillna(randint(0, 1))
df["smoker"] = df["smoker"].fillna(randint(0, 1))

reg = LinearRegression()
y = np.array(df['charges']).reshape(-1, 1)
x = np.array(df.drop('charges', axis=1))

reg.fit(x,y)

score_1 = reg.score(x, y)
print('score_1:',score_1)
#score 0.12009819576246927


#4.limpieza, imputación de valores en los vacíos, regresion lineal 2
# Escribe aquí tu código
#filling nan
df.isna().any()
#no hay

#5Crea nuevas variables y transforma las ya existentes si es necesario
#selecciona las variables mas reelevantes con alguna técnica de selección de variables.
#Luego crea nuevamente un modelo de regresión lineal calculando su score.
#Hay muy pocas variables

#6 Escalamiento de variables

y = np.array(df['charges']).reshape(-1, 1)
x = np.array(df.drop('charges', axis=1))

x_std = StandardScaler().fit_transform(x)
y_std = StandardScaler().fit_transform(y)

reg.fit(x_std, y_std)

score_4 = reg.score(x_std, y_std)
print('score_4:',score_4)
#se conserva la score
#score 0.12009819576246927


#7,8 PCA
def get_pca_components(pca, var):
    cumm_var = pca.explained_variance_ratio_
    total_var = 0.
    N_COMPONENTS = 0
    for i in cumm_var:
        N_COMPONENTS += 1
        total_var += i
        if total_var >= var:
            break
    return N_COMPONENTS


pca = PCA().fit(x_std)
n_components = get_pca_components(pca, 0.75)

print(n_components)
# 3 componentes principales (age,bmi,charges)

#7,8 Reduccion de dimencionalidad y regresion
df = df.drop(["sex"], axis=1)
df = df.drop(["smoker"], axis=1)

y = np.array(df['charges']).reshape(-1, 1)
x = np.array(df.drop('charges', axis=1))

x_std = StandardScaler().fit_transform(x)
y_std = StandardScaler().fit_transform(y)

reg.fit(x_std, y_std)

score_5 = reg.score(x_std, y_std)
print('score_5:',score_5)
#se conserva la score
#score 0.12009819576246927


# 7,8 dar complejidad a las variables realizando transformaciones no lineales
#regresion polimial

x_train, x_test, y_train, y_test = train_test_split (x_std, y_std, test_size=0.3)
pol_reg = PolynomialFeatures(degree=3)

x_train_pol = pol_reg.fit_transform(x_train)
x_test_pol = pol_reg.fit_transform(x_test)

#regresion
reg.fit(x_train_pol, y_train)
#prediccion
reg.predict(x_train_pol)

score_6 = reg.score(x_train_pol, y_train)
print('score_6:',score_6)
#Score: 0.15285449105845994

#9: Grafica los 6 scores calculados
ls_scores = [score_1,score_4,score_5,score_6]
plt.plot(ls_scores)
print(plt.show())

#CONCLUSIONES paso 9:
# una parte de la grafica es plana debido a que no hubo cambios en la score
# mas que al usar regresion polinomial

