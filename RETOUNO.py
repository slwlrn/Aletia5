# Importamos las bilbiotecas necesarias
import matplotlib.pyplot as plt
import seaborn as sns;
sns.set()
import numpy as np
import pandas as pd

# Cargamos los dos datasets
counts = pd.read_csv('Fremont_Bridge_Bicycle_Counter.csv', index_col='Date', parse_dates=True)
weather = pd.read_csv('BicycleWeather.csv', index_col='DATE', parse_dates=True)

# Calculamos el trafico diario de bicicletas
daily = counts.resample('d').sum()
daily['Total'] = daily.sum(axis=1)
daily = daily[['Total']]

# Hint: Algunos días como los fines de semana incentivan el uso de bicicleta
# Solución: Creamos las primeras 7 variables las cuales serán un ID del día de la semana
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
for i in range(7):
    daily[days[i]] = (daily.index.dayofweek == i).astype(float)

# Hint: Los días libres son buenos para andar en bicicleta
# Solución: Crearemos también una variable para identificar los días festivos
from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()
holidays = cal.holidays('2012', '2016')
daily = daily.join(pd.Series(1, index=holidays, name='holiday'))
daily['holiday'].fillna(0, inplace=True)

# Hint: Seattle es una ciudad donde el número de horas de luz en el día varían mucho durante el año
# Solución: Crearemos una variable que nos indique el número de horas de luz en el día
def hours_of_daylight(date, axis=23.44, latitude=47.61):
    """Compute the hours of daylight for the given date"""
    days = (date - pd.datetime(2000, 12, 21)).days
    m = (1. - np.tan(np.radians(latitude))
         * np.tan(np.radians(axis) * np.cos(days * 2 * np.pi / 365.25)))
    return 24. * np.degrees(np.arccos(1 - np.clip(m, 0, 2))) / 180.

daily['daylight_hrs'] = list(map(hours_of_daylight, daily.index))
daily[['daylight_hrs']].plot()
plt.ylim(8, 17)

# Hint: El clima es importante cuando queremos salir a montar bicicleta
# Solución: Calculamos un promedio de temperatura por día
# Agregamos una variable de probabilidad de lluvia
# Agregamos una variable para los días sin lluvia (es importante saber si habrá poca lluvia pero más importante saber si no habrá)

# Las temperaturas estan en 1/10 grados C; Convertimos los datos a grados C
weather['TMIN'] /= 10
weather['TMAX'] /= 10
weather['Temp (C)'] = 0.5 * (weather['TMIN'] + weather['TMAX'])
# La precipitación esta en 1/10 mm; Convertimos a pulgadas
weather['PRCP'] /= 254
weather['dry day'] = (weather['PRCP'] == 0).astype(int)
daily = daily.join(weather[['PRCP', 'Temp (C)', 'dry day']])

# Hint: El calentamiento global modifica las temperaturas cada año
# Sol: Creamos una variable "contador" que nos ayuda a dar al modelo intuición sobre el año en curso
daily['annual'] = (daily.index - daily.index[0]).days / 365.

# Veamos el resultado
print(daily.head())


# Importamos las bibliotecas
from sklearn.linear_model import LinearRegression

# Eliminamos las filas con valores nulos
#deleting nans
df = daily.dropna()


print(df.head())

# Seleccionamos las variables X a utilizar
# Columnas: 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'holiday', 'daylight_hrs', 'PRCP', 'dry day', 'Temp (C)', 'annual'
x = np.array(df.drop('Total', axis=1))
print(x.shape)

# Seleccionamos la variable Y a predecir
# Columna: 'Total'
y = np.array(df['Total'])
print(y.shape)


# Creamos un modelo de regresión lineal.
# Las variables de día de la semana funcionan como intercepto,
# por eso decidimos configurar la regresión para que no incluya un intercepto
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
lr = LinearRegression(fit_intercept=False)
reg = lr.fit(x, y)

# Entrenamos el modelo
print(reg.score(x,y))
print(reg.coef_)
print(reg.intercept_)

# Calculamos la predicción

df['predicted'] = reg.predict(x)

df[['Total', 'predicted']].plot(alpha=0.5);

plt.show()



