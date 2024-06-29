from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import TheilSenRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import Binarizer
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
import pandas
import numpy
import math

numpy.seterr(invalid='ignore')
data = pandas.read_csv("diamonds.csv") # читаем csv-файл через библиотеку pandas CrabAgePrediction.csv diamonds.csv
data = data.drop(columns = data.columns[0]) # удаляем столбец нумерации, тк он не нужен в наших вычислениях
data = data.drop(data.index[:50000]) # убираем лишние записи для предотвращения долго выполнения алгоритма
print('price', min(data['price']), max(data['price']))

# Избавление от категориальных данных
replace_dicts = [{'Ideal': 1.0, 'Premium': 2.0, 'Very Good': 3.04, 'Good': 4.1, 'Fair': 5.2}, 
                 {'D': 1.0, 'E': 2.0, 'F': 3.03, 'G': 4.079, 'H': 5.13, 'I': 6.2, 'J': 7.389},
                 {'IF': 1.0, 'VVS1': 2.0, 'VVS2': 3.065, 'VS1': 4.11, 'VS2': 5.181, 'SI1': 6.27, 'SI2': 7.377, 'I1': 9.139}]

for rep in replace_dicts:
   data = data.replace(rep)

X = data.drop('price', axis = 1)
y = data['price']

# Предобработка данных
binCut = Binarizer(threshold = 4, copy = False)
binColor = Binarizer(threshold = 4, copy = False)
binClarity = Binarizer(threshold = 6, copy = False)

X_cut = binCut.fit_transform(X['cut'].values.reshape(-1, 1)).ravel()
X['cut'] = X_cut

X_color = binColor.fit_transform(X['color'].values.reshape(-1, 1)).ravel()
X['color'] = X_color

X_clarity = binClarity.fit_transform(X['clarity'].values.reshape(-1, 1)).ravel()
X['clarity'] = X_clarity

print(X.head)

print("-----------------------------------------------------------------")
print("Датасет: Diamond Dataset")
print("-----------------------------------------------------------------")

X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size = 0.7)
Poly_train, Poly_test = PolynomialFeatures(degree = 3).fit_transform(X_train), PolynomialFeatures(degree = 3).fit_transform(X_test) # пропускаем данные через полиномиальную регрессию
AllModels = ['KNeighborsRegressor', 'RadiusNeighborsRegressor', 'LinearSVR', 'SVR', 'LinearRegression', 'Ridge', 'Lasso', 'ElasticNet', 
             'LinearRegression + Полиномиальная регрессия', 'Ridge + Полиномиальная регрессия', 'Lasso + Полиномиальная регрессия', 'ElasticNet + Полиномиальная регрессия', 
             'Полиномиальная регрессия + TheilSenRegressor', 'Полиномиальная регрессия + HuberRegressor', 'RANSACRegressor + LinearRegression', 'RANSACRegressor + Ridge', 
             'RANSACRegressor + Lasso', 'RANSACRegressor + ElasticNet', 'RANSACRegressor + LinearRegression + Полиномиальная регрессия', 
             'RANSACRegressor + Ridge + Полиномиальная регрессия', 'RANSACRegressor + Lasso + Полиномиальная регрессия', 'RANSACRegressor + ElasticNet + Полиномиальная регрессия']
MSE = numpy.array([])

#-------------------------------------------------------------------------------------------------------------------------------------------------------
# KNeighborsRegressor
#-------------------------------------------------------------------------------------------------------------------------------------------------------

BestKNeighbor = numpy.array([])

for n in range(1, 20):
   model = KNeighborsRegressor(n_neighbors = n)
   model.fit(X_train, Y_train)
   BestKNeighbor = numpy.append(BestKNeighbor, mean_squared_error(Y_test, model.predict(X_test)))

print(BestKNeighbor)
print("Лучшее число соседей", numpy.argmin(BestKNeighbor) + 1, "с среднеквадратичной ошибкой", math.sqrt(BestKNeighbor[numpy.argmin(BestKNeighbor)]))
MSE = numpy.append(MSE, math.sqrt(BestKNeighbor[numpy.argmin(BestKNeighbor)]))
print("Score: ", model.score(X_test, Y_test) * 100)
print("-----------------------------------------------------------------")

#-------------------------------------------------------------------------------------------------------------------------------------------------------
# RadiusNeighborsRegressor
#-------------------------------------------------------------------------------------------------------------------------------------------------------

BestRadiusNeighbor = numpy.array([])

for r in range(20, 100, 10):
   #r = i * 0.1
   model = RadiusNeighborsRegressor(radius = r)
   model.fit(X_train, Y_train)
   BestRadiusNeighbor = numpy.append(BestRadiusNeighbor, mean_squared_error(Y_test, model.predict(X_test)))

print(BestRadiusNeighbor)
print("Лучший радиус", numpy.argmin(BestRadiusNeighbor) * 10 + 20, "с среднеквадратичной ошибкой", math.sqrt(BestRadiusNeighbor[numpy.argmin(BestRadiusNeighbor)]))
MSE = numpy.append(MSE, math.sqrt(BestRadiusNeighbor[numpy.argmin(BestRadiusNeighbor)]))
print("Score: ", model.score(X_test, Y_test) * 100)
print("-----------------------------------------------------------------")

#-------------------------------------------------------------------------------------------------------------------------------------------------------
# LinearSVR
#-------------------------------------------------------------------------------------------------------------------------------------------------------

model = LinearSVR(dual = True, max_iter = 100000000)
model.fit(X_train, Y_train)

mse = mean_squared_error(Y_test, model.predict(X_test))
print("Cреднеквадратичная ошибка linearSVR", math.sqrt(mse))
MSE = numpy.append(MSE, math.sqrt(mse))
print("Score: ", model.score(X_test, Y_test) * 100)
print("-----------------------------------------------------------------")

#-------------------------------------------------------------------------------------------------------------------------------------------------------
# SVR
#-------------------------------------------------------------------------------------------------------------------------------------------------------

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
BestSVR = numpy.array([])

for kernel in kernels:
   model = SVR(kernel = kernel)
   model.fit(X_train, Y_train)
   BestSVR = numpy.append(BestSVR, mean_squared_error(Y_test, model.predict(X_test)))

print(kernels)
print(BestSVR)
print("Лучшее ядро", kernels[numpy.argmin(BestSVR)], "с среднеквадратичной ошибкой", math.sqrt(BestSVR[numpy.argmin(BestSVR)]))
MSE = numpy.append(MSE, math.sqrt(BestSVR[numpy.argmin(BestSVR)]))
print("Score: ", model.score(X_test, Y_test) * 100)
print("-----------------------------------------------------------------")

#-------------------------------------------------------------------------------------------------------------------------------------------------------
# LinearRegression
#-------------------------------------------------------------------------------------------------------------------------------------------------------
model = LinearRegression()
model.fit(X_train, Y_train)

mse = mean_squared_error(Y_test, model.predict(X_test))
print("Cреднеквадратичная ошибка LinearRegression", math.sqrt(mse))
MSE = numpy.append(MSE, math.sqrt(mse))
print("Score: ", model.score(X_test, Y_test) * 100)
print("-----------------------------------------------------------------")

#-------------------------------------------------------------------------------------------------------------------------------------------------------
# Ridge
#-------------------------------------------------------------------------------------------------------------------------------------------------------

BestRidge = numpy.array([])

for a in range(20, 40):
   alpha = a / 2
   model = Ridge(alpha = alpha)
   model.fit(X_train, Y_train)
   BestRidge = numpy.append(BestRidge, mean_squared_error(Y_test, model.predict(X_test)))

print(BestRidge)
print("Лучший alpha в Ridge", numpy.argmin(BestRidge) / 2 + 20, "с среднеквадратичной ошибкой", math.sqrt(BestRidge[numpy.argmin(BestRidge)]))
MSE = numpy.append(MSE, math.sqrt(BestRidge[numpy.argmin(BestRidge)]))
print("Score: ", model.score(X_test, Y_test) * 100)
print("-----------------------------------------------------------------")

#-------------------------------------------------------------------------------------------------------------------------------------------------------
# Lasso
#-------------------------------------------------------------------------------------------------------------------------------------------------------

BestLasso = numpy.array([])

for a in range(10, 31):
   alpha = a / 2
   model = Lasso(alpha = alpha)
   model.fit(X_train, Y_train)
   BestLasso = numpy.append(BestLasso, mean_squared_error(Y_test, model.predict(X_test)))

print(BestLasso)
print("Лучший alpha в Lasso", (numpy.argmin(BestLasso) + 10) / 2, "с среднеквадратичной ошибкой", math.sqrt(BestLasso[numpy.argmin(BestLasso)]))
MSE = numpy.append(MSE, math.sqrt(BestLasso[numpy.argmin(BestLasso)]))
print("Score: ", model.score(X_test, Y_test) * 100)
print("-----------------------------------------------------------------")

#-------------------------------------------------------------------------------------------------------------------------------------------------------
# ElasticNet
#-------------------------------------------------------------------------------------------------------------------------------------------------------

BestElastic = numpy.array([])

for ratio in range(10):
   alpha = ratio * 2 + 1
   l1 = (10 - ratio) / 10

   model = ElasticNet(alpha = alpha, l1_ratio = l1, max_iter = 100000)
   model.fit(X_train, Y_train)
   BestElastic = numpy.append(BestElastic, mean_squared_error(Y_test, model.predict(X_test)))

print(BestElastic)
print("Лучшие alpha и l1_ratio в Полиномиальной + ElasticNet", numpy.argmin(BestElastic) * 2 + 1, (10 - numpy.argmin(BestElastic)) / 10, "с среднеквадратичной ошибкой", math.sqrt(BestElastic[numpy.argmin(BestElastic)]))
MSE = numpy.append(MSE, math.sqrt(BestElastic[numpy.argmin(BestElastic)]))
print("Score: ", model.score(X_test, Y_test))
print("-----------------------------------------------------------------")

#-------------------------------------------------------------------------------------------------------------------------------------------------------
# LinearRegression + Полиномиальная регрессия
#-------------------------------------------------------------------------------------------------------------------------------------------------------

model = LinearRegression()
model.fit(Poly_train, Y_train)

mse = mean_squared_error(Y_test, model.predict(Poly_test))
print("Cреднеквадратичная ошибка Полиномиальная + LinearRegression", math.sqrt(mse))
MSE = numpy.append(MSE, math.sqrt(mse))
print("Score: ", model.score(Poly_test, Y_test) * 100)
print("-----------------------------------------------------------------")

#-------------------------------------------------------------------------------------------------------------------------------------------------------
# Ridge + Полиномиальная регрессия
#-------------------------------------------------------------------------------------------------------------------------------------------------------

BestRidge = numpy.array([])

for a in range(100, 120):
   alpha = a / 2
   model = Ridge(alpha = alpha)
   model.fit(Poly_train, Y_train)
   BestRidge = numpy.append(BestRidge, mean_squared_error(Y_test, model.predict(Poly_test)))

print(BestRidge)
print("Лучший alpha в Полиномиальной + Ridge", numpy.argmin(BestRidge) / 2 + 100, "с среднеквадратичной ошибкой", math.sqrt(BestRidge[numpy.argmin(BestRidge)]))
MSE = numpy.append(MSE, math.sqrt(BestRidge[numpy.argmin(BestRidge)]))
print("Score: ", model.score(Poly_test, Y_test) * 100)
print("-----------------------------------------------------------------")

#-------------------------------------------------------------------------------------------------------------------------------------------------------
# Lasso + Полиномиальная регрессия
#-------------------------------------------------------------------------------------------------------------------------------------------------------

BestLasso = numpy.array([])

for a in range(10000, 11000, 100):
   model = Lasso(alpha = a)
   model.fit(Poly_train, Y_train)
   BestLasso = numpy.append(BestLasso, mean_squared_error(Y_test, model.predict(Poly_test)))

print(BestLasso)
print("Лучший alpha в Полиномиальной + Lasso", numpy.argmin(BestLasso) * 100 + 10000, "с среднеквадратичной ошибкой", math.sqrt(BestLasso[numpy.argmin(BestLasso)]))
MSE = numpy.append(MSE, math.sqrt(BestLasso[numpy.argmin(BestLasso)]))
print("Score: ", model.score(Poly_test, Y_test) * 100)
print("-----------------------------------------------------------------")

#-------------------------------------------------------------------------------------------------------------------------------------------------------
# ElasticNet + Полиномиальная регрессия
#-------------------------------------------------------------------------------------------------------------------------------------------------------

#Lasso - aplha is big, Ridge - alpha is small
#Lasso - li_ratio > 0.5, Ridge - l1_ratio < 0.5

BestElastic = numpy.array([])

for ratio in range(10):
   alpha = ratio * 10000 + 10000
   l1 = (10 - ratio) / 10

   model = ElasticNet(alpha = alpha, l1_ratio = l1, max_iter = 100000)
   model.fit(Poly_train, Y_train)
   BestElastic = numpy.append(BestElastic, mean_squared_error(Y_test, model.predict(Poly_test)))

print(BestElastic)
print("Лучшие alpha и l1_ratio в Полиномиальной + ElasticNet", numpy.argmin(BestElastic) * 10000 + 10000, (10 - numpy.argmin(BestElastic)) / 10, "с среднеквадратичной ошибкой", math.sqrt(BestElastic[numpy.argmin(BestElastic)]))
MSE = numpy.append(MSE, math.sqrt(BestElastic[numpy.argmin(BestElastic)]))
print("Score: ", model.score(Poly_test, Y_test) * 100)
print("-----------------------------------------------------------------")

#-------------------------------------------------------------------------------------------------------------------------------------------------------
# Полиномиальная регрессия + TheilSenRegressor
#-------------------------------------------------------------------------------------------------------------------------------------------------------

model = TheilSenRegressor(random_state = 100000, max_iter = 1000000, n_subsamples = 500, max_subpopulation = 150)
model.fit(Poly_train, Y_train.ravel())

mse = mean_squared_error(Y_test, model.predict(Poly_test))
print("Cреднеквадратичная ошибка Полиномиальная + TheilSenRegressor", math.sqrt(mse))
MSE = numpy.append(MSE, math.sqrt(mse))
print("Score: ", model.score(Poly_test, Y_test) * 100)
print("-----------------------------------------------------------------")

#-------------------------------------------------------------------------------------------------------------------------------------------------------
# Полиномиальная регрессия + HuberRegressor
#-------------------------------------------------------------------------------------------------------------------------------------------------------

model = HuberRegressor(epsilon = 100, max_iter=10000, alpha=0.0001)
model.fit(Poly_train, Y_train)

mse = mean_squared_error(Y_test, model.predict(Poly_test))
print("Cреднеквадратичная ошибка Полиномиальная + HuberRegressor", math.sqrt(mse))
MSE = numpy.append(MSE, math.sqrt(mse))
print("Score: ", model.score(Poly_test, Y_test) * 100)
print("-----------------------------------------------------------------")

#-------------------------------------------------------------------------------------------------------------------------------------------------------
# RANSACRegressor + LinearRegression
#-------------------------------------------------------------------------------------------------------------------------------------------------------

# min_samples - число точек модели (берем примерно 70% тестовой части)
# residual_threshold - максимальная возможная степень отклонения точки до того, как она перестанет быть частью модели

model = RANSACRegressor(LinearRegression(), min_samples = 2500, residual_threshold = 50.0) 
model.fit(X_train, Y_train)

mse = mean_squared_error(Y_test, model.predict(X_test))
print("Cреднеквадратичная ошибка RANSACRegressor + LinearRegression", math.sqrt(mse))
MSE = numpy.append(MSE, math.sqrt(mse))
print("Score: ", model.score(X_test, Y_test) * 100)
print("-----------------------------------------------------------------")

#-------------------------------------------------------------------------------------------------------------------------------------------------------
# RANSACRegressor + Ridge
#-------------------------------------------------------------------------------------------------------------------------------------------------------

BestRidge = numpy.array([])

for a in range(21):
   alpha = a / 2
   model = RANSACRegressor(Ridge(alpha = alpha) , min_samples = 2500, residual_threshold = 50.0)
   model.fit(X_train, Y_train)
   BestRidge = numpy.append(BestRidge, mean_squared_error(Y_test, model.predict(X_test)))

print(BestRidge)
print("Лучший alpha в RANSACRegressor + Ridge", numpy.argmin(BestRidge) / 2, "с среднеквадратичной ошибкой", math.sqrt(BestRidge[numpy.argmin(BestRidge)]))
MSE = numpy.append(MSE, math.sqrt(BestRidge[numpy.argmin(BestRidge)]))
print("Score: ", model.score(X_test, Y_test) * 100)
print("-----------------------------------------------------------------")

#-------------------------------------------------------------------------------------------------------------------------------------------------------
# RANSACRegressor + Lasso
#-------------------------------------------------------------------------------------------------------------------------------------------------------

BestLasso = numpy.array([])

for a in range(100, 200, 10):
   alpha = a / 2
   model = RANSACRegressor(Lasso(alpha = alpha), min_samples = 2500, residual_threshold = 50.0)
   model.fit(X_train, Y_train)
   BestLasso = numpy.append(BestLasso, mean_squared_error(Y_test, model.predict(X_test)))

print(BestLasso)
print("Лучший alpha в RANSACRegressor + Lasso", (numpy.argmin(BestLasso) * 10 + 100) / 2, "с среднеквадратичной ошибкой", math.sqrt(BestLasso[numpy.argmin(BestLasso)]))
MSE = numpy.append(MSE, math.sqrt(BestLasso[numpy.argmin(BestLasso)]))
print("Score: ", model.score(X_test, Y_test) * 100)
print("-----------------------------------------------------------------")

#-------------------------------------------------------------------------------------------------------------------------------------------------------
# RANSACRegressor + ElasticNet
#-------------------------------------------------------------------------------------------------------------------------------------------------------

#Lasso - aplha is big, Ridge - alpha is small
#Lasso - li_ratio < 0.5, Ridge - l1_ratio > 0.5

BestElastic = numpy.array([])

for ratio in range(10):
   alpha = ratio * 10 + 5
   l1 = (10 - ratio) / 10

   model = RANSACRegressor(ElasticNet(alpha = alpha, l1_ratio = l1), min_samples = 2500, residual_threshold = 50.0)
   model.fit(X_train, Y_train)
   BestElastic = numpy.append(BestElastic, mean_squared_error(Y_test, model.predict(X_test)))

print(BestElastic)
print("Лучшие alpha и l1_ratio в RANSACRegressor + ElasticNet", numpy.argmin(BestElastic) * 2 + 1, (10 - numpy.argmin(BestElastic)) / 10, "с среднеквадратичной ошибкой", math.sqrt(BestElastic[numpy.argmin(BestElastic)]))
MSE = numpy.append(MSE, math.sqrt(BestElastic[numpy.argmin(BestElastic)]))
print("Score: ", model.score(X_test, Y_test) * 100)
print("-----------------------------------------------------------------")

#-------------------------------------------------------------------------------------------------------------------------------------------------------
# RANSACRegressor + LinearRegression + Полиномиальная регрессия
#-------------------------------------------------------------------------------------------------------------------------------------------------------

# min_samples - число точек модели (берем примерно 70% тестовой части)
# residual_threshold - максимальная возможная степень отклонения точки до того, как она перестанет быть частью модели

model = RANSACRegressor(LinearRegression(), min_samples = 2500, residual_threshold = 50.0) 
model.fit(Poly_train, Y_train)

mse = mean_squared_error(Y_test, model.predict(Poly_test))
print("Cреднеквадратичная ошибка Полиномиальная + RANSACRegressor + LinearRegression", math.sqrt(mse))
MSE = numpy.append(MSE, math.sqrt(mse))
print("Score: ", model.score(Poly_test, Y_test) * 100)
print("-----------------------------------------------------------------")

#-------------------------------------------------------------------------------------------------------------------------------------------------------
# RANSACRegressor + Ridge + Полиномиальная регрессия
#-------------------------------------------------------------------------------------------------------------------------------------------------------

BestRidge = numpy.array([])

for a in range(21):
   alpha = a / 2
   model = RANSACRegressor(Ridge(alpha = alpha) , min_samples = 2500, residual_threshold = 50.0)
   model.fit(Poly_train, Y_train)
   BestRidge = numpy.append(BestRidge, mean_squared_error(Y_test, model.predict(Poly_test)))

print(BestRidge)
print("Лучший alpha в Полиномиальная + RANSACRegressor + Ridge", numpy.argmin(BestRidge) / 2, "с среднеквадратичной ошибкой", math.sqrt(BestRidge[numpy.argmin(BestRidge)]))
MSE = numpy.append(MSE, math.sqrt(BestRidge[numpy.argmin(BestRidge)]))
print("Score: ", model.score(Poly_test, Y_test) * 100)
print("-----------------------------------------------------------------")

#-------------------------------------------------------------------------------------------------------------------------------------------------------
# RANSACRegressor + Lasso + Полиномиальная регрессия
#-------------------------------------------------------------------------------------------------------------------------------------------------------

BestLasso = numpy.array([])

for a in range(100000, 200000, 10000):
   alpha = a / 2
   model = RANSACRegressor(Lasso(alpha = alpha), min_samples = 2500, residual_threshold = 50.0)
   model.fit(Poly_train, Y_train)
   BestLasso = numpy.append(BestLasso, mean_squared_error(Y_test, model.predict(Poly_test)))

print(BestLasso)
print("Лучший alpha в Полиномиальная + RANSACRegressor + Lasso", (numpy.argmin(BestLasso) * 10000 + 100000) / 2, "с среднеквадратичной ошибкой", math.sqrt(BestLasso[numpy.argmin(BestLasso)]))
MSE = numpy.append(MSE, math.sqrt(BestLasso[numpy.argmin(BestLasso)]))
print("Score: ", model.score(Poly_test, Y_test) * 100)
print("-----------------------------------------------------------------")

#-------------------------------------------------------------------------------------------------------------------------------------------------------
# RANSACRegressor + ElasticNet + Полиномиальная регрессия
#-------------------------------------------------------------------------------------------------------------------------------------------------------

#Lasso - aplha is big, Ridge - alpha is small
#Lasso - li_ratio < 0.5, Ridge - l1_ratio > 0.5

BestElastic = numpy.array([])

for ratio in range(10):
   alpha = ratio * 100000 + 100000
   l1 = (10 - ratio) / 10

   model = RANSACRegressor(ElasticNet(alpha = alpha, l1_ratio = l1), min_samples = 2500, residual_threshold = 50.0)
   model.fit(Poly_train, Y_train)
   BestElastic = numpy.append(BestElastic, mean_squared_error(Y_test, model.predict(Poly_test)))

print(BestElastic)
print("Лучшие alpha и l1_ratio в Полиномиальная + RANSACRegressor + ElasticNet", numpy.argmin(BestElastic) * 100000 + 100000, (10 - numpy.argmin(BestElastic)) / 10, "с среднеквадратичной ошибкой", math.sqrt(BestElastic[numpy.argmin(BestElastic)]))
MSE = numpy.append(MSE, math.sqrt(BestElastic[numpy.argmin(BestElastic)]))
print("Score: ", model.score(Poly_test, Y_test) * 100)
print("-----------------------------------------------------------------")

#-------------------------------------------------------------------------------------------------------------------------------------------------------
# Итоги
#-------------------------------------------------------------------------------------------------------------------------------------------------------

print("Лучше всего отработала модель", AllModels[numpy.argmin(MSE)], "с среднеквадратической ошибкой", MSE[numpy.argmin(MSE)])