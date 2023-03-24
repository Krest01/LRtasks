import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import numpy as np

# 1
df = pd.read_csv('LifeExpectancy.csv')
# df.replace('NaN', np.NaN, inplace=True)
# imp = SimpleImputer(missing_values=np.NaN)
# idf = pd.DataFrame(imp.fit_transform(df))
# idf.index = df.index

# 2a

train_ds = df.sample(frac=0.8)
test_ds = df.drop(train_ds.index)
print(len(train_ds), len(test_ds))

# 2c
countries = df.sort_values(by='Life expectancy', ascending=False)
countries = countries[['Country', 'Year', 'Life expectancy']]
print(f"Best life expectancy \n {countries.nlargest(3, 'Life expectancy')}")

# 3 GDP
GDP_ds = train_ds[['GDP', 'Life expectancy']]
x1 = GDP_ds[['Life expectancy']]
y1 = GDP_ds[['GDP']]
imp = SimpleImputer(strategy='most_frequent')
imp_x1 = pd.DataFrame(imp.fit_transform(x1))
imp_y1 = pd.DataFrame(imp.fit_transform(y1))

model1 = LinearRegression(fit_intercept=True).fit(imp_x1, imp_y1)
print(model1.score(imp_x1, imp_y1))
print(model1.coef_)

# 3 Total Expenditure
TE_ds = train_ds[['Total expenditure', 'Life expectancy']]
x2 = TE_ds[['Life expectancy']]
y2 = TE_ds[['Total expenditure']]
imp_x2 = pd.DataFrame(imp.fit_transform(x2))
imp_y2 = pd.DataFrame(imp.fit_transform(y2))

model2 = LinearRegression(fit_intercept=True).fit(imp_x2, imp_y2)
print(model2.score(imp_x2, imp_y2))
print(model2.coef_)

# 3 Alcohol
A_ds = train_ds[['Alcohol', 'Life expectancy']]
x3 = A_ds[['Life expectancy']]
y3 = A_ds[['Alcohol']]
imp_x3 = pd.DataFrame(imp.fit_transform(x3))
imp_y3 = pd.DataFrame(imp.fit_transform(y3))

model3 = LinearRegression(fit_intercept=True).fit(imp_x3, imp_y3)
print(model3.score(imp_x3, imp_y3))
print(model3.coef_)

x_pred1 = np.linspace(35, 90, 200)
print(x_pred1)
x_pred1 = x_pred1.reshape(-1, 1)
y_pred1 = model1.predict(x_pred1)


plt.style.use('default')
plt.style.use('ggplot')

fig1, ax1 = plt.subplots(figsize=(7, 3.5))

ax1.plot(x_pred1, y_pred1, color='r', label='Regression line', linewidth=4, alpha=0.5)
ax1.scatter(x1, y1, edgecolor='k', facecolor='blue', alpha=0.7, label='data')
ax1.set_ylabel('y', fontsize=20)
ax1.set_xlabel('x', fontsize=20)
ax1.legend(facecolor='white', fontsize=11)
ax1.text(0.55, 0.15, f'$y = {round(model1.coef_[0][0], 4)} x - {round(abs(model1.intercept_[0]), 2)} $', fontsize=17,
        transform=ax1.transAxes)

fig1.tight_layout()
plt.show()