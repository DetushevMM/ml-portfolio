# 🚀 Машинное обучение: Классификация и Кластеризация

Коллекция проектов по машинному обучению, охватывающая задачи **регрессии, классификации, кластеризации и построения рекомендательных систем**. Ниже представлена сводная таблица всех проектов в репозитории.

## 📊 Сводная таблица проектов

| № | Название проекта | Задача / Описание | Тип | Алгоритмы | Лучшая метрика |
|---|---|---|---|---|---|
| 1 | **Property Price Prediction Regression** | *Прогнозирование стоимости недвижимости.* | Регрессия | Random Forest/ RandomizedSearchCV | R²: 0.82 |
| 2 | **Analises Houses and Price Prediction Model, Elastic_Grid** | *Анализ домов и построение модели прогноза цен.* | Регрессия | Elastic Net, GridSearch | MAE / Средняя цена	~93.5% |
| 3 | **Car Price Prediction** | *Прогнозирование стоимости автомобилей.* | Регрессия | Глубокая полносвязная сеть перцептрон | MAE/Средняя цена	~12.86% |
| 4 | **Classification of writers** | *Классификация авторов текстов.* | Классификация | Embedding + Dense | accuracy ~ 72% |
| 5 | **Electronics Store Recommender System** | *Система рекомендаций для магазина электроники.* | Рекомендательные системы | Item-Item Collaborative Filtering | RMSE ~0.39 / MAE	~0.20 |
| 6 | **Hierarchical Clustering of Patents** | *Иерархическая кластеризация патентов.* | Кластеризация |  Hierarchical Clustering | — |
| 7 | **Impact of advertising on sales, Polynomial regression analysis** | *Анализ влияния рекламы на продажи.* | Регрессия | Полиномиальная регрессия | — |
| 8 | **Predicting Heart Disease: Logistic Regression Classification** | *Прогнозирование сердечных заболеваний.* | Классификация | LogisticRegression | f1-score 0.84 |
| 9 | **Predicting the density of rock_RandomForestRegressor_Model comparison** | *Прогнозирование плотности горной породы.* | Регрессия | Random Forest | — |
| 10 | **Classification client and Credit Approval** | *Система одобрения кредитов* | Классификация | **Logistic Regression** | **F1 0.85** |
| 11 | **Credit Fraud XGB Analysis** | *Обнаружение мошенничества* | Классификация | **XGBoost** | **ROC-AUC 0.98** |
| 12 | **Glass identification classification** | *Идентификация типа стекла* | Классификация | **Decision Tree** | **Precision 0.76** |
| 13 | **Predicting subscriber churn_Boosting** | *Прогноз оттока клиентов* | Классификация | **AdaBoost** | **Recall 0.91** |
| 14 | **Country Clustering** | *Группировка стран* | Кластеризация | **K-Means++** | **(метод локтя)** |
| 15 | **Population of popularity in social networks_Clusterization** | *Анализ активности в соцсетях* | Кластеризация | **GMM** | **(BIC)** |
| 16 | **Features of cancer tumors method_PCA** | *Исследование характеристик опухолей* | Снижение размерности | **Kernel PCA** | — |
| 17 | **clustering of wholesale customers_DBSCAN** | *Сегментация оптовых покупателей* | Кластеризация | **DBSCAN** | **Silhouette 0.61** |

---

### 🗂️ Классификация проектов по типам задач

**Регрессия (5 проектов)**
*   Property Price Prediction Regression
*   Analises Houses and Price Prediction Model, Elastic_Grid
*   Car Price Prediction
*   Impact of advertising on sales, Polynomial regression analysis
*   Predicting the density of rock_RandomForestRegressor_Model comparison

**Классификация (6 проектов)**
*   Classification of writers
*   Predicting Heart Disease: Logistic Regression Classification
*   Classification client and Credit Approval
*   Credit Fraud XGB Analysis
*   Glass identification classification
*   Predicting subscriber churn_Boosting

**Кластеризация и снижение размерности (5 проектов)**
*   Hierarchical Clustering of Patents
*   Country Clustering
*   Population of popularity in social networks_Clusterization
*   Features of cancer tumors method_PCA
*   clustering of wholesale customers_DBSCAN

**Рекомендательные системы (1 проект)**
*   Electronics Store Recommender System

## 🛠 Технологический стек и запуск

### Используемые библиотеки Python
```python
# Классификация
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

# Кластеризация и снижение размерности
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, KernelPCA

# Общие инструменты
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             silhouette_score, calinski_harabasz_score)

# Визуализация
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px  # Для интерактивных 3D-графиков

Рекомендуемые версии
Python >= 3.8
scikit-learn >= 1.0.0
pandas >= 1.3.0
xgboost >= 1.5.0

🚀 Быстрый старт
Клонируйте репозиторий:

bash
git clone https://github.com/DetushevMM/ml-portfolio.git
cd ml-portfolio
Установите зависимости: Рекомендуется создать виртуальное окружение.

bash
pip install -r requirements.txt  # Если файл есть

# Или установите основные библиотеки:
pip install numpy pandas scikit-learn matplotlib seaborn xgboost plotly

Запустите Jupyter Notebook:
bash
jupyter notebook
Откройте интересующий вас ноутбук из папки notebooks/.

📚 Полезные материалы
Scikit-learn: Choosing the right estimator — карта выбора алгоритма.

Scikit-learn: Clustering guide — руководство по кластеризации.

Официальная документация XGBoost — подробное руководство по градиентному бустингу.
