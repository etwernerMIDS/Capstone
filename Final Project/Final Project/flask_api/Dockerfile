FROM jupyter/scipy-notebook

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY api.py ./api.py
COPY arima_prod.py ./arima_prod.py
COPY xgboost_prod.py ./xgboost_prod.py
COPY xgboost.joblib ./xgboost.joblib

EXPOSE 8000
CMD ["python", "api.py"]
