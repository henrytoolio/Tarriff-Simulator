FROM python:3.9-slim

WORKDIR /
EXPOSE 8501

COPY / ./
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

ENTRYPOINT ["streamlit", "run", "src/Home.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.runOnSave=true"]

