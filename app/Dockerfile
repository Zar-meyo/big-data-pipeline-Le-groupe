FROM python:3.12

# Installe curl et OpenJDK 11
RUN apt-get update && \
    apt-get install -y curl default-jdk && \
    apt-get clean;

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "./ingestion.py" ]
