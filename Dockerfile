FROM python:3.11-slim

WORKDIR /code

# Install system-level dependencies if needed
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /code/requirements.txt

# Force installation to the main system environment
RUN pip install --no-cache-dir -r /code/requirements.txt

COPY . .

# Run uvicorn directly
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]