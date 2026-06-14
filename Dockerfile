FROM python:3.11-slim

# Set up a new user named "user" with UID 1000 (Required by Hugging Face)
RUN useradd -m -u 1000 user

WORKDIR /code

# Install system-level dependencies and clean up cache to keep the image slim
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

# Copy the rest of your application files and explicitly change ownership to our new user
COPY --chown=user:user . .

# Switch to the non-root user
USER user

# Inform Docker/Hugging Face that the container listens on port 7860
EXPOSE 7860

# Run uvicorn directly (using the same port format expected by Spaces)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]