# Base Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy the Python script
COPY company_scoring.py /app/

# If you have dependencies, include requirements.txt
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Ensure Python output is unbuffered
ENV PYTHONUNBUFFERED=1

# Default command to run your script
CMD ["python", "/app/company_scoring.py"]

