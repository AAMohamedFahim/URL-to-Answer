# Use Python 3.12.3 as base image
FROM python:3.12.3-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY main_api.py .
COPY interface.py .
COPY .env .

# Expose ports for FastAPI and Streamlit
# FastAPI typically uses 8000, Streamlit typically uses 8501
EXPOSE 8000 8501

# Create entry point script
RUN echo '#!/bin/bash\n\
python main_api.py & \n\
streamlit run interface.py\n\
wait\n' > /app/entrypoint.sh && \
chmod +x /app/entrypoint.sh

# Run both services
CMD ["/app/entrypoint.sh"]