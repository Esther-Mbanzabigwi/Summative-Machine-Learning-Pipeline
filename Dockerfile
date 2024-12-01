# Use Python 3.7.6 as the base image
FROM python:3.7.6-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Upgrade pip to the latest compatible version for Python 3.7
RUN pip install --no-cache-dir --upgrade pip

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit uses
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.enableCORS=false"]
