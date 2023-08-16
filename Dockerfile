FROM python:3.8-slim-buster

RUN apt update -y && apt install awscli -y
WORKDIR /app

COPY . /app
RUN pip install -r requirements.txt

# Expose port 80
EXPOSE 80

# Command to run the application on port 80
CMD ["streamlit", "run", "--server.port", "80", "app.py"]
