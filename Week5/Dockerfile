ARG PYTHON_VERSION=3.11.6
FROM python:${PYTHON_VERSION}-slim

# set the working directory
ADD . /app

WORKDIR /app

# copy the requirements.txt
COPY requirements.txt /app/requirements.txt

# Install the dependencies
RUN pip install -r /app/requirements.txt 

# copy the py file to /app
COPY rest_api_template.py /app

EXPOSE 57700

# run the application 
CMD ["python", "rest_api_template.py"]