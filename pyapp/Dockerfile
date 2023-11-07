# Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container to /app
WORKDIR /pyapp

# Copy package.json and yarn.lock
COPY requirements.txt  ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
# This includes the requirements.txt file
COPY . /pyapp



