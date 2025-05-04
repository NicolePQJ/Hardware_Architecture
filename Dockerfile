# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
# You can also install individual packages here if you don’t have a requirements.txt
RUN pip install --upgrade pip && \
    pip install matplotlib jupyter pandas numpy

# Expose port 8888 for Jupyter notebook
EXPOSE 8888

# Set up the entrypoint for Jupyter
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''"]
