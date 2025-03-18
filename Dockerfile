# 1. Use Python 3.9 as the base image
FROM python:3.9

# 2. Set the working directory
WORKDIR /app

# 3. Install Poetry
RUN pip install --no-cache-dir poetry

# 4. Copy all files in the MIDSTModels directory to the /app directory
COPY . /app

# 5. Set Poetry to use the virtual environment external environment
RUN poetry config virtualenvs.create false

# 6. Install MIDSTModels dependencies (including tabsyn and clavaddpm)
RUN poetry install --with "tabsyn, clavaddpm"

# 7. Install Jupyter Notebook
RUN pip install jupyter

# 8. Start Jupyter Notebook by default when running the container
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]