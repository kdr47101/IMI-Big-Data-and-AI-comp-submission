# Use the official TensorFlow 2 CPU base image
FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-11.py310@sha256:1337b180d99ceb024c3a52854947811703e7265244522cfc69aeed0b55a45154

# Set the environment variable to use the legacy pip resolver
ENV PIP_USE_LEGACY_RESOLVER=1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files into the container
COPY . .

# Ensure the pipeline script is executable
RUN chmod +x run_pipeline.sh

# Set the entrypoint to run your pipeline via the shell script
ENTRYPOINT ["sh", "run_pipeline.sh"]
