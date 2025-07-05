# Use the specified base image
RUN curl -I https://ghcr.io/v2/
RUN docker pull ghcr.io/dimvy-clothing-brand:latest --debug
RUN docker login ghcr.io
RUN sudo systemctl start docker
RUN docker run hello-world
RUN sudo systemctl start docker
FROM ghcr.io/dimvy-clothing-brand:latest

# Add any additional instructions below
# Example: Install dependencies
# RUN apk add --no-cache <package-name>

# Set the working directory
WORKDIR /app

# Copy application files
COPY . /app

# Define the default command
CMD ["sh"]

LABEL "org.opencontainers.image.description"="DESCRIPTION"
