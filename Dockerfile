# Use the specified base image
curl -I https://ghcr.io/v2/docker pull ghcr.io/dimvy-clothing-brand:latest --debugdocker login ghcr.iocurl -I https://ghcr.io/v2/sudo systemctl start dockerdocker run hello-worldsudo systemctl start dockersudo systemctl start dockerFROM ghcr.io/dimvy-clothing-brand:latest

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
