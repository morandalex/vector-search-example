# Dockerfile
# Base image
FROM node:18


# Set working directory
WORKDIR /app

# Copy package.json and yarn.lock
COPY package.json yarn.lock ./

# Install dependencies
RUN yarn install

# Copy all files
COPY ./ ./

# Build the application
#RUN yarn build

# Run the application
CMD [ "yarn", "dev" ]