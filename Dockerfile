FROM node:22-bookworm-slim

ENV NODE_ENV=production
ENV PORT=3000

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY package.json server.js main.py nvidiadata.py index.html ./

EXPOSE 3000
CMD ["npm", "start"]
