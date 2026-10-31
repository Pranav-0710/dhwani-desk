# The phone page and the supervisor console, built and served by nginx on port 80, which also
# forwards /ws and /api to the backend: the browser talks to one origin.
FROM node:22-slim AS build
WORKDIR /console
COPY console/package.json console/package-lock.json ./
RUN npm ci
COPY console ./
RUN npm run build

FROM nginx:1.29-alpine
COPY console/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /console/dist /usr/share/nginx/html
EXPOSE 80
