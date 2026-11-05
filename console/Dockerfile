# The phone page and the supervisor console, built and served by nginx on port 80, which also
# forwards /ws and /api to the backend: the browser talks to one origin.
# Base images pinned by digest (the tag kept for reading): a re-pushed tag cannot change a build.
FROM node:22-slim@sha256:4b150b48b4116e2a22f2c39cc27025fbf4f153e2078fd3b14d159fdb5c7279da AS build
WORKDIR /console
COPY console/package.json console/package-lock.json ./
RUN npm ci
COPY console ./
RUN npm run build

FROM nginx:1.29-alpine@sha256:5616878291a2eed594aee8db4dade5878cf7edcb475e59193904b198d9b830de
COPY console/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /console/dist /usr/share/nginx/html
EXPOSE 80
