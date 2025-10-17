# Frontend — csv-agent-frontend

This repository contains a small React frontend located in the `frontend/csv-agent-frontend` folder. This document summarizes what the frontend is, its tech stack, important files, and a recommended developer workflow for running and building the app.

## What this frontend does

- Provides a simple UI for uploading CSV files and viewing results returned by backend/lambda services.
- The main user-facing component is `UploadandResults.jsx` which handles file selection, upload (to a presigned URL), and displaying results returned from the server.

## Tech stack

- React (Create React App style project)
- JavaScript (no TypeScript)
- Tailwind CSS for styling (configured via `tailwind.config.js`)
- PostCSS (configured via `postcss.config.js`)
- Standard CRA dev tooling: Webpack, react-scripts (installed via `package.json`)
