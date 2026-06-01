# 🩺 Skin Disease Detection Web App — Backend & Database

> Node.js + FastAPI backend with MongoDB integration for the AI-powered skin disease detection platform. Built as a Final Year Project (FYP) and funded by the **Ignite National Technology Fund**.

---

## 🔗 Related Repository

This repository contains the **Node.js backend**, **MongoDB database connection**, and **FastAPI ML service**.  
The React.js frontend is in a separate repo:  
👉 [Skin-Disease-Detection-Website-1](https://github.com/Areesha-Fatima/Skin-Disease-Detection-Website-1)

---

## ✨ Features

- 🔐 User authentication with JWT middleware
- 🗄️ MongoDB database integration for user data & history
- 🤖 FastAPI ML service for CNN-based skin disease classification
- 🔁 RESTful API routes for frontend communication
- 🚀 Vercel-ready deployment configuration

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend Server | Node.js, Express.js |
| ML Service | FastAPI, Python |
| Database | MongoDB |
| Authentication | JWT (JSON Web Tokens) |
| Deployment | Vercel |
| ML Model | TensorFlow, OpenCV, CNN |

---

## 📁 Project Structure

```
Skin-Disease-Detection-Website-2/
├── app.js                  # Express app entry point
├── main.py                 # FastAPI ML inference service
├── connectiondb/           # MongoDB connection setup
├── controllers/            # Route controller logic
├── middlewares/            # Auth & validation middleware
├── models/                 # MongoDB data models/schemas
├── routes/                 # API route definitions
├── vercel.json             # Vercel deployment config
└── package.json            # Node.js dependencies
```

---

## 🚀 Getting Started

### Prerequisites
- Node.js (v16+)
- Python 3.8+
- MongoDB (local or MongoDB Atlas)

### 1. Clone the repository
```bash
git clone https://github.com/Areesha-Fatima/Skin-Disease-Detection-Website-2.git
cd Skin-Disease-Detection-Website-2
```

### 2. Install Node.js dependencies
```bash
npm install
```

### 3. Set up environment variables
Create a `.env` file in the root directory:
```env
MONGO_URI=your_mongodb_connection_string
JWT_SECRET=your_jwt_secret_key
PORT=5000
```

### 4. Install Python dependencies
```bash
pip install fastapi uvicorn tensorflow opencv-python
```

### 5. Run the FastAPI ML service
```bash
uvicorn main:app --reload --port 8000
```

### 6. Run the Node.js backend
```bash
npm start
```

> Make sure the frontend (Repo 1) is also running for the full application.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/auth/signup` | Register a new user |
| POST | `/api/auth/login` | Login and receive JWT |
| POST | `/api/detect` | Submit image for skin disease detection |
| GET | `/api/history` | Get user's past detection results |

---

## 🔄 Architecture Overview

```
React Frontend (Repo 1)
        ↓
Node.js / Express API  ←→  MongoDB
        ↓
FastAPI ML Service
        ↓
CNN Model (TensorFlow + OpenCV)
        ↓
Classification Result
```

---

## ☁️ Deployment

This project is configured for **Vercel** deployment via `vercel.json`.  
The FastAPI ML service can be deployed separately on **Render** or **Railway**.

---

## 🏆 Recognition

- 🎓 Final Year Project (FYP) — Iqra University
- 💰 **Funded by Ignite National Technology Fund (Pakistan)**

---

## 👩‍💻 Author

**Areesha Fatima**  
Web Developer | ML-Integrated Applications  
[LinkedIn](https://www.linkedin.com/in/areesha-fatima-718659299/) · [GitHub](https://github.com/Areesha-Fatima) · [Portfolio](https://areeshafatima-portfolio.netlify.app)
