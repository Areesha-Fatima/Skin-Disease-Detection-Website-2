const dotenv = require("dotenv");
dotenv.config();
const express = require("express");
const connectDB = require("./connectiondb/db");
const authRoutes = require("./routes/authRoutes");
const cors = require('cors');
const bodyParser = require('body-parser')
// Uncomment this line if using the body-parser package
// const bodyParser = require('body-parser'); 

const app = express();

// Connect to the database
connectDB();

// CORS options
const corsOptions = {
  origin: 'http://localhost:3000', 
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  credentials: true, 
};

app.use(cors(corsOptions));

// Body parser middleware
app.use(express.json()); // Parse JSON bodies
// Uncomment this line if using the body-parser package
// app.use(bodyParser.json()); 

app.use(express.urlencoded({ extended: true })); 
// Uncomment this line if using the body-parser package
// app.use(bodyParser.urlencoded({ extended: true })); 

app.use("/api", authRoutes);

app.get("/hello", (req, res) => {
    res.send("Hello, world!");
});

// Start the server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));

