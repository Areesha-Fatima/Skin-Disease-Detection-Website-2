const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const User = require("../models/userModel");

const JWT_SECRET = process.env.JWT_SECRET;
const JWT_EXPIRY = "10m";

const USER_EXISTS_ERROR = "User already exists. Please login or use a different email.";
const INVALID_CREDENTIALS_ERROR = "Invalid credentials. Please check your email and password.";
const SERVER_ERROR_MESSAGE = "Server error. Please try again later.";
const USER_REGISTERED_SUCCESS = "User registered successfully";
const LOGIN_SUCCESS_MESSAGE = "Login successful";

exports.signup = async (req, res) => {
  const { username, email, password } = req.body;

  try {
    const existingUser = await User.findOne({ email });

    if (existingUser) {
      return res.status(400).json({
        success: false,
        error: USER_EXISTS_ERROR,
      });
    }

    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(password, salt);

    const newUser = new User({
      username,
      email,
      password: hashedPassword,
    });

    await newUser.save();

    const payload = { userId: newUser._id };
    const token = jwt.sign(payload, JWT_SECRET, { expiresIn: JWT_EXPIRY });

    res.status(201).json({
      success: true,
      message: USER_REGISTERED_SUCCESS,
      token,
    });
  } catch (error) {
    console.error("Error during signup: ", error.message);
    res.status(500).json({
      success: false,
      error: SERVER_ERROR_MESSAGE,
    });
  }
};

exports.login = async (req, res) => {
  const { email, password } = req.body;

  try {
    const user = await User.findOne({ email });
    if (!user) {
      return res.status(400).json({
        success: false,
        error: INVALID_CREDENTIALS_ERROR,
      });
    }

    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) {
      return res.status(400).json({
        success: false,
        error: INVALID_CREDENTIALS_ERROR,
      });
    }

    const payload = { userId: user._id };
    const token = jwt.sign(payload, JWT_SECRET, { expiresIn: JWT_EXPIRY });

    res.status(200).json({
      success: true,
      message: LOGIN_SUCCESS_MESSAGE,
      token,
    });
  } catch (error) {
    console.error("Error during login: ", error.message);
    res.status(500).json({
      success: false,
      error: SERVER_ERROR_MESSAGE,
    });
  }
};

exports.getUsers = async (req, res) => {
  try {
    const users = await User.find();

    if (!users || users.length === 0) {
      return res.status(404).json({
        success: false,
        message: "No users found",
      });
    }

    return res.status(200).json({
      success: true,
      message: "Users fetched successfully",
      data: users,
    });
  } catch (error) {
    console.error("Error fetching users:", error);

    return res.status(500).json({
      success: false,
      message: SERVER_ERROR_MESSAGE,
      error: error.message,
    });
  }
};
