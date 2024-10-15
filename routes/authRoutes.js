const express = require("express");
const router = express.Router();
const authController = require("../controllers/authController");
const authMiddleware = require("../middlewares/authMiddleware");

router.post("/signup", authController.signup);
router.post("/login", authController.login);
router.get("/getUsers",authMiddleware,authController.getUsers)
router.get("/protected", authMiddleware, (req, res) => {
  res.send("Access granted to protected route");
});

module.exports = router;
