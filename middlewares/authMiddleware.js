const jwt = require("jsonwebtoken");

module.exports = (req, res, next) => {
  const authHeader = req.header("Authorization");

  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    return res.status(401).json({ msg: "Authorization token missing or malformed" });
  }

  const token = authHeader.split(" ")[1];

  if (!token) {
    return res.status(401).json({ msg: "Authorization denied, token not found" });
  }

  try {
 
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    
    req.user = decoded.userId;

    next();
  } catch (err) {
    console.error("Token verification failed:", err.message);
    
    if (err.name === "TokenExpiredError") {
      return res.status(401).json({ msg: "Token has expired, please login again" });
    }
    
    return res.status(401).json({ msg: "Invalid token, authorization denied" });
  }
};
