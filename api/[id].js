const fs = require("fs");
const path = require("path");

module.exports = (req, res) => {
  const { id } = req.query;

  try {
    const filePath = path.join(process.cwd(), "programs", `${id}.py`);
    const content = fs.readFileSync(filePath, "utf8");

    res.setHeader("Content-Type", "text/plain");
    res.status(200).send(content);
  } catch (e) {
    res.status(404).send("Program not found");
  }
};