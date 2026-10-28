const http = require("node:http");
const { spawn } = require("node:child_process");
const readline = require("node:readline");
const path = require("node:path");
const fs = require("node:fs");

const FEATURE_NAMES = [
  "return_yesterday",
  "return_2_days_ago",
  "return_3_days_ago",
  "return_5_days_ago",
  "return_10_days_ago",
  "average_5_day_return",
  "volatility_10_days",
  "relative_volume",
];
const MAX_BODY_BYTES = 16 * 1024;
const webRoot = __dirname;

class ModelWorker {
  constructor() {
    this.child = null;
    this.nextId = 1;
    this.pending = new Map();
    this.ready = false;
    this.readySettled = false;
    this.stopping = false;
    this.readyPromise = new Promise((resolve, reject) => {
      this.resolveReady = resolve;
      this.rejectReady = reject;
    });
  }

  start() {
    const python = process.env.PYTHON_EXECUTABLE ||
      (process.platform === "win32" ? "python" : "python3");
    this.child = spawn(python, [path.join(webRoot, "main.py"), "--worker"], {
      cwd: webRoot,
      stdio: ["pipe", "pipe", "pipe"],
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    });

    const lines = readline.createInterface({ input: this.child.stdout });
    lines.on("line", (line) => this.handleLine(line));
    this.child.stderr.on("data", (chunk) => process.stderr.write(chunk));
    this.child.on("error", (error) => this.fail(error));
    this.child.on("exit", (code, signal) => {
      if (!this.stopping) {
        this.fail(
          new Error(`Python model worker exited (code ${code}, signal ${signal}).`),
        );
      }
    });
    this.child.stdin.on("error", (error) => this.fail(error));
    return this.readyPromise;
  }

  handleLine(line) {
    let message;
    try {
      message = JSON.parse(line);
    } catch (error) {
      this.fail(new Error(`Invalid response from model worker: ${error.message}`));
      return;
    }

    if (message.ready === true) {
      this.ready = true;
      this.readySettled = true;
      this.resolveReady();
      return;
    }

    const pending = this.pending.get(message.id);
    if (!pending) {
      process.stderr.write("Received an unmatched response from the model worker.\n");
      return;
    }

    this.pending.delete(message.id);
    if (message.error) {
      pending.reject(new Error(message.error));
    } else {
      pending.resolve(message.result);
    }
  }

  fail(error) {
    this.ready = false;
    if (!this.readySettled) {
      this.readySettled = true;
      this.rejectReady(error);
    }
    for (const pending of this.pending.values()) {
      pending.reject(error);
    }
    this.pending.clear();
  }

  async predict(inputs) {
    await this.readyPromise;
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.child.stdin.write(`${JSON.stringify({ id, inputs })}\n`, (error) => {
        if (error) {
          this.pending.delete(id);
          reject(error);
        }
      });
    });
  }

  stop() {
    this.stopping = true;
    if (this.child && !this.child.killed) {
      this.child.kill();
    }
  }
}

class HttpError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

function sendJson(response, status, value) {
  response.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "cache-control": "no-store",
    "x-content-type-options": "nosniff",
  });
  response.end(JSON.stringify(value));
}

function readJson(request) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    let tooLarge = false;

    request.on("data", (chunk) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        tooLarge = true;
        chunks.length = 0;
      } else if (!tooLarge) {
        chunks.push(chunk);
      }
    });
    request.on("end", () => {
      if (tooLarge) {
        reject(new HttpError(413, "Request body is too large."));
        return;
      }
      try {
        resolve(JSON.parse(Buffer.concat(chunks).toString("utf8")));
      } catch {
        reject(new HttpError(400, "Request body must be valid JSON."));
      }
    });
    request.on("error", reject);
  });
}

function validateFeatures(body) {
  if (!body || !Array.isArray(body.features) ||
      body.features.length !== FEATURE_NAMES.length) {
    throw new HttpError(
      400,
      `Provide exactly ${FEATURE_NAMES.length} numeric features in the documented order.`,
    );
  }

  const { features } = body;
  if (!features.every((value) => typeof value === "number" && Number.isFinite(value))) {
    throw new HttpError(400, "Every feature must be a finite number.");
  }
  if (features.slice(0, 6).some((value) => Math.abs(value) > 5) ||
      features[6] < 0 || features[6] > 5 ||
      features[7] < 0 || features[7] > 1000) {
    throw new HttpError(
      400,
      "Returns and volatility must be between -5 and 5, volatility cannot be negative, and relative volume must be between 0 and 1000.",
    );
  }
  return features;
}

function servePage(response) {
  const pagePath = path.join(webRoot, "index.html");
  let page;
  try {
    page = fs.readFileSync(pagePath);
  } catch (error) {
    process.stderr.write(`Unable to read the web page: ${error.message}\n`);
    sendJson(response, 500, { error: "The web page is unavailable." });
    return;
  }

  response.writeHead(200, {
    "content-type": "text/html; charset=utf-8",
    "cache-control": "no-store",
    "x-content-type-options": "nosniff",
  });
  response.end(page);
}

async function handleRequest(request, response, worker) {
  try {
    const requestUrl = new URL(request.url, "http://localhost");
    if (request.method === "GET" && ["/", "/index.html"].includes(requestUrl.pathname)) {
      servePage(response);
      return;
    }
    if (request.method === "GET" && requestUrl.pathname === "/api/health") {
      sendJson(response, 200, { status: "ok", modelReady: worker.ready });
      return;
    }
    if (request.method === "POST" && requestUrl.pathname === "/api/predict") {
      if (!request.headers["content-type"]?.includes("application/json")) {
        throw new HttpError(415, "Content-Type must be application/json.");
      }
      const body = await readJson(request);
      const features = validateFeatures(body);
      const result = await worker.predict(features);
      sendJson(response, 200, {
        ...result,
        featureNames: FEATURE_NAMES,
      });
      return;
    }
    sendJson(response, 404, { error: "Not found." });
  } catch (error) {
    if (error instanceof HttpError) {
      sendJson(response, error.status, { error: error.message });
      return;
    }
    process.stderr.write(`Request failed: ${error.stack || error.message}\n`);
    sendJson(response, 500, { error: "Prediction service is temporarily unavailable." });
  }
}

async function startServer() {
  const port = Number(process.env.PORT || 3000);
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error("PORT must be an integer between 1 and 65535.");
  }

  const worker = new ModelWorker();
  await worker.start();
  const server = http.createServer((request, response) => {
    void handleRequest(request, response, worker);
  });
  const host = process.env.HOST || "::";
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, host, resolve);
  });
  process.stdout.write(`Stock model site listening on ${host}:${port}\n`);

  const shutdown = () => {
    server.close(() => worker.stop());
  };
  process.once("SIGINT", shutdown);
  process.once("SIGTERM", shutdown);
}

startServer().catch((error) => {
  process.stderr.write(`Unable to start stock model site: ${error.stack || error.message}\n`);
  process.exitCode = 1;
});
