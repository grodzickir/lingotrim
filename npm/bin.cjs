#!/usr/bin/env node
// Runs the Python `lingotrim` CLI through uv (which installs Python itself if needed).
const { spawnSync } = require("node:child_process");
const path = require("node:path");

const uvLauncher = path.join(path.dirname(require.resolve("@manzt/uv/package.json")), "bin.cjs");
const version = require("./package.json").version;
const spec = process.env.LINGOTRIM_SPEC || `lingotrim[corpus,gguf]==${version}`;

const result = spawnSync(
  process.execPath,
  [uvLauncher, "tool", "run", "--python", ">=3.10", "--from", spec, "lingotrim", ...process.argv.slice(2)],
  { stdio: "inherit" },
);
if (result.error) throw result.error;
process.exitCode = result.status ?? 1;
