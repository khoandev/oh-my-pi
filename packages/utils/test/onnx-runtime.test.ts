import { afterEach, describe, expect, test } from "bun:test";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import {
	ONNX_RUNTIME_DARWIN_X64_SPEC,
	onnxRuntimeBindingPath,
	onnxRuntimeInstallOverrides,
	onnxRuntimeSupportsBlockQuantization,
} from "../src/onnx-runtime";

// Contract under test: onnxruntime-node 1.24.0+ ships no prebuilt macOS x64
// binding, so `require("onnxruntime-node")` fails on Intel Macs with
// `Cannot find module '../bin/napi-v6/darwin/x64/onnxruntime_binding.node'`.
// Side runtimes install a pinned release instead, and ambient installs are
// usable only when the binding for the host actually exists.

const tempDirs: string[] = [];

afterEach(async () => {
	await Promise.all(tempDirs.splice(0).map(dir => fs.rm(dir, { recursive: true, force: true })));
});

async function fixtureDir(): Promise<string> {
	const dir = await fs.mkdtemp(path.join(os.tmpdir(), "omp-onnx-runtime-"));
	tempDirs.push(dir);
	return dir;
}

describe("onnxRuntimeInstallOverrides", () => {
	test("pins a prebuilt-capable release for darwin/x64", () => {
		expect(onnxRuntimeInstallOverrides("darwin", "x64")).toEqual({
			"onnxruntime-node": ONNX_RUNTIME_DARWIN_X64_SPEC,
		});
	});

	test("leaves hosts upstream ships prebuilts for on stock resolution", () => {
		expect(onnxRuntimeInstallOverrides("darwin", "arm64")).toBeUndefined();
		expect(onnxRuntimeInstallOverrides("linux", "x64")).toBeUndefined();
		expect(onnxRuntimeInstallOverrides("win32", "arm64")).toBeUndefined();
	});

	test("reports darwin/x64 as the only host without block-quantized weight support", () => {
		expect(onnxRuntimeSupportsBlockQuantization("darwin", "x64")).toBe(false);
		expect(onnxRuntimeSupportsBlockQuantization("darwin", "arm64")).toBe(true);
		expect(onnxRuntimeSupportsBlockQuantization("linux", "x64")).toBe(true);
	});
});

describe("onnxRuntimeBindingPath", () => {
	test("finds the binding per ABI directory and reports a missing platform/arch", async () => {
		const packageDir = await fixtureDir();
		const binding = path.join(packageDir, "bin", "napi-v6", "linux", "x64", "onnxruntime_binding.node");
		await fs.mkdir(path.dirname(binding), { recursive: true });
		await Bun.write(binding, "");

		expect(onnxRuntimeBindingPath(packageDir, "linux", "x64")).toBe(binding);
		// The 1.24.0+ darwin layout: `bin/napi-v6/darwin/arm64` exists, x64 does not.
		expect(onnxRuntimeBindingPath(packageDir, "darwin", "x64")).toBeNull();
	});

	test("reports a package without a bin directory", async () => {
		const packageDir = await fixtureDir();
		expect(onnxRuntimeBindingPath(packageDir, "linux", "x64")).toBeNull();
	});
});
