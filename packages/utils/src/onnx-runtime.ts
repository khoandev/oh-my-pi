import * as fs from "node:fs";
import * as path from "node:path";

/**
 * ONNX Runtime native-addon support for hosts whose prebuilt binaries the npm
 * release no longer ships.
 *
 * `onnxruntime-node` bundles a prebuilt `onnxruntime_binding.node` (plus the
 * runtime library it links) under `bin/napi-v<N>/<platform>/<arch>/` for every
 * platform it supports. macOS x64 support ended after 1.23.0:
 *
 * - 1.24.0+ ships `bin/napi-v6/darwin/arm64` only, so `require("onnxruntime-node")`
 *   fails with `Cannot find module '../bin/napi-v6/darwin/x64/onnxruntime_binding.node'`.
 * - 1.22.0 and 1.23.0 still ship the x64 dylib, but it references
 *   `__ZNSt3__18to_charsEPcS0_d`, absent from the libc++ of macOS 12 and older, so
 *   `dlopen` fails with `Symbol not found` there.
 * - 1.21.0 loads on macOS 12 Intel (verified on 12.7.6).
 *
 * Fastembed already pins 1.21.0 exactly, so the embeddings path is unaffected;
 * every other consumer (Transformers.js, kokoro-js) floats a newer release and
 * fails on an Intel Mac, which is why side runtimes pin
 * {@link ONNX_RUNTIME_DARWIN_X64_SPEC} instead.
 */

/** Newest `onnxruntime-node` release whose prebuilt macOS x64 binding loads on macOS 12. */
export const ONNX_RUNTIME_DARWIN_X64_SPEC = "1.21.0";

/**
 * `bun install` overrides pinning `onnxruntime-node` to a release with a
 * prebuilt binding for this host, or `undefined` when the stock resolution
 * already ships one.
 */
export function onnxRuntimeInstallOverrides(
	platform = process.platform,
	arch = process.arch,
): Record<string, string> | undefined {
	if (platform !== "darwin" || arch !== "x64") return undefined;
	return { "onnxruntime-node": ONNX_RUNTIME_DARWIN_X64_SPEC };
}

/**
 * Whether the ONNX Runtime this host can load supports block-quantized weights
 * (`GatherBlockQuantized.bits`). False only on darwin/x64, whose pin predates it.
 */
export function onnxRuntimeSupportsBlockQuantization(platform = process.platform, arch = process.arch): boolean {
	return onnxRuntimeInstallOverrides(platform, arch) === undefined;
}

/**
 * Path of the prebuilt native binding an installed `onnxruntime-node` package
 * carries for `platform`/`arch`, or `null` when the package has none — the
 * condition that makes `require("onnxruntime-node")` fail at load.
 */
export function onnxRuntimeBindingPath(
	packageDir: string,
	platform = process.platform,
	arch = process.arch,
): string | null {
	const binDir = path.join(packageDir, "bin");
	let abiDirs: string[];
	try {
		abiDirs = fs.readdirSync(binDir);
	} catch {
		return null;
	}
	for (const abiDir of abiDirs) {
		const binding = path.join(binDir, abiDir, platform, arch, "onnxruntime_binding.node");
		if (fs.existsSync(binding)) return binding;
	}
	return null;
}
