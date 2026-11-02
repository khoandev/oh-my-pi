import * as path from "node:path";
import { getTinyModelsCacheDir, onnxRuntimeInstallOverrides } from "@oh-my-pi/pi-utils";

export const KOKORO_PACKAGE = "kokoro-js";
export const KOKORO_VERSION = "1.2.1";
export const ONNXRUNTIME_NODE_PACKAGE = "onnxruntime-node";
export const ONNXRUNTIME_NODE_VERSION = "1.26.0";

/**
 * `onnxruntime-node` release the TTS side runtime pins: the host's
 * prebuilt-capable release (darwin/x64, where 1.24.0+ ships no native binding),
 * else {@link ONNXRUNTIME_NODE_VERSION}.
 */
export function ttsOnnxRuntimeVersion(): string {
	return onnxRuntimeInstallOverrides()?.[ONNXRUNTIME_NODE_PACKAGE] ?? ONNXRUNTIME_NODE_VERSION;
}

export function getTtsRuntimeDir(): string {
	const runtimeKey = KOKORO_VERSION.replace(/[^A-Za-z0-9._-]/g, "_");
	// Only hosts that pin a different ONNX Runtime get their own cache directory,
	// so the stock install keeps the directory name it already populated.
	const pin = onnxRuntimeInstallOverrides()?.[ONNXRUNTIME_NODE_PACKAGE];
	const ort = pin === undefined ? "" : `_ort-${pin.replace(/[^A-Za-z0-9._-]/g, "_")}`;
	return path.join(path.dirname(getTinyModelsCacheDir()), "tts-runtime", `kokoro-${runtimeKey}${ort}`);
}

export async function isTtsRuntimeCached(): Promise<boolean> {
	try {
		const pkg = await Bun.file(path.join(getTtsRuntimeDir(), "node_modules", KOKORO_PACKAGE, "package.json")).json();
		return typeof pkg === "object" && pkg !== null && "version" in pkg && pkg.version === KOKORO_VERSION;
	} catch {
		return false;
	}
}
