import { describe, expect, it } from "bun:test";
import {
	normalizeTinyModelDtype,
	resolveLocalModelDtype,
	resolveTinyModelDtypeOverride,
	TINY_MODEL_DTYPE_DEFAULT,
	TINY_MODEL_DTYPE_SETTING_VALUES,
	tinyModelDtypeSettingToEnv,
} from "@oh-my-pi/pi-coding-agent/tiny/dtype";
import {
	getTinyLocalModelSpec,
	LEGACY_ONNX_TEXT_REASON,
	planTinyTextOnnxLoad,
} from "@oh-my-pi/pi-coding-agent/tiny/models";

describe("tiny model dtype selection", () => {
	it("returns undefined when unset so callers keep the per-model spec dtype", () => {
		expect(resolveTinyModelDtypeOverride(undefined)).toBeUndefined();
		expect(resolveTinyModelDtypeOverride("")).toBeUndefined();
		expect(resolveTinyModelDtypeOverride("   ")).toBeUndefined();
	});

	it("canonicalizes a valid precision regardless of case/whitespace", () => {
		expect(resolveTinyModelDtypeOverride("  FP16 ")).toBe("fp16");
		expect(resolveTinyModelDtypeOverride("q4f16")).toBe("q4f16");
		expect(normalizeTinyModelDtype("Q8")).toBe("q8");
	});

	it("rejects an unsupported precision", () => {
		expect(() => resolveTinyModelDtypeOverride("int4")).toThrow("Unsupported PI_TINY_DTYPE");
	});
});

describe("host dtype narrowing", () => {
	it("keeps the requested dtype where the runtime loads block-quantized weights", () => {
		expect(resolveLocalModelDtype(undefined, "q4", true)).toBe("q4");
		expect(resolveLocalModelDtype("q4", "q8", true)).toBe("q4");
	});

	it("replaces a block-quantized dtype where the runtime cannot load it", () => {
		expect(resolveLocalModelDtype(undefined, "q4", false)).toBe("fp16");
		expect(resolveLocalModelDtype("bnb4", "q4", false)).toBe("fp16");
	});

	it("leaves dtypes the older runtime still loads untouched", () => {
		expect(resolveLocalModelDtype(undefined, "q8", false)).toBe("q8");
		expect(resolveLocalModelDtype("fp32", "q4", false)).toBe("fp32");
	});
});

describe("tiny text ONNX load planning", () => {
	const spec = getTinyLocalModelSpec("lfm2.5-230m");
	if (!spec) throw new Error("lfm2.5-230m missing from the registry");

	it("loads the shipped q4 export where the runtime understands block-quantized weights", () => {
		expect(planTinyTextOnnxLoad(spec, undefined, true)).toEqual({ dtype: "q4" });
	});

	it("reports the host as unsupported instead of downloading weights that cannot load", () => {
		const plan = planTinyTextOnnxLoad(spec, undefined, false);
		expect(plan.unsupportedReason).toBe(LEGACY_ONNX_TEXT_REASON);
		// The reason must name both blockers; a vaguer message sends users to retry.
		expect(plan.unsupportedReason).toContain("GatherBlockQuantized");
		expect(plan.unsupportedReason).toContain("GroupQueryAttention");
	});

	it("refuses an explicit dtype override too: downcasting does not rescue that host", () => {
		// Verified against the pinned runtime: the fp32 graph fails on the same
		// GroupQueryAttention node, and the abort takes the worker process down.
		expect(planTinyTextOnnxLoad(spec, "fp32", false).unsupportedReason).toBe(LEGACY_ONNX_TEXT_REASON);
	});

	it("keeps a per-model refusal on every host", () => {
		const blocked = getTinyLocalModelSpec("qwen3-1.7b");
		if (!blocked) throw new Error("qwen3-1.7b missing from the registry");
		const reason = planTinyTextOnnxLoad(blocked, undefined, true).unsupportedReason;
		expect(reason).toContain("RotaryEmbedding");
	});
});

describe("tiny model dtype setting → PI_TINY_DTYPE mapping", () => {
	it("returns undefined for the default sentinel so the worker keeps each model's spec dtype", () => {
		expect(tinyModelDtypeSettingToEnv(TINY_MODEL_DTYPE_DEFAULT)).toBeUndefined();
		expect(tinyModelDtypeSettingToEnv(undefined)).toBeUndefined();
		expect(tinyModelDtypeSettingToEnv("")).toBeUndefined();
	});

	it("keeps every non-default setting value resolvable by the worker", () => {
		for (const value of TINY_MODEL_DTYPE_SETTING_VALUES) {
			if (value === TINY_MODEL_DTYPE_DEFAULT) continue;
			expect(normalizeTinyModelDtype(tinyModelDtypeSettingToEnv(value))).toBe(value);
		}
	});
});
