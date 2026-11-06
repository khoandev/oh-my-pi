import { describe, expect, test } from "bun:test";
import { buildModel } from "../src/build";
import type { ModelSpec } from "../src/types";
import { requiresOpenCodeEnvironment } from "../src/wire/opencode";

/**
 * The free-tier gate is a provider deployment contract, so the decision has to
 * survive a stale catalog: `models.json` is a frozen census and a discovery
 * cache can hold a row baked by an older build, neither of which carries the
 * flag the current KDL rule tree resolves. A row that silently answers `false`
 * ships no `x-opencode-client`, no `<env>` block and no gate tool names, which
 * the gateway answers with `429 FreeUsageLimitError`.
 */
function zenSpec(
	id: string,
	overrides: Partial<ModelSpec<"openai-completions">> = {},
): ModelSpec<"openai-completions"> {
	return {
		id,
		name: id,
		api: "openai-completions",
		provider: "opencode-zen",
		baseUrl: "https://opencode.ai/zen/v1",
		reasoning: true,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 1_000_000,
		maxTokens: 384_000,
		...overrides,
	};
}

describe("requiresOpenCodeEnvironment", () => {
	test("resolves the gate for a free model whose baked compat predates the rule", () => {
		const built = buildModel(zenSpec("deepseek-v4.1-flash-free"));
		// A row baked by an older build: the axis is absent from its compat.
		const stale = { ...built, compat: { ...built.compat, requiresOpenCodeEnvironment: undefined } };

		expect(requiresOpenCodeEnvironment(stale)).toBe(true);
	});

	test("leaves paid models on the same host ungated", () => {
		const built = buildModel(
			zenSpec("deepseek-v4.1-flash", { cost: { input: 0.3, output: 1.2, cacheRead: 0, cacheWrite: 0 } }),
		);
		const stale = { ...built, compat: { ...built.compat, requiresOpenCodeEnvironment: undefined } };

		expect(requiresOpenCodeEnvironment(stale)).toBe(false);
	});

	test("honors an explicit baked value over the rule tree", () => {
		const built = buildModel(zenSpec("deepseek-v4.1-flash-free"));

		expect(
			requiresOpenCodeEnvironment({ ...built, compat: { ...built.compat, requiresOpenCodeEnvironment: false } }),
		).toBe(false);
	});

	test("gates a custom provider declared as the OpenCode backend", () => {
		const custom = buildModel(
			zenSpec("deepseek-v4.1-flash-free", { provider: "my-zen-proxy", providerType: "opencode-zen" }),
		);
		const stale = { ...custom, compat: { ...custom.compat, requiresOpenCodeEnvironment: undefined } };

		expect(requiresOpenCodeEnvironment(custom)).toBe(true);
		expect(requiresOpenCodeEnvironment(stale)).toBe(true);
	});

	test("never gates a host the OpenCode gateways do not serve", () => {
		const built = buildModel(zenSpec("deepseek-v4.1-flash-free", { provider: "openai", baseUrl: undefined }));

		expect(requiresOpenCodeEnvironment(built)).toBe(false);
	});
});
