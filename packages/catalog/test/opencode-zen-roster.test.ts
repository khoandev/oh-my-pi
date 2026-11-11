import { describe, expect, test } from "bun:test";
import { isIncludedModel } from "../src/compat/behavior";

/**
 * OpenCode Zen's roster allow-list (`include-models` in
 * `runtime/behavior.kdl`) keeps the picker on what the contributor credential
 * can actually run. Paid SKUs answer `401`/`403` for `Bearer public`, and ids
 * the gateway retired leave with the live roster, so listing them only offers
 * a model that cannot answer.
 */
describe("OpenCode Zen roster allow-list", () => {
	test("lists the free tier and big-pickle", () => {
		expect(isIncludedModel("opencode-zen", "deepseek-v4.1-flash-free")).toBe(true);
		expect(isIncludedModel("opencode-zen", "big-pickle")).toBe(true);
		expect(isIncludedModel("opencode-zen", "muse-spark-1.3-contributor-free")).toBe(true);
	});

	test("hides paid SKUs the free-tier credential cannot call", () => {
		expect(isIncludedModel("opencode-zen", "deepseek-v4.1-flash")).toBe(false);
		expect(isIncludedModel("opencode-zen", "claude-opus-5")).toBe(false);
		expect(isIncludedModel("opencode-zen", "minimax-m3")).toBe(false);
	});

	test("leaves the sibling gateway's roster untouched", () => {
		expect(isIncludedModel("opencode-go", "deepseek-v4-flash")).toBe(true);
	});
});
