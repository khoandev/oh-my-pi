import { describe, expect, it } from "bun:test";
import { NO_AUTH_SENTINEL } from "@oh-my-pi/pi-ai";
import { type AuthGatewayCredentialOptions, resolveGatewayApiKey } from "@oh-my-pi/pi-ai/auth-gateway/dispatch";
import { buildModel } from "@oh-my-pi/pi-catalog/build";

const options: AuthGatewayCredentialOptions = {
	storage: {
		keys: { getWithCredential: async () => undefined },
	} as unknown as AuthGatewayCredentialOptions["storage"],
};

function openCodeModel(id: string, inputCost: number) {
	return buildModel({
		id,
		name: id,
		api: "openai-completions",
		provider: "opencode-zen",
		baseUrl: "https://opencode.ai/zen/v1",
		reasoning: false,
		input: ["text"],
		cost: { input: inputCost, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 1000,
		maxTokens: 100,
	});
}

describe("auth gateway keyless providers", () => {
	it("resolves the free OpenCode lane to the no-auth sentinel", async () => {
		const result = await resolveGatewayApiKey(
			options,
			openCodeModel("jev-1.13-free", 0),
			"session",
			new AbortController().signal,
			"peer",
		);

		expect(result).toEqual({ apiKey: NO_AUTH_SENTINEL });
	});

	it("rejects a paid OpenCode lane without a credential", async () => {
		const result = await resolveGatewayApiKey(
			options,
			openCodeModel("jev-1.14", 0.042),
			"session",
			new AbortController().signal,
			"peer",
		);

		expect(result).toMatchObject({ status: 401, type: "authentication_error" });
	});
});
