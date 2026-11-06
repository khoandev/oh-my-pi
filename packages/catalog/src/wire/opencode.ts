/**
 * OpenCode gateway (Zen, Go) wire identity.
 *
 * The contributor free tier is gated on client identity: every request must
 * carry a `User-Agent` whose leading token is `opencode/<version>` and an
 * `x-opencode-session` matching `ses_<12 hex><14 alnum>`. Both are wire-shape
 * checks applied before credentials are evaluated
 * ([#12306](https://github.com/can1357/oh-my-pi/issues/12306)).
 */

import { resolveModelPolicy } from "../compat/resolve";
import type { Api, CatalogPolicyInputs, ModelSpec } from "../types";

/** Canonical OpenCode client User-Agent; the gate reads only the leading token. */
export const OPENCODE_USER_AGENT = "opencode/1.19.3";

/**
 * What the free tier expects as a credential: the literal `public`, not an
 * account key — the gate runs before any credential is evaluated.
 */
export const OPENCODE_PUBLIC_KEY = "public";

/** Provider ids whose hosts sit behind the OpenCode gateways (Zen, Go). */
export const OPENCODE_PROVIDER_IDS: ReadonlySet<string> = new Set(["opencode-zen", "opencode-go"]);

/** Whether `provider` is served by an OpenCode gateway. */
export function isOpenCodeProvider(provider: string): boolean {
	return OPENCODE_PROVIDER_IDS.has(provider);
}

/**
 * Model facts the gate read needs: the policy-cascade inputs a model or a
 * discovery spec both carry, plus the compat surface a built row exposes.
 */
export interface OpenCodeEnvironmentInput extends CatalogPolicyInputs {
	/** Resolved compat of a built row (absent on authored specs). */
	compat?: object;
	/** Sparse authored overlay (`Model.compatConfig`) the rule engine folds in. */
	compatConfig?: unknown;
}

/** The single compat field the gate read needs, read off an erased compat record. */
type EnvironmentCompat = { requiresOpenCodeEnvironment?: boolean };

/** One resolution per distinct model; the rule walk is too costly per request. */
const ENVIRONMENT_CACHE_LIMIT = 8192;
const openCodeEnvironmentCache = new Map<string, boolean>();

/**
 * Whether a model's route carries the OpenCode free-tier gate contract
 * (`requires-opencode-environment` in the provider KDL), and therefore needs
 * the client identity, the `<env>` block and the gate tool names.
 *
 * Prefers the row's resolved `compat`, and otherwise asks the rule tree
 * directly: `models.json` is a frozen census, and a discovery cache can hold a
 * row baked by an older build, so a rule authored after the last `gen:models`
 * run reaches shipped binaries only through `compat/rules.json`. A row that
 * silently misses the flag ships neither `x-opencode-client` nor the `<env>`
 * block, which the gateway answers with `429 FreeUsageLimitError`
 * ([#17075](https://github.com/can1357/oh-my-pi/pull/17075)).
 *
 * Resolutions are memoized per model — the cascade is cheap once per process
 * and too costly per request.
 */
export function requiresOpenCodeEnvironment(model: OpenCodeEnvironmentInput): boolean {
	if (!isOpenCodeProvider(model.providerType ?? model.provider)) return false;
	const baked = (model.compat as EnvironmentCompat | undefined)?.requiresOpenCodeEnvironment;
	if (baked !== undefined) return baked;
	const key = `${model.provider}\u0000${model.providerType ?? ""}\u0000${model.api}\u0000${model.id}`;
	const cached = openCodeEnvironmentCache.get(key);
	if (cached !== undefined) return cached;
	// Hand the engine the sparse authored overlay, never the baked record it
	// produced (mirrors the bake↔runtime parity gate).
	const { compat: _bakedCompat, compatConfig, ...inputs } = model;
	const policy = resolveModelPolicy({ ...inputs, compat: compatConfig } as unknown as ModelSpec<Api>);
	const resolved = (policy.compat as EnvironmentCompat | undefined)?.requiresOpenCodeEnvironment === true;
	if (openCodeEnvironmentCache.size >= ENVIRONMENT_CACHE_LIMIT) openCodeEnvironmentCache.clear();
	openCodeEnvironmentCache.set(key, resolved);
	return resolved;
}

/**
 * The gate requires all five of these OpenCode core tool NAMES in `tools[]`
 * (schemas are ignored, extra non-OpenCode tools are fine, and the count is
 * over distinct names). Agent harnesses carrying a normal tool roster satisfy
 * this natively; tool-less and partial-roster calls (e.g. advisor with only
 * `advise`) must pad.
 */
export const OPENCODE_GATE_TOOL_NAMES = ["bash", "edit", "glob", "grep", "read"] as const;

/** Minimum distinct gate tool names the gateway requires in `tools[]`. */
export const OPENCODE_GATE_MIN_TOOL_NAMES = 5;

/**
 * Gate tool names missing from `names` (exact-match — the gate compares names
 * only). Returns empty when all five gate names are already present; otherwise
 * returns every gate name not yet in the roster so the caller can pad to the
 * full set.
 */
export function missingOpenCodeGateToolNames(names: Iterable<string>): string[] {
	const present = new Set(names);
	return OPENCODE_GATE_TOOL_NAMES.filter(name => !present.has(name));
}

/**
 * Pad `tools` with all missing gate-required OpenCode tool names so every call
 * (tool-less helpers, advisors carrying only `advise`, partial rosters) passes
 * the body check. Pads carry no parameters and an empty description — the gate
 * ignores schemas — but callers MUST keep them out of the agent loop's tool
 * registry: they are wire-shape filler, not executable work.
 *
 * Returns the input array unchanged (same identity) when all five gate names
 * are already present.
 */
export function withOpenCodeGateTools<T extends { name: string }>(tools: readonly T[] | undefined): T[] {
	const existing = tools ?? [];
	const missing = missingOpenCodeGateToolNames(existing.map(tool => tool.name));
	if (missing.length === 0) return existing as T[];
	// The stubs structurally differ from T only in fields the wire never reads.
	// `strict: false` keeps every transport's strict-mode normalization inert.
	const pads = missing.map(name => ({
		name,
		description: "",
		parameters: { type: "object" },
		strict: false,
	})) as unknown as T[];
	return [...existing, ...pads];
}

/** Shape the OpenCode gate requires for `x-opencode-session`. */
export const OPENCODE_SESSION_TOKEN_PATTERN = /^ses_[0-9a-f]{12}[0-9A-Za-z]{14}$/;

/** Base62 alphabet used by OpenCode's session tail encoding. */
const BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/**
 * Encode `n` (arbitrary-precision) as exactly `width` base-62 characters,
 * left-padded with `0` when shorter. Mirrors Go's `base62Fixed` in the
 * upstream `internal/identity/request.go`.
 */
function base62Fixed(n: bigint, width: number): string {
	const base = BigInt(BASE62.length);
	let result = "";
	let v = n < 0n ? -n : n;
	while (result.length < width) {
		result = BASE62[Number(v % base)]! + result;
		v /= base;
	}
	return result.slice(-width);
}

/** Per-process atomic session counter (12-bit, mirrors Go's `sessionCounter`). */
let sessionCounter = 0;

/**
 * Mint the 12-hex timestamp prefix Zen decodes as `~(unix_ms*0x1000+counter)`
 * truncated to its low 48 bits. Digest-derived prefixes decode to a
 * non-monotonic value that Zen classifies as a non-OpenCode client and
 * rejects with 403 FreeTierError (issue #68).
 *
 * Mirrors Go's `mintedTimePrefix()` in `internal/identity/request.go`.
 */
function mintedTimePrefix(): string {
	const counter = BigInt(++sessionCounter & 0xfff);
	const ms = BigInt(Date.now());
	const raw = ~(ms * 0x1000n + counter) & 0xffffffffffffn;
	return raw.toString(16).padStart(12, "0");
}

/**
 * Bounded cache keyed by the raw sha256 digest of `"ses\x00" + sessionId`.
 * Keeps the session stable per conversation so key and proxy affinity do not
 * change across turns. When the limit is reached the whole map is replaced
 * (mirrors Go's full-map eviction strategy in `CanonicalSessionID`).
 */
const SESSION_CACHE_LIMIT = 8192;
let sessionCache = new Map<string, string>();

/**
 * Format a session identity as OpenCode's canonical session token
 * (`ses_` + 12 lowercase hex + 14 base-62 chars).
 *
 * Passes through any value that already matches {@link OPENCODE_SESSION_TOKEN_PATTERN}
 * unchanged, preserving upstream prompt-cache affinity. Any other identity
 * (UUIDs, foreign client sessions, conversation seeds) gets a stable minted
 * session: the 12-hex head is clocked (`~(unix_ms*0x1000+counter)` low 48 bits)
 * so Zen's client-identity check passes (a digest prefix is rejected with
 * 403 FreeTierError — issue #68); the 14-char base-62 tail derives from the
 * digest for per-session uniqueness. A bounded in-process cache keeps the full
 * token constant across turns.
 *
 * Mirrors `CanonicalSessionID` in the upstream Go `internal/identity/request.go`
 * (PR jasonxu114514/opencode2api#69).
 */
export function toOpenCodeSessionToken(sessionId: string): string {
	if (OPENCODE_SESSION_TOKEN_PATTERN.test(sessionId)) return sessionId;

	// Key: raw bytes of sha256("ses\x00" + signal), mirroring Go's sum[:].
	const sum = new Bun.CryptoHasher("sha256").update("ses\x00").update(sessionId).digest();
	// Use a hex string as the Map key (Uint8Array is not value-equal in JS).
	const key = Buffer.from(sum).toString("hex");

	const cached = sessionCache.get(key);
	if (cached !== undefined) return cached;

	// Tail: 10 bytes of the digest starting at byte 6 (sum[6:16] in Go).
	const tailInt = BigInt(`0x${key.slice(12, 32)}`);
	const tail = base62Fixed(tailInt, 14);
	const id = `ses_${mintedTimePrefix()}${tail}`;

	if (sessionCache.size >= SESSION_CACHE_LIMIT) {
		sessionCache = new Map();
	}
	sessionCache.set(key, id);
	return id;
}
