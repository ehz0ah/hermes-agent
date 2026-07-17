# OpenViking Memory Provider

Context database by Volcengine (ByteDance) with filesystem-style knowledge hierarchy, tiered retrieval, and automatic memory extraction.

## Requirements

- `pip install openviking`
- OpenViking server running (`openviking-server`)
- Embedding + VLM model configured in `~/.openviking/ov.conf`

## Setup

```bash
hermes memory setup    # select "openviking"
```

The setup can link to an existing `~/.openviking/ovcli.conf`, copy its current
connection values into Hermes, or create a minimal `ovcli.conf` when one does
not exist.

Or manually:
```bash
hermes config set memory.provider openviking
echo "OPENVIKING_ENDPOINT=http://localhost:1933" >> ~/.hermes/.env
```

## Config

All config via environment variables in `.env`:

| Env Var | Default | Description |
|---------|---------|-------------|
| `OPENVIKING_ENDPOINT` | `http://127.0.0.1:1933` | Server URL |
| `OPENVIKING_API_KEY` | (none) | User/admin API key for authenticated servers |
| `OPENVIKING_ACCOUNT` | `default` | Tenant account for local/trusted mode |
| `OPENVIKING_USER` | `default` | Tenant user for local/trusted mode |
| `OPENVIKING_AGENT` | `hermes` | Hermes peer ID in OpenViking, used for peer-scoped memories |
| `OPENVIKING_IDENTITY_MODE` | `solo` | `solo` for one human, `team` for messaging gateways where multiple humans share one Hermes agent |
| `OPENVIKING_IDLE_COMMIT_SECONDS` | `900` | Commit/extract an idle OpenViking session after this many seconds; `0` disables idle checkpoints |
| `OPENVIKING_IDLE_COMMIT_KEEP_RECENT` | `0` | Recent messages to leave unarchived during idle checkpoints; `0` extracts short quiet sessions immediately |

When `OPENVIKING_API_KEY` is set, Hermes lets OpenViking derive account/user
identity from the key. In local or trusted deployments without an API key,
Hermes sends `OPENVIKING_ACCOUNT` and `OPENVIKING_USER` as identity headers.

## Identity Modes

`solo` is the default and preserves the existing behavior: Hermes writes under
the configured `OPENVIKING_AGENT` peer.

`team` is for gateways such as Feishu/Lark where one Hermes agent participates
as a team colleague. Hermes keeps one shared OpenViking user namespace, writes
human messages under a deterministic platform peer derived from the sender's
stable gateway identity, and writes assistant procedural memories under the
Hermes peer. Search/read/browse use the shared user namespace without an actor
peer so the agent can recall what it has learned across the team.

Team mode is not valid for direct CLI sessions. Run `hermes memory setup
openviking` again and choose `solo` for CLI-only profiles.

OpenViking also supports idle checkpoints. Terminal session boundaries such as
shutdown, `/new`, and compression still commit normally. Separately, when a
session has been quiet for `OPENVIKING_IDLE_COMMIT_SECONDS`, Hermes asks
OpenViking to commit/extract that session without ending the Hermes session.
The default leaves no recent messages unarchived so short quiet gateway
conversations become searchable promptly.

Team mode also provides an immediate, prompt-only bridge across Hermes gateway
sessions. Before an addressed turn, Hermes reads up to 50 recent user/assistant
messages from other chats, DMs, or threads in the same profile and includes the
newest excerpt that fits a 10,000-token budget. The excerpt carries readable
platform, chat, thread, sender, and timestamp provenance. It is not persisted,
added to the agent cache key, or synchronized to OpenViking.

The bridge is enabled by default only for OpenViking Team mode. Operators can
override it in `config.yaml`:

```yaml
gateway:
  cross_session_context:
    enabled: true
    max_messages: 50
    max_tokens: 10000
```

## Tools

| Tool | Description |
|------|-------------|
| `viking_search` | Semantic search with fast/deep/auto modes |
| `viking_read` | Read content at a viking:// URI (abstract/overview/full) |
| `viking_browse` | Filesystem-style navigation (list/tree/stat) |
| `viking_remember` | Store a fact directly with OpenViking `content/write` |
| `viking_forget` | Delete one exact `viking://` memory file URI |
| `viking_add_resource` | Ingest URLs/docs into the knowledge base |

## Memory Writes And Deletes

`viking_remember` writes directly to OpenViking with `POST /api/v1/content/write`
and `mode=create`. It creates peer-scoped memory files under
`viking://user/peers/${OPENVIKING_AGENT}/memories/...`; OpenViking may return a
canonical user-scoped form such as
`viking://user/default/peers/${OPENVIKING_AGENT}/memories/...` in API-key mode.
Explicit remembers do not depend on session commit extraction.

In `team` mode, `viking_remember` uses the optional `owner` argument to choose
the namespace: `human` writes under the active sender's peer, while `self`
writes under `viking://user/memories/...` for Hermes's own reusable procedures
or commitments. When omitted, `owner` defaults to `human`; `category` only
chooses the memory subdirectory such as `preferences`, `entities`, `events`,
`cases`, or `patterns`. Hermes also maintains a small best-effort
`resources/profile.md` file for each observed human peer with readable
display/mention metadata for attribution.

Hermes built-in `memory` tool additions are mirrored to OpenViking after the
local memory operation succeeds:

| Hermes action | OpenViking operation |
|---------------|----------------------|
| `add` | `content/write` with `mode=create` under the configured peer memory namespace |

Built-in `replace` and `remove` operations are not mirrored because Hermes
native memory entries do not yet carry stable OpenViking file URIs. Use
`viking_forget` when the user explicitly asks to delete a specific OpenViking
memory URI.

`viking_forget` is intentionally narrow. It only accepts concrete user memory
file URIs, such as
`viking://user/peers/hermes/memories/preferences/mem_abc123.md` or the canonical
`viking://user/default/peers/hermes/memories/preferences/mem_abc123.md`. Files
directly under `memories/`, such as `viking://user/default/memories/profile.md`,
are also allowed because OpenViking supports them. The tool rejects directories,
resources, skills, sessions, generated summary files, and URIs with query
strings or fragments. Use OpenViking's MCP, CLI, or admin APIs for broader
resource and directory cleanup.
