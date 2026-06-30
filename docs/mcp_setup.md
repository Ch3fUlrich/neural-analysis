# MCP Server Setup (Serena)

This repository ships a ready-to-use [Serena](https://github.com/oraios/serena)
MCP server configuration so that Claude Code (and other MCP-aware agents) get
**LSP-powered semantic code navigation** — finding symbols, references, and
file outlines without reading whole files (large token savings on a codebase
this size).

The setup is committed in the repo so it works from a clean checkout. It is
modeled on the canonical configuration in the `agent-skills` repo
(`mcp-servers/config/mcp-claude-code.json`), reduced to **Serena only** (this
repo does not run the Mem0 / Superpowers infrastructure from that stack;
Superpowers is already available as a Claude Code plugin).

## Files

| File | Committed? | Purpose |
|---|---|---|
| `.mcp.json` | ✅ yes | Registers the `serena` MCP server (the missing piece) |
| `.serena/project.yml` | ✅ yes | Serena project config (name + language list) |
| `.claude/settings.local.json` | ❌ gitignored | Pre-approves the `serena` server so it connects without a prompt (personal) |

## How it works

`.mcp.json` launches Serena via `uvx`:

```jsonc
uvx --from serena-agent serena start-mcp-server \
    --project-from-cwd \          // auto-detect THIS repo from cwd (.serena/project.yml / .git)
    --open-web-dashboard false \
    --enable-gui-log-window false
```

- **`--project-from-cwd`** means Serena auto-activates the correct project based
  on the working directory — no manual `activate_project` call needed, and the
  same config works across repos.
- **`SERENA_HOME` is set to `C:/Users/mauls/.serena`** to exactly match the
  known-working Gemini/Antigravity config (`~/.gemini/antigravity/mcp_config.json`).
  That path is also Serena's default home, but setting it explicitly removes any
  ambiguity about how the home resolves inside Claude Code's MCP subprocess.
  *(On a different machine/user, change this path or drop the `env` block to use
  the default `~/.serena`.)*
- **`excludeTools`** hides Serena's memory and project-lifecycle tools
  (`write_memory`/`read_memory`/…, `activate_project`, `onboarding`,
  `open_dashboard`, …), matching the Gemini config. `--project-from-cwd` already
  auto-activates the project, so `activate_project` is unnecessary; persistent
  memory is handled separately (Mem0 / the harness file-memory). The
  semantic-navigation and refactoring tools — `find_symbol`,
  `get_symbols_overview`, `find_referencing_symbols`, `find_declaration`,
  `rename_symbol`, `replace_symbol_body`, `insert_after_symbol`, … — remain
  available. (Serena's global `~/.serena/serena_config.yml` additionally filters
  redundant file/shell tools that Claude Code already provides.)

## Prerequisites

- **`uv` / `uvx`** on `PATH` (already required by this project for env
  management — see `README.md`). Serena itself is fetched/run on demand by
  `uvx --from serena-agent`.
- First launch may take a few seconds while `uvx` resolves `serena-agent`
  (cached afterwards).

## Activating it

MCP servers are initialized at **session start**, so after pulling these files
you must **restart Claude Code** (or run `/mcp` → reconnect) for the `serena`
tools to appear. They are not hot-loaded into a running session.

## Verify

Once connected, confirm Serena works:

```
mcp__serena__get_symbols_overview(relative_path="src/neural_analysis/pipeline.py")
mcp__serena__find_symbol(name_path="compute_structure_index")
```

Or from a shell:

```bash
serena --version                 # installed CLI (currently Serena 1.5.3)
# quick boot smoke test (Ctrl-C after it prints "Activating neural-analysis"):
uvx --from serena-agent serena start-mcp-server --project-from-cwd \
    --open-web-dashboard false --enable-gui-log-window false
```

## Languages

`.serena/project.yml` lists only the languages this repo actually uses
(`python`, `markdown`, `yaml`, `toml`, `json`). The `go` and `csharp` language
servers were removed because they fail to start without Go / .NET 10 installed
and produced noisy startup errors; `cpp`, `rust`, `typescript`, `html`, and
`scss` were removed as unused. Add a language back to that list if you start
working in it.
