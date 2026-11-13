# RegRadar MCP setup

> **Pending nemsoft-eu/regradar#44.** RegRadar does not serve MCP yet. The
> Executive's side is in place and stays inert until the entry below is
> added. Do not add it before that issue's server ships and the checks in
> [Verify before enabling](#verify-before-enabling) pass.

RegRadar is Nemsoft's internal regulatory horizon scanner. It runs as a
separate service on the same host as the Executive and will serve its
findings over MCP with three read tools: `search_findings`, `get_finding` and
`latest_digest`. The Executive reaches it through its MCP gateway as a plain
url server, so its tools appear in chat as `regradar__search_findings` and so
on.

The gateway runs only those three names. Any other `regradar__` tool,
including one RegRadar adds later, is refused, audited as a blocked RegRadar
call, and never offered by a tool search, until someone adds its name to
`_REGRADAR_READ_TOOLS` in `packages/core/openexecutive/orchestrator/mcp_gateway.py`
on purpose. That check covers names, not behaviour: it trusts that the three
tools only read, which is why the live server is checked before the entry is
enabled.

RegRadar's tools are available on ordinary chat turns. Private, hosted,
MCP-client and delegated scheduled turns do not offer them.

---

## Prerequisites

- RegRadar's MCP server is running (`mcp serve`), with the port and transport
  given in nemsoft-eu/regradar#44.
- A bearer token for the Executive, issued by RegRadar.
- MCP is enabled in the Executive (`/data/company/mcp_servers.json` exists;
  see `docs/deployment.md`).

---

## Verify before enabling

Before you add the entry, list the live server's tools (an MCP `tools/list`
request against RegRadar's endpoint, with the token) and read each schema:

1. The tool names must be exactly `search_findings`, `get_finding` and
   `latest_digest`.
2. No tool may write, and none may take a URL or fetch anything on the
   caller's behalf.
3. Each tool's description should explain a finding's `status`, and each
   result row whose status is `unconfirmed_secondary` should say that it rests
   on interpretive sources only and is not a confirmed obligation.

If the names differ, change `_REGRADAR_READ_TOOLS` and its tests
(`packages/core/tests/unit/test_regradar_readonly.py`) deliberately, in a
reviewed change. Never widen the list just to make a call work.

---

## Configure the entry

Add a url entry to `/data/company/mcp_servers.json` under `mcpServers`:

```json
"regradar": {"url": "http://host.containers.internal:<port>/mcp"}
```

Keep the name exactly `regradar`. The gateway's check is keyed to the
`regradar__` prefix, so an entry under any other name carries none of it.

Add the token to the extensible-mcp tokens file (`tokens` in its config
directory, or the path in `$EXTENSIBLE_MCP_TOKENS_FILE`) as one line:

```
regradar=<token>
```

Edit the file in an editor and keep it at mode `0600`. Never put the token on
a command line, where process listings and shell history can show it.

Optionally, deny the name for runtime loads as a second line of defence, by
adding this key to the existing `filters.load_control` block of the same
file:

```json
"deny_name_patterns": ["regradar*"]
```

The Executive already refuses a runtime `load_mcp_server` named `regradar` or
`linear`, in any spelling, so a model-chosen URL can never receive that
entry's token. The filter repeats that refusal in the deployment config.

---

## Start order

The Executive fetches a url server's tools once, at boot. RegRadar must be up
before the Executive starts. If it was down, the boot log says
`Failed to connect to 'regradar', skipping`, and RegRadar's tools stay
unavailable until you restart the Executive after RegRadar is up. Restarting
RegRadar later does not reconnect it either.

---

## Verify after enabling

1. The boot log lists the `regradar` server with three tools.
2. A chat question such as "Which in-force obligations affect the cleaning
   sector?" calls `regradar__search_findings`.
3. A call to any other `regradar__` name is refused with "Do not retry", and
   the audit log shows it as `integration_outbound_blocked` with address
   `<regradar-write>`.
