# Security

Efface Magique LR runs entirely on your machine. The only network surface is the companion's HTTP bridge (`companion/live_bridge.py`), which the Lightroom plugin (`plugin/ai_eraser.lrplugin/LiveBridge.lua`) talks to. This document states what that bridge protects against and what it does not. See [docs/architecture.md](docs/architecture.md) for the full data flow.

## What the bridge enforces

- **Loopback only.** The server binds `127.0.0.1` (port `51739`, or an OS-assigned port if that one is busy). It never binds `0.0.0.0`, so other machines on the network cannot reach it.
- **Origin / Host guard.** Every GET and POST first passes `_reject_untrusted`. A request is answered with **403** if:
  - it has an `Origin` header (any value), or
  - its `Host` header is not exactly `127.0.0.1:<port>`. `localhost:<port>`, a missing `Host`, or any other hostname is rejected too.
- **No CORS headers.** The bridge sends no `Access-Control-Allow-*` headers.
- **Body limits on POST.** A `Content-Length` over 64 KiB (`MAX_BODY_BYTES = 64 * 1024`) gets **413** before the body is read. A negative or non-integer `Content-Length` gets **400**. A body that is not valid UTF-8 JSON gets **400**.

### Why removing CORS headers was not enough

Without CORS headers a web page cannot *read* the bridge's responses, but it can still *send* requests. A browser sends a "simple" cross-origin `POST` (for example `Content-Type: text/plain`) without a preflight. So a malicious page could hit `/api/close` or `/api/select`, and the side effect would happen even though the page never sees the response. Browsers add an `Origin` header to such requests, and the Origin check rejects them.

DNS rebinding is the second attack. An attacker's hostname is made to resolve to `127.0.0.1`, so the page becomes "same-origin" with the bridge. Requests from that page carry `Host: attacker.example:<port>`, and the exact-`Host` check rejects them.

## Endpoints

| Method | Path | Side effect |
| :--- | :--- | :--- |
| GET | `/api/ping` | None. Returns `app: "EffaceMagique"`, version, the companion's PID and `active_image` (the last `/api/select` path) |
| GET | `/api/pending_imports` | None. Returns the queued edits (`path`, `original_path`, `photo_id`) waiting for Lightroom to import |
| POST | `/api/select` | Sets `active_image_path` and emits `photo_selected`. The companion then loads that file (and may first ask to discard unsynced layers). Requires `path` |
| POST | `/api/import_done` | Removes `path` from the pending-import queue and emits `import_completed` (status bar message) |
| POST | `/api/focus` | Raises and activates the companion window |
| POST | `/api/close` | Emits `close_requested`, so the companion window closes, stops the bridge and clears `.tmp` |
| POST | `/api/companion_closed` | None. Returns 200. The companion POSTs this to itself on close |

Any other path gets 404. While the bridge is stopping, requests get 503.

## Residual risks

- **No authentication.** There is no token or secret. Any local process, running as any user on the machine, can reach `127.0.0.1:<port>` and drive the bridge. It can select a file to load, focus or close the window, read the pending imports (file paths of your photos) and acknowledge them. It cannot make the plugin delete or overwrite arbitrary files: the plugin only imports `live_<id>.tif` files it exported itself (see below). At most it can trigger an import of an edit from the current session, into that photo's own folder. The Origin/Host guard only stops browsers. It does not stop local programs, which can set any headers they like.
- **`.tmp/live_bridge.json` is trusted for port discovery.** The plugin reads the port from `<project>/.tmp/live_bridge.json`, falling back to `51739`. Anything that can write that file, or bind the port first, can point the plugin at a different local server.
- **Anything that answers the ping is trusted.** The plugin treats any responder whose `/api/ping` body contains the text `EffaceMagique` as the companion. It does not check the PID or any secret.

## What the plugin trusts

- **Paths from `/api/pending_imports`: validated, not trusted.** Before any file operation, `ImportGuard.lua` checks each item. It accepts an item only if `path` is absolute, has no `..` segment, sits directly in `<project>/.tmp` (exact folder match, so `.tmp2/` fails), and is named `live_<id>.tif` (or `.tiff`), where `<id>` equals `photo_id` and is a photo the plugin sent to `/api/select` this session. The original folder comes from the plugin's own record for that id. The server's `original_path` is never used to build the destination, and an item whose `original_path` differs from the record is rejected. Separators are normalised, and Windows paths are compared case-insensitively. A rejected item is shown in the Live Sync caption ("Import skipped (<reason>)"), acknowledged with `/api/import_done` so it is not retried, and no file is touched. (Before this check, a crafted `path` / `original_path` could make the plugin delete and replace a same-named file in any folder Lightroom could write to. That is fixed.) For an accepted item the plugin:
  1. builds `<folder of the recorded original>/live_<id>.tif`;
  2. deletes any existing file with that name there, then copies the edited file there. Re-syncing the same photo replaces the earlier `live_<id>.tif` next to the original;
  3. imports it with `catalog:addPhoto`, stacked above the original;
  4. deletes the source file at `path` **after a successful import**, if it was copied.

  Side effects: a Sync of a file opened outside Lightroom (Open File, saved as `*_ai_edit.*`) is queued but skipped by the plugin. Items queued before a Lightroom restart are skipped too, because the session record starts empty. An `_ai_edit` file stays beside its source. A stale `.tmp` edit is removed by the normal `.tmp` cleanup.
- **Ping responders**, as described above.

## What the companion trusts

- **`/api/select` paths.** The only check is `os.path.isfile(path)`. Any existing file the companion user can read can be loaded. On save, a file under `<project>/.tmp` is overwritten in place. Any other file is saved as `<name>_ai_edit<ext>` beside it, and the original is never overwritten.
- `original_path` and `photo_id` from `/api/select` are stored as they are and echoed back through `/api/pending_imports`.

## Reporting a vulnerability

Please report security issues through GitHub. Open a [private security advisory](https://github.com/LLiu223/efface-magique-lr/security/advisories/new) for anything sensitive, or a regular [issue](https://github.com/LLiu223/efface-magique-lr/issues) for low-risk findings. Include the commit, your OS and Lightroom version, and steps to reproduce.
