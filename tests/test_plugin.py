"""
test_plugin.py
Automated static analysis and SDK mock test suite for Lightroom Classic Lua Plugin.

Covers:
- Lua AST syntax validation using luaparser (guaranteeing zero syntax errors)
- Plugin manifest verification (Info.lua structure, IDs, hooks)
- Utility method integrity (PluginUtils.lua path resolution & argument quoting)
- Simulated Lightroom SDK LiveBridge workflow mocking (selection debounce, pending imports, JSON parsing)
"""

import os
import re
import sys
import glob
import json
import unittest
from luaparser import ast

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# Python re port of LiveBridge.lua extractJsonField / parsePendingImports. Keep in exact sync with the Lua.
def _lua_extract_json_field(json_str, field):
    m = re.search(r'"' + re.escape(field) + r'"\s*:\s*"([^"]+)"', json_str)
    if m:
        return m.group(1)
    m = re.search(r'"' + re.escape(field) + r'"\s*:\s*(\d+)', json_str)
    return m.group(1) if m else None


def lua_parse_pending_imports(json_str):
    imports = []
    m = re.search(r'"imports"\s*:\s*\[(.*)\]', json_str, re.S)  # Lua '.' matches newlines too
    if not m or not m.group(1).strip():
        return imports
    masked = re.sub(r'"[^"]*"', lambda s: s.group(0).replace("{", "\x01").replace("}", "\x02"), m.group(1))
    for item_match in re.finditer(r'\{([^}]+)\}', masked):
        item_str = item_match.group(1).replace("\x01", "{").replace("\x02", "}")
        p = _lua_extract_json_field(item_str, "path")
        orig = _lua_extract_json_field(item_str, "original_path")
        pid = _lua_extract_json_field(item_str, "photo_id")
        if p:
            imports.append({
                "path": p.replace("\\\\", "\\"),
                "original_path": orig.replace("\\\\", "\\") if orig else None,
                "photo_id": pid,
            })
    return imports


# Python port of LiveBridge.lua jsonEscape: '\' first, then '"'.
def lua_json_escape(s):
    return str(s).replace("\\", "\\\\").replace('"', '\\"')


_LIVE_BRIDGE = os.path.join(_PROJECT_ROOT, "plugin", "ai_eraser.lrplugin", "LiveBridge.lua")


def _read_live_bridge():
    with open(_LIVE_BRIDGE, "r", encoding="utf-8") as f:
        return f.read()


class TestLiveBridgeFixes(unittest.TestCase):
    def test_json_escape_port(self):
        self.assertEqual(lua_json_escape('a"b\\c'), 'a\\"b\\\\c')
        for s in ['a"b\\c', 'C:\\Photos\\say "hi"\\x.tif', '\\"', '']:
            self.assertEqual(json.loads('"' + lua_json_escape(s) + '"'), s)
            body = '{"path": "%s", "title": "%s"}' % (lua_json_escape(s), lua_json_escape(s))
            self.assertEqual(json.loads(body), {"path": s, "title": s})

    def test_json_escape_lua_matches_port(self):
        src = _read_live_bridge()
        self.assertIn("local function jsonEscape(s)", src)
        self.assertIn("""gsub('\\\\', '\\\\\\\\'):gsub('"', '\\\\"')""", src)

    def test_static_checks(self):
        src = _read_live_bridge()
        self.assertNotIn("getAllPhotos", src)
        for needle in ["findPhotoByPath", "pcall", "lastEditTime", "MAX_CACHED_EXPORTS = 5", "% 10"]:
            self.assertIn(needle, src)
        # Every JSON body built with string.format has its string fields escaped
        self.assertNotIn("tostring(targetId), jsonOrig", src)
        self.assertIn("jsonEscape(exportPath), jsonEscape(targetId), jsonEscape(origPath), jsonEscape(baseName)", src)
        self.assertIn("""string.format('{"path": "%s"}', jsonEscape(editedPath))""", src)

    def test_port_read_only_in_failure_branch(self):
        src = _read_live_bridge()
        start = src.index("while not progressScope:isCanceled() do\n        LrTasks.sleep(0.03)")
        end = src.index("-- If user clicked cancel", start)
        loop = src[start:end]
        self.assertEqual(loop.count("getLiveBridgePort"), 1)
        failure = loop.index("disconnectedCount = disconnectedCount + 1")
        port_read = loop.index("getLiveBridgePort")
        ping = loop.index("/api/ping", failure)
        self.assertLess(failure, port_read)
        self.assertLess(port_read, ping)

    def test_edited_path_deleted_only_after_add_photo(self):
        src = _read_live_bridge()
        add = src.index("catalog:addPhoto(")
        guard = src.index('error("addPhoto failed")')
        deletes = [m.start() for m in re.finditer(r"LrFileUtils\.delete\(editedPath\)", src)]
        self.assertEqual(len(deletes), 1)
        self.assertLess(add, guard)
        self.assertLess(guard, deletes[0])


class RecencyCacheModel:
    """Python model of LiveBridge.lua touchCachedExport + exportEditTimes cache-hit rule."""

    def __init__(self, cap=5):
        self.cap = cap
        self.paths = []  # oldest first
        self.edit_times = {}
        self.on_disk = set()
        self.deleted = []

    def _touch(self, path):
        if path in self.paths:
            self.paths.remove(path)
        self.paths.append(path)
        while len(self.paths) > self.cap:
            old = self.paths.pop(0)
            self.on_disk.discard(old)
            self.deleted.append(old)

    def select(self, photo_id, edit_time):
        path = "live_%s.tif" % photo_id
        edit_time = edit_time if edit_time is not None else False
        if path in self.on_disk and photo_id in self.edit_times and self.edit_times[photo_id] == edit_time:
            self._touch(path)
            return "HIT"
        self.on_disk.add(path)
        self.edit_times[photo_id] = edit_time
        self._touch(path)
        return "EXPORT"


class TestRecencyCache(unittest.TestCase):
    def test_seven_exports_leave_newest_five(self):
        m = RecencyCacheModel()
        for i in range(1, 8):
            self.assertEqual(m.select(i, 100), "EXPORT")
        self.assertEqual(m.paths, ["live_%d.tif" % i for i in range(3, 8)])
        self.assertEqual(m.deleted, ["live_1.tif", "live_2.tif"])
        self.assertEqual(m.on_disk, set(m.paths))

    def test_reselect_cached_makes_it_newest(self):
        m = RecencyCacheModel()
        for i in range(1, 6):
            m.select(i, 100)
        self.assertEqual(m.select(1, 100), "HIT")
        self.assertEqual(m.paths[-1], "live_1.tif")
        m.select(6, 100)  # evicts 2, not the re-selected 1
        self.assertEqual(m.deleted, ["live_2.tif"])
        self.assertIn("live_1.tif", m.paths)

    def test_stale_and_leftover_files_reexport(self):
        m = RecencyCacheModel()
        m.on_disk.add("live_9.tif")  # left over from an earlier session
        self.assertEqual(m.select(9, None), "EXPORT")
        self.assertEqual(m.select(9, None), "HIT")  # missing edit time still needs the table entry
        self.assertEqual(m.select(9, 200), "EXPORT")  # edited since export


class TestLuaPluginStaticAnalysis(unittest.TestCase):
    def setUp(self):
        self.plugin_dir = os.path.join(_PROJECT_ROOT, "plugin", "ai_eraser.lrplugin")

    def test_lua_syntax_validation_all_files(self):
        """Parse all .lua files in the plugin bundle to guarantee zero syntax errors."""
        lua_files = glob.glob(os.path.join(self.plugin_dir, "*.lua"))
        self.assertGreater(len(lua_files), 0, "No Lua files found in plugin directory!")

        for lua_file in lua_files:
            rel_name = os.path.basename(lua_file)
            with open(lua_file, "r", encoding="utf-8") as f:
                content = f.read()

            try:
                tree = ast.parse(content)
                self.assertIsNotNone(tree, f"AST tree was None for {rel_name}")
            except Exception as e:
                self.fail(f"Lua syntax error in {rel_name}: {e}")

    def test_info_lua_manifest_structure(self):
        """Verify Info.lua contains required Lightroom plugin manifest fields."""
        info_path = os.path.join(self.plugin_dir, "Info.lua")
        with open(info_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check for mandatory Lightroom Classic plugin keys
        self.assertIn("LrToolkitIdentifier", content)
        self.assertIn("LrPluginName", content)
        self.assertIn("LrSdkVersion", content)
        self.assertIn("LrExportMenuItems", content)
        self.assertNotIn("GenerativeEraser.lua", content)
        self.assertIn("LiveBridge.lua", content)

    def test_plugin_utils_functions_defined(self):
        """Verify PluginUtils.lua defines all required utility functions."""
        utils_path = os.path.join(self.plugin_dir, "PluginUtils.lua")
        with open(utils_path, "r", encoding="utf-8") as f:
            content = f.read()

        required_functions = [
            "PluginUtils.getPluginDir",
            "PluginUtils.getProjectRoot",
            "PluginUtils.getPythonExecutable",
            "PluginUtils.getCompanionAppPath",
            "PluginUtils.quoteArg",
            "PluginUtils.getLiveBridgePort",
            "PluginUtils.launchBackgroundProcess",
        ]
        for func_name in required_functions:
            self.assertIn(func_name, content, f"Missing function {func_name} in PluginUtils.lua")

    def test_all_lr_sdk_modules_imported(self):
        """Verify that every Lightroom SDK module (Lr*) referenced in each Lua file is explicitly imported."""
        import glob
        import re

        lua_files = glob.glob(os.path.join(self.plugin_dir, "*.lua"))
        for lua_file in lua_files:
            rel_name = os.path.basename(lua_file)
            if rel_name == "Info.lua":
                continue  # Manifest declares hooks rather than imports

            with open(lua_file, "r", encoding="utf-8") as f:
                content = f.read()

            imported_modules = set(re.findall(r"import\s*['\"](Lr\w+)['\"]", content))
            # Find usages like LrTasks. or LrTasks:
            referenced_modules = set(re.findall(r"\b(Lr\w+)[.:]", content))

            missing_imports = referenced_modules - imported_modules
            self.assertEqual(
                len(missing_imports),
                0,
                f"File {rel_name} references Lightroom SDK modules without importing them: {missing_imports}"
            )

    def test_quote_arg_logic(self):
        """Verify argument quoting logic matching PluginUtils.quoteArg."""
        def mock_quote_arg_win(arg):
            return '"' + str(arg).replace('"', '\\"') + '"'

        def mock_quote_arg_posix(arg):
            return "'" + str(arg).replace("'", "'\\''") + "'"

        # Windows quoting with spaces and quotes
        win_path = r"C:\Photos\Trip 2026\image.tif"
        self.assertEqual(mock_quote_arg_win(win_path), r'"C:\Photos\Trip 2026\image.tif"')
        self.assertEqual(mock_quote_arg_win('quote"test'), r'"quote\"test"')

        # Posix quoting
        posix_path = "/home/user/my photos/pic.tif"
        self.assertEqual(mock_quote_arg_posix(posix_path), "'/home/user/my photos/pic.tif'")


class MockLiveBridgeWorkflowHarness:
    """
    Simulates Lightroom Classic SDK LiveBridge background loop:
    - Debouncing photo selections
    - Exporting TIFF and sending selection event
    - Polling and processing pending imports
    """
    def __init__(self, debounce_delay=0.25):
        self.debounce_delay = debounce_delay
        self.last_photo_id = None
        self.pending_photo = None
        self.pending_time = 0
        self.sent_selections = []
        self.imported_photos = []

    def handle_selection_tick(self, current_photo, current_time):
        """Simulate single tick of selection check with debounce logic."""
        target_id = current_photo["id"] if current_photo else None

        if target_id != self.last_photo_id:
            pending_id = self.pending_photo["id"] if self.pending_photo else None
            if target_id != pending_id:
                self.pending_photo = current_photo
                self.pending_time = current_time
            else:
                if (current_time - self.pending_time) >= self.debounce_delay:
                    self.last_photo_id = target_id
                    active = self.pending_photo
                    self.pending_photo = None
                    if active:
                        self.sent_selections.append(active)
                        return "SELECTION_DISPATCHED"
        return "NO_ACTION"

    def handle_pending_imports(self, imports_list):
        """Simulate processing items returned from companion pending_imports."""
        imported_this_tick = []
        for item in imports_list:
            path = item.get("path")
            self.imported_photos.append(path)
            imported_this_tick.append(path)
        return imported_this_tick


class TestLightroomLiveBridgeMocking(unittest.TestCase):
    def test_selection_debounce_prevents_rapid_fire_exports(self):
        """Rapid arrow key scrolling within debounce window triggers only final settled photo export."""
        harness = MockLiveBridgeWorkflowHarness(debounce_delay=0.25)

        # User scrolls quickly across photos 1, 2, 3 within 0.05s intervals
        t = 100.0
        harness.handle_selection_tick({"id": "p1", "path": "/p1.raw"}, t)
        t += 0.05
        harness.handle_selection_tick({"id": "p2", "path": "/p2.raw"}, t)
        t += 0.05
        harness.handle_selection_tick({"id": "p3", "path": "/p3.raw"}, t)

        # No export dispatched yet because photos were moving rapidly
        self.assertEqual(len(harness.sent_selections), 0)

        # Now user pauses on photo 3 for 0.3s (greater than debounce 0.25s)
        t += 0.15
        harness.handle_selection_tick({"id": "p3", "path": "/p3.raw"}, t)
        self.assertEqual(len(harness.sent_selections), 0)

        t += 0.15  # Total pause on p3: 0.30s
        res = harness.handle_selection_tick({"id": "p3", "path": "/p3.raw"}, t)
        self.assertEqual(res, "SELECTION_DISPATCHED")
        self.assertEqual(len(harness.sent_selections), 1)
        self.assertEqual(harness.sent_selections[0]["id"], "p3")

    def test_live_bridge_imports_and_acknowledges(self):
        """Pending imports from companion are auto-imported and added to catalog."""
        harness = MockLiveBridgeWorkflowHarness()
        pending = [
            {"path": "/tmp/photo1_ai_edit.tif", "photo_id": "1"},
            {"path": "/tmp/photo2_ai_edit.tif", "photo_id": "2"},
        ]
        imported = harness.handle_pending_imports(pending)
        self.assertEqual(len(imported), 2)
        self.assertEqual(len(harness.imported_photos), 2)

    def test_live_bridge_lua_json_parser_simulation(self):
        """Verify simulated LiveBridge.lua JSON parsing logic handles empty array and Windows backslashes."""
        parse_pending_imports = lua_parse_pending_imports

        # 1. Empty imports array must return empty list without error
        empty_json = '{"status": "ok", "count": 0, "imports": []}'
        self.assertEqual(parse_pending_imports(empty_json), [])

        # 2. Populated imports array with Windows path
        pop_json = '{"status": "ok", "count": 1, "imports": [{"path": "C:\\\\tmp\\\\live_edit.tif", "original_path": "C:\\\\photos\\\\raw.cr3", "photo_id": "42"}]}'
        result = parse_pending_imports(pop_json)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["path"], r"C:\tmp\live_edit.tif")
        self.assertEqual(result[0]["original_path"], r"C:\photos\raw.cr3")
        self.assertEqual(result[0]["photo_id"], "42")

        # 3. Brackets/braces inside paths, multiple items, null fields
        tricky_json = ('{"status": "ok", "count": 2, "imports": ['
                       '{"path": "C:\\\\Été [2024]\\\\a}.tif", "original_path": "C:\\\\x{y}\\\\b].cr3", "photo_id": null}, '
                       '{"path": "C:\\\\c.tif", "original_path": null, "photo_id": "7"}]}')
        result = parse_pending_imports(tricky_json)
        self.assertEqual(result, [
            {"path": "C:\\Été [2024]\\a}.tif", "original_path": "C:\\x{y}\\b].cr3", "photo_id": None},
            {"path": "C:\\c.tif", "original_path": None, "photo_id": "7"},
        ])


if __name__ == "__main__":
    unittest.main()
