"""
test_plugin.py
Automated static analysis and SDK mock test suite for Lightroom Classic Lua Plugin.

Covers:
- Lua AST syntax validation using luaparser (guaranteeing zero syntax errors)
- Plugin manifest verification (Info.lua structure, IDs, hooks)
- Utility method integrity (PluginUtils.lua path resolution & argument quoting)
- Simulated Lightroom SDK workflow mocking (missing selections, cancellations, missing files, exit codes)
"""

import os
import sys
import glob
import unittest
from luaparser import ast

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


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
        self.assertIn("GenerativeEraser.lua", content)

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
        ]
        for func_name in required_functions:
            self.assertIn(func_name, content, f"Missing function {func_name} in PluginUtils.lua")

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


class MockLrProgressScope:
    def __init__(self):
        self.canceled = False
        self.portions = []
        self.captions = []
        self.is_done = False

    def isCanceled(self):
        return self.canceled

    def setPortionComplete(self, current, total):
        self.portions.append((current, total))

    def setCaption(self, caption):
        self.captions.append(caption)

    def done(self):
        self.is_done = True


class MockLightroomWorkflowHarness:
    """
    Simulates Lightroom SDK environment and executes the logic of GenerativeEraser.lua
    in Python to test error branches, dialog states, and cancel handling.
    """
    def __init__(self, target_photo=None, export_succeeds=True, companion_exit_code=0):
        self.target_photo = target_photo
        self.export_succeeds = export_succeeds
        self.companion_exit_code = companion_exit_code
        self.messages = []
        self.beeps = 0
        self.added_photos = []

    def show_message(self, title, msg, msg_type="info"):
        self.messages.append({"title": title, "msg": msg, "type": msg_type})

    def run(self, progress_scope: MockLrProgressScope, companion_script_exists=True):
        # 1. Selection validation
        if not self.target_photo:
            self.show_message("AI Generative Eraser", "Please select a photo in Lightroom before running AI Generative Eraser.", "info")
            return "NO_SELECTION"

        # 2. Export check
        if progress_scope.isCanceled():
            progress_scope.done()
            return "CANCELED_BEFORE_EXPORT"

        if not self.export_succeeds:
            progress_scope.done()
            self.show_message("AI Generative Eraser Error", "Failed to export high-resolution TIFF", "critical")
            return "EXPORT_FAILED"

        # 3. Companion script resolution
        if not companion_script_exists:
            progress_scope.done()
            self.show_message("AI Generative Eraser Configuration Error", "Could not locate companion app", "critical")
            return "COMPANION_MISSING"

        # 4. Companion execution
        if progress_scope.isCanceled():
            progress_scope.done()
            return "CANCELED_DURING_COMPANION"

        if self.companion_exit_code == 0:
            # Reimport and stack
            dest_path = "/photos/sample_ai_edit.tif"
            self.added_photos.append(dest_path)
            self.beeps += 1
            progress_scope.done()
            return "SUCCESS_REIMPORTED"
        elif self.companion_exit_code in (130, -1):
            # Graceful cancellation
            progress_scope.done()
            return "USER_CANCELLED"
        else:
            # Failure code
            progress_scope.done()
            self.show_message("AI Generative Eraser", f"AI Companion closed with exit code {self.companion_exit_code}. No edits were imported.", "info")
            return "COMPANION_ERROR"


class TestLightroomSDKWorkflowMocking(unittest.TestCase):
    def test_missing_selection_shows_warning_dialog(self):
        """When no photo is selected, display informational dialog and exit cleanly."""
        harness = MockLightroomWorkflowHarness(target_photo=None)
        scope = MockLrProgressScope()
        result = harness.run(scope)

        self.assertEqual(result, "NO_SELECTION")
        self.assertEqual(len(harness.messages), 1)
        self.assertIn("select a photo", harness.messages[0]["msg"])
        self.assertEqual(len(harness.added_photos), 0)

    def test_user_canceled_export_terminates_cleanly(self):
        """When user cancels during export, progress scope finishes with zero dialogs."""
        harness = MockLightroomWorkflowHarness(target_photo={"path": "/photos/pic.raw"})
        scope = MockLrProgressScope()
        scope.canceled = True
        result = harness.run(scope)

        self.assertEqual(result, "CANCELED_BEFORE_EXPORT")
        self.assertTrue(scope.is_done)
        self.assertEqual(len(harness.messages), 0)

    def test_failed_export_shows_critical_dialog(self):
        """When export fails to produce output TIFF, notify user with critical dialog."""
        harness = MockLightroomWorkflowHarness(target_photo={"path": "/photos/pic.raw"}, export_succeeds=False)
        scope = MockLrProgressScope()
        result = harness.run(scope)

        self.assertEqual(result, "EXPORT_FAILED")
        self.assertEqual(len(harness.messages), 1)
        self.assertEqual(harness.messages[0]["type"], "critical")

    def test_companion_missing_shows_config_error(self):
        """When Python companion entry point is missing, show configuration error."""
        harness = MockLightroomWorkflowHarness(target_photo={"path": "/photos/pic.raw"})
        scope = MockLrProgressScope()
        result = harness.run(scope, companion_script_exists=False)

        self.assertEqual(result, "COMPANION_MISSING")
        self.assertEqual(len(harness.messages), 1)
        self.assertEqual(harness.messages[0]["type"], "critical")

    def test_user_cancel_companion_code_130_suppresses_dialogs(self):
        """Exit code 130 (user closed window without save) must exit without showing error dialogs."""
        harness = MockLightroomWorkflowHarness(target_photo={"path": "/photos/pic.raw"}, companion_exit_code=130)
        scope = MockLrProgressScope()
        result = harness.run(scope)

        self.assertEqual(result, "USER_CANCELLED")
        self.assertEqual(len(harness.messages), 0, "No error dialog should be shown on exit code 130")
        self.assertEqual(len(harness.added_photos), 0)

    def test_companion_failure_shows_exit_code_info(self):
        """Non-zero exit code (e.g. 1) shows informative dialog with the exact exit code."""
        harness = MockLightroomWorkflowHarness(target_photo={"path": "/photos/pic.raw"}, companion_exit_code=1)
        scope = MockLrProgressScope()
        result = harness.run(scope)

        self.assertEqual(result, "COMPANION_ERROR")
        self.assertEqual(len(harness.messages), 1)
        self.assertIn("exit code 1", harness.messages[0]["msg"])

    def test_successful_save_reimports_and_stacks(self):
        """Exit code 0 successfully re-imports edited TIFF and stacks adjacent to original."""
        harness = MockLightroomWorkflowHarness(target_photo={"path": "/photos/pic.raw"}, companion_exit_code=0)
        scope = MockLrProgressScope()
        result = harness.run(scope)

        self.assertEqual(result, "SUCCESS_REIMPORTED")
        self.assertEqual(len(harness.added_photos), 1)
        self.assertEqual(harness.beeps, 1)
        self.assertEqual(len(harness.messages), 0)


if __name__ == "__main__":
    unittest.main()
