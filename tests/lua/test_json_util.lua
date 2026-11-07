-- Executes plugin/ai_eraser.lrplugin/JsonUtil.lua with a real Lua 5.1 / LuaJIT interpreter.
-- Same cases as the Python ports in tests/test_plugin.py. Exits non-zero on any failure.
-- Long-bracket strings ([=[ ]=]) are raw: no escape processing, so backslashes are literal.

local scriptDir = (arg and arg[0] and arg[0]:match('^(.*)[/\\]')) or '.'
package.path = scriptDir .. '/../../plugin/ai_eraser.lrplugin/?.lua;' .. package.path

local JsonUtil = require 'JsonUtil'

local failures = 0

local function check(name, got, want)
    if got ~= want then
        failures = failures + 1
        io.stderr:write(string.format('FAIL %s: got %s, want %s\n', name, tostring(got), tostring(want)))
    end
end

-- 1. Empty imports array
local r = JsonUtil.parsePendingImports([=[{"status": "ok", "count": 0, "imports": []}]=])
check('empty: count', #r, 0)

-- 2. Single item with Windows paths
r = JsonUtil.parsePendingImports([=[{"status": "ok", "count": 1, "imports": [{"path": "C:\\tmp\\live_edit.tif", "original_path": "C:\\photos\\raw.cr3", "photo_id": "42"}]}]=])
check('single: count', #r, 1)
if r[1] then
    check('single: path', r[1].path, [=[C:\tmp\live_edit.tif]=])
    check('single: original_path', r[1].original_path, [=[C:\photos\raw.cr3]=])
    check('single: photo_id', r[1].photo_id, '42')
end

-- 3. Brackets/braces inside paths, multiple items, null fields
r = JsonUtil.parsePendingImports([=[{"status": "ok", "count": 2, "imports": [{"path": "C:\\Été [2024]\\a}.tif", "original_path": "C:\\x{y}\\b].cr3", "photo_id": null}, {"path": "C:\\c.tif", "original_path": null, "photo_id": "7"}]}]=])
check('tricky: count', #r, 2)
if r[1] then
    check('tricky[1]: path', r[1].path, [=[C:\Été [2024]\a}.tif]=])
    check('tricky[1]: original_path', r[1].original_path, [=[C:\x{y}\b].cr3]=])
    check('tricky[1]: photo_id', r[1].photo_id, nil)
end
if r[2] then
    check('tricky[2]: path', r[2].path, [=[C:\c.tif]=])
    check('tricky[2]: original_path', r[2].original_path, nil)
    check('tricky[2]: photo_id', r[2].photo_id, '7')
end

-- 4. jsonEscape: '\' first, then '"'
check('escape: a"b\\c', JsonUtil.jsonEscape([=[a"b\c]=]), [=[a\"b\\c]=])
check('escape: windows path', JsonUtil.jsonEscape([=[C:\Photos\say "hi"\x.tif]=]), [=[C:\\Photos\\say \"hi\"\\x.tif]=])
check('escape: \\"', JsonUtil.jsonEscape([=[\"]=]), [=[\\\"]=])
check('escape: empty', JsonUtil.jsonEscape(''), '')

-- 5. Preview-first sync bodies (job 019)
check('preview leaf: number id', JsonUtil.previewLeafName(12345), 'preview_12345.jpg')
check('preview leaf: string id', JsonUtil.previewLeafName('7'), 'preview_7.jpg')
check('preview body', JsonUtil.previewBody([=[C:\Été "x"\.tmp\preview_7.jpg]=], 7),
    [=[{"path": "C:\\Été \"x\"\\.tmp\\preview_7.jpg", "photo_id": "7"}]=])
check('export failed body', JsonUtil.exportFailedBody(7), [=[{"photo_id": "7", "export_failed": true}]=])
check('select body: export_ms rounded', JsonUtil.selectBody([=[C:\t\live_7.tif]=], 7, [=[D:\a "b".cr3]=], 'a "b"', 1234.6),
    [=[{"path": "C:\\t\\live_7.tif", "photo_id": "7", "original_path": "D:\\a \"b\".cr3", "title": "a \"b\"", "export_ms": 1235}]=])
check('select body: cache hit 0', JsonUtil.selectBody('p', '1', 'o', 't', 0):match('"export_ms": (%-?%d+)}$'), '0')
check('select body: negative clamps to 0', JsonUtil.selectBody('p', '1', 'o', 't', -5):match('"export_ms": (%-?%d+)}$'), '0')

if failures > 0 then
    io.stderr:write(string.format('%d failure(s)\n', failures))
    os.exit(1)
end
print('test_json_util.lua: all passed')
