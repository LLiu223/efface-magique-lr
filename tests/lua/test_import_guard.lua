-- Executes plugin/ai_eraser.lrplugin/ImportGuard.lua with a real Lua 5.1 / LuaJIT interpreter.
-- Same cases as the Python port in tests/test_plugin.py. Exits non-zero on any failure.
-- Long-bracket strings ([=[ ]=]) are raw: backslashes are literal.

local scriptDir = (arg and arg[0] and arg[0]:match('^(.*)[/\\]')) or '.'
package.path = scriptDir .. '/../../plugin/ai_eraser.lrplugin/?.lua;' .. package.path

local ImportGuard = require 'ImportGuard'

local failures = 0

local function check(name, got, want)
    if got ~= want then
        failures = failures + 1
        io.stderr:write(string.format('FAIL %s: got %s, want %s\n', name, tostring(got), tostring(want)))
    end
end

local TEMP = [=[C:\Proj\.tmp]=]
local ORIG = [=[D:\Photos\IMG_42.CR3]=]
local SESSION = { ['42'] = ORIG }

local function item(path, orig, id)
    return { path = path, original_path = orig, photo_id = id }
end

local function accepts(name, it, tempDir)
    local edited, orig = ImportGuard.validate(it, tempDir or TEMP, SESSION)
    check(name .. ': edited', edited, it.path)
    check(name .. ': orig', orig, ORIG)
end

local function rejects(name, it, tempDir)
    local edited, reason = ImportGuard.validate(it, tempDir or TEMP, SESSION)
    check(name .. ': rejected', edited, nil)
    check(name .. ': has reason', type(reason), 'string')
end

-- Accepted
accepts('exact', item([=[C:\Proj\.tmp\live_42.tif]=], ORIG, '42'))
accepts('no original_path', item([=[C:\Proj\.tmp\live_42.tif]=], nil, '42'))
accepts('forward slashes + case', item([=[c:/proj/.TMP/live_42.tif]=], [=[d:/photos/img_42.cr3]=], '42'))
accepts('tiff ext', item([=[C:\Proj\.tmp\live_42.tiff]=], ORIG, '42'))
accepts('temp dir trailing sep', item([=[C:\Proj\.tmp\live_42.tif]=], ORIG, '42'), [=[C:\Proj\.tmp\]=])
accepts('posix', item('/home/u/proj/.tmp/live_42.tif', ORIG, '42'), '/home/u/proj/.tmp')

-- Rejected
rejects('dotdot', item([=[C:\Proj\.tmp\..\.tmp\live_42.tif]=], ORIG, '42'))
rejects('dotdot escape', item([=[C:\Proj\.tmp\..\..\Photos\live_42.tif]=], ORIG, '42'))
rejects('relative', item([=[.tmp\live_42.tif]=], ORIG, '42'))
rejects('absolute outside', item([=[D:\Photos\live_42.tif]=], ORIG, '42'))
rejects('sibling prefix', item([=[C:\Proj\.tmp2\live_42.tif]=], ORIG, '42'))
rejects('subdir', item([=[C:\Proj\.tmp\sub\live_42.tif]=], ORIG, '42'))
rejects('wrong leaf', item([=[C:\Proj\.tmp\IMG_42_ai_edit.tif]=], ORIG, '42'))
rejects('wrong ext', item([=[C:\Proj\.tmp\live_42.jpg]=], ORIG, '42'))
rejects('trailing sep on path', item([=[C:\Proj\.tmp\live_42.tif\]=], ORIG, '42'))
rejects('unknown id', item([=[C:\Proj\.tmp\live_7.tif]=], ORIG, '7'))
rejects('id mismatch', item([=[C:\Proj\.tmp\live_42.tif]=], ORIG, '7'))
rejects('original_path mismatch', item([=[C:\Proj\.tmp\live_42.tif]=], [=[C:\Windows\x.dll]=], '42'))
rejects('missing photo_id', item([=[C:\Proj\.tmp\live_42.tif]=], ORIG, nil))
rejects('numeric photo_id', item([=[C:\Proj\.tmp\live_42.tif]=], ORIG, 42))
rejects('missing path', item(nil, ORIG, '42'))
rejects('non-string path', item(5, ORIG, '42'))
rejects('posix case differs', item('/home/u/proj/.TMP/live_42.tif', ORIG, '42'), '/home/u/proj/.tmp')

if failures > 0 then
    io.stderr:write(string.format('%d failure(s)\n', failures))
    os.exit(1)
end
print('test_import_guard.lua: all passed')
