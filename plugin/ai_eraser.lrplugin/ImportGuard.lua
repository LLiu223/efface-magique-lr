--[[----------------------------------------------------------------------------
ImportGuard.lua
Pure validator for /api/pending_imports items (no Lightroom SDK), so a plain
Lua 5.1 interpreter can test it (tests/lua/test_import_guard.lua).
An item is trusted only if its path is <tempDir>/live_<id>.tif(f) for an id the
plugin exported this session; the original folder comes from the plugin's own
record, never from the server.
------------------------------------------------------------------------------]]

-- Forward slashes, no trailing separator; lower-cased for Windows-style paths.
local function normalize(p, windows)
    p = p:gsub('\\', '/')
    p = p:gsub('(.)/+$', '%1')
    if windows then p = p:lower() end
    return p
end

local function isWindowsStyle(p)
    return p:match('^%a:') ~= nil or p:find('\\', 1, true) ~= nil
end

-- Returns editedPath, origPath when trusted; otherwise nil, reason.
local function validate(item, tempDir, sessionPhotos)
    if type(item) ~= 'table' then return nil, 'item is not a table' end
    local path, photoId = item.path, item.photo_id
    if type(path) ~= 'string' or path == '' then return nil, 'missing path' end
    if type(photoId) ~= 'string' or photoId == '' then return nil, 'missing photo_id' end
    if type(tempDir) ~= 'string' or tempDir == '' then return nil, 'no temp dir' end

    local windows = isWindowsStyle(tempDir) or isWindowsStyle(path)
    local p = normalize(path, windows)
    if ('/' .. p .. '/'):find('/%.%./') then return nil, 'path contains ..' end
    if not (p:match('^/') or p:match('^%a:/')) then return nil, 'path is relative' end

    local parent = p:match('^(.*)/[^/]+$')
    -- Exact parent match: a prefix test would accept <tempDir>2/...
    if not parent or parent ~= normalize(tempDir, windows) then
        return nil, 'path is outside the temp dir'
    end

    -- Leaf from the raw path so the id keeps its case.
    local id = (path:match('([^/\\]+)$') or ''):match('^live_(.+)%.tiff?$')
    if not id then return nil, 'file name is not live_<id>.tif' end
    if photoId ~= id then return nil, 'photo_id does not match file name' end

    local origPath = sessionPhotos and sessionPhotos[id]
    if type(origPath) ~= 'string' then return nil, 'photo id not exported this session' end

    local serverOrig = item.original_path
    if serverOrig ~= nil and (type(serverOrig) ~= 'string'
        or normalize(serverOrig, windows) ~= normalize(origPath, windows)) then
        return nil, 'original_path does not match the exported photo'
    end

    return path, origPath
end

local ImportGuard = {
    validate = validate,
}

return ImportGuard
