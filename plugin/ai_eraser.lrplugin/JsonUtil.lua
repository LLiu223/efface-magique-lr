--[[----------------------------------------------------------------------------
JsonUtil.lua
Pure JSON helpers for LiveBridge.lua (no Lightroom SDK), so a plain Lua 5.1
interpreter can test them (tests/lua/test_json_util.lua).
------------------------------------------------------------------------------]]

local function extractJsonField(jsonStr, field)
    if not jsonStr or not field then return nil end
    local pattern = '"' .. field .. '"%s*:%s*"([^"]+)"'
    local val = jsonStr:match(pattern)
    if val then return val end
    local numPattern = '"' .. field .. '"%s*:%s*(%d+)'
    return jsonStr:match(numPattern)
end

local function parsePendingImports(jsonStr)
    local imports = {}
    if not jsonStr then return imports end

    -- Greedy: paths may contain ']'. Relies on "imports" being the LAST key in the response.
    local arrayContent = jsonStr:match('"imports"%s*:%s*%[(.*)%]')
    if not arrayContent or arrayContent:match('^%s*$') then
        return imports
    end

    -- Hide braces inside quoted strings ('"' is illegal in Windows paths, so strings are "[^"]*"),
    -- leaving only structural braces for the item split.
    local masked = arrayContent:gsub('"[^"]*"', function(s)
        return (s:gsub('{', '\1'):gsub('}', '\2'))
    end)

    for maskedItem in masked:gmatch('{([^}]+)}') do
        local itemStr = maskedItem:gsub('\1', '{'):gsub('\2', '}')
        local path = extractJsonField(itemStr, "path")
        local origPath = extractJsonField(itemStr, "original_path")
        local photoId = extractJsonField(itemStr, "photo_id")
        if path then
            -- Unescape backslashes if escaped in JSON
            path = (path:gsub('\\\\', '\\'))
            if origPath then
                origPath = (origPath:gsub('\\\\', '\\'))
            end
            table.insert(imports, {
                path = path,
                original_path = origPath,
                photo_id = photoId,
            })
        end
    end
    return imports
end

-- Escape a string for a JSON string literal: '\' first, then '"'.
local function jsonEscape(s)
    return (tostring(s):gsub('\\', '\\\\'):gsub('"', '\\"'))
end

local JsonUtil = {
    extractJsonField = extractJsonField,
    parsePendingImports = parsePendingImports,
    jsonEscape = jsonEscape,
}

return JsonUtil
