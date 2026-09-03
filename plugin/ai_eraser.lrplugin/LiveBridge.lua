--[[----------------------------------------------------------------------------
LiveBridge.lua
Efface Magique LR - Adobe Lightroom Classic Plugin
Seamless Live Sync Bridge: Observes Lightroom photo selection, debounces exports,
syncs active photo to Companion window, and auto-imports completed inpaintings.
------------------------------------------------------------------------------]]

local LrApplication = import 'LrApplication'
local LrDialogs = import 'LrDialogs'
local LrExportSession = import 'LrExportSession'
local LrFileUtils = import 'LrFileUtils'
local LrHttp = import 'LrHttp'
local LrPathUtils = import 'LrPathUtils'
local LrProgressScope = import 'LrProgressScope'
local LrTasks = import 'LrTasks'

local PluginUtils = require 'PluginUtils'

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

    -- Extract array contents inside "imports": [ ... ]
    local arrayContent = jsonStr:match('"imports"%s*:%s*%[(.-)%]')
    if not arrayContent or arrayContent:match('^%s*$') then
        return imports
    end

    for itemStr in arrayContent:gmatch('{([^}]+)}') do
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

local function runLiveBridge()
    local catalog = LrApplication.activeCatalog()

    local progressScope = LrProgressScope({
        title = "⚡ AI Generative Eraser: Live Sync",
        caption = "Connecting to companion window...",
    })
    progressScope:setCancelable(true)

    local root = PluginUtils.getProjectRoot()
    local tempDir = LrPathUtils.child(root, ".tmp")
    if not LrFileUtils.exists(tempDir) then
        LrFileUtils.createAllDirectories(tempDir)
    end

    local port = PluginUtils.getLiveBridgePort()
    local pingUrl = string.format("http://127.0.0.1:%d/api/ping", port)
    local isConnected = false

    -- Check if companion is already running
    local response = LrHttp.get(pingUrl)
    if response and string.find(response, "EffaceMagique") then
        isConnected = true
    end

    -- If not running, launch companion detached in live mode
    if not isConnected then
        progressScope:setCaption("Launching local AI Companion in live mode...")
        local pythonExe = PluginUtils.getPythonExecutable()
        local companionScript = PluginUtils.getCompanionAppPath()

        if LrFileUtils.exists(companionScript) ~= "file" then
            progressScope:done()
            LrDialogs.message(
                "AI Generative Eraser Error",
                "Could not locate companion app at:\n" .. companionScript,
                "critical"
            )
            return
        end

        local launchCmd = string.format(
            '%s %s --live',
            PluginUtils.quoteArg(pythonExe),
            PluginUtils.quoteArg(companionScript)
        )
        PluginUtils.launchBackgroundProcess(launchCmd)

        -- Poll until companion responds to ping
        local attempts = 0
        while not isConnected and attempts < 25 and not progressScope:isCanceled() do
            LrTasks.sleep(0.4)
            attempts = attempts + 1
            port = PluginUtils.getLiveBridgePort()
            pingUrl = string.format("http://127.0.0.1:%d/api/ping", port)
            response = LrHttp.get(pingUrl)
            if response and string.find(response, "EffaceMagique") then
                isConnected = true
                break
            end
        end
    end

    if not isConnected then
        progressScope:done()
        if not progressScope:isCanceled() then
            LrDialogs.message(
                "AI Generative Eraser",
                "Could not connect to Live Companion window.",
                "warning"
            )
        end
        return
    end

    progressScope:setPortionComplete(1.0, 1.0)
    progressScope:setCaption("Live Sync Active. Navigating photos updates companion.")

    -- Focus companion window on connect
    local focusUrl = string.format("http://127.0.0.1:%d/api/focus", port)
    LrHttp.post(focusUrl, "{}", { { field = "Content-Type", value = "application/json" } })

    -- Main selection & import polling loop
    local lastPhotoId = nil
    local pendingPhoto = nil
    local pendingTime = 0
    local disconnectedCount = 0
    local DEBOUNCE_DELAY = 0.08 -- 80ms snappy debounce

    local headers = {
        { field = "Content-Type", value = "application/json" }
    }

    while not progressScope:isCanceled() do
        LrTasks.sleep(0.03) -- 30ms high-frequency polling

        port = PluginUtils.getLiveBridgePort()
        local targetPhoto = catalog:getTargetPhoto()
        local targetId = targetPhoto and targetPhoto.localIdentifier or nil

        -- 1. Check selection change with snappy debounce
        if targetId ~= lastPhotoId then
            if targetId ~= (pendingPhoto and pendingPhoto.localIdentifier) then
                pendingPhoto = targetPhoto
                pendingTime = os.clock()
            else
                if (os.clock() - pendingTime) >= DEBOUNCE_DELAY then
                    lastPhotoId = targetId
                    local activePhoto = pendingPhoto
                    pendingPhoto = nil

                    if activePhoto then
                        local origPath = activePhoto:getRawMetadata('path')
                        local baseName = LrPathUtils.removeExtension(LrPathUtils.leafName(origPath))
                        local exportName = string.format("live_%s.tif", tostring(targetId))
                        local exportPath = LrPathUtils.child(tempDir, exportName)

                        local selectUrl = string.format("http://127.0.0.1:%d/api/select", port)
                        -- Escape Windows backslashes for JSON
                        local jsonPath = (exportPath:gsub('\\', '\\\\'))
                        local jsonOrig = (origPath:gsub('\\', '\\\\'))
                        local jsonBody = string.format(
                            '{"path": "%s", "photo_id": "%s", "original_path": "%s", "title": "%s"}',
                            jsonPath, tostring(targetId), jsonOrig, baseName
                        )

                        -- Instant Cache Hit: if already exported in this session, swap immediately in 0ms!
                        if LrFileUtils.exists(exportPath) == "file" then
                            LrHttp.post(selectUrl, jsonBody, headers)
                        else
                            -- Ultra-fast uncompressed 16-bit TIFF export (no CPU-heavy ZIP compression)
                            local exportSettings = {
                                LR_export_destinationType = 'specificFolder',
                                LR_export_destinationPathPrefix = tempDir,
                                LR_export_useSubfolder = false,
                                LR_collisionHandling = 'overwrite',
                                LR_format = 'TIFF',
                                LR_export_bitDepth = 16,
                                LR_export_colorSpace = 'ProPhotoRGB',
                                LR_export_compressionMethod = 'None',
                                LR_size_doConstrain = false,
                                LR_outputSharpeningOn = false,
                                LR_metadata_exportExif = true,
                                LR_reimportExportedPhoto = false,
                                LR_renamingTokensOn = true,
                                LR_tokens = string.format("live_%s", tostring(targetId)),
                            }

                            local exportSession = LrExportSession({
                                photosToExport = { activePhoto },
                                exportSettings = exportSettings,
                            })

                            exportSession:doExportOnCurrentTask()

                            if LrFileUtils.exists(exportPath) == "file" then
                                LrHttp.post(selectUrl, jsonBody, headers)
                            end
                        end
                    end
                end
            end
        end

        -- 2. Check for completed edits waiting to be imported into Lightroom
        local importsUrl = string.format("http://127.0.0.1:%d/api/pending_imports", port)
        local importsResp = LrHttp.get(importsUrl)
        if importsResp then
            disconnectedCount = 0
            if string.find(importsResp, '"imports"') then
                local items = parsePendingImports(importsResp)
                for _, item in ipairs(items) do
                    local editedPath = item.path
                    if editedPath and LrFileUtils.exists(editedPath) == "file" then
                        local targetForImport = targetPhoto
                        if item.photo_id then
                            -- Attempt to find matching photo object
                            local allPhotos = catalog:getAllPhotos()
                            for _, p in ipairs(allPhotos) do
                                if tostring(p.localIdentifier) == tostring(item.photo_id) then
                                    targetForImport = p
                                    break
                                end
                            end
                        end

                        -- Move/copy edited TIFF adjacent to original photo folder
                        local origPath = item.original_path or (targetForImport and targetForImport:getRawMetadata('path'))
                        local finalDestPath = editedPath

                        if origPath then
                            local origFolder = LrPathUtils.parent(origPath)
                            local leaf = LrPathUtils.leafName(editedPath)
                            finalDestPath = LrPathUtils.child(origFolder, leaf)

                            if LrFileUtils.exists(finalDestPath) then
                                LrFileUtils.delete(finalDestPath)
                            end
                            local copySuccess = LrFileUtils.copy(editedPath, finalDestPath)
                            if copySuccess then
                                LrFileUtils.delete(editedPath)
                            else
                                finalDestPath = editedPath
                            end
                        end

                        catalog:withWriteAccessDo("Import AI Generative Eraser Edit", function()
                            local newPhoto = catalog:addPhoto(finalDestPath, targetForImport, 'above')
                            if newPhoto then
                                catalog:setSelectedPhotos(newPhoto, { newPhoto })
                            end
                        end)

                        if LrDialogs.showBeep then
                            LrDialogs.showBeep()
                        end

                        -- Acknowledge import to companion
                        local doneUrl = string.format("http://127.0.0.1:%d/api/import_done", port)
                        local jsonDoneBody = string.format('{"path": "%s"}', (editedPath:gsub('\\', '\\\\')))
                        LrHttp.post(doneUrl, jsonDoneBody, headers)
                    end
                end
            end
        else
            disconnectedCount = disconnectedCount + 1
            -- If companion closed or connection lost for ~150ms (5 ticks)
            if disconnectedCount >= 5 then
                local pingUrl = string.format("http://127.0.0.1:%d/api/ping", port)
                local pingResp = LrHttp.get(pingUrl)
                if not pingResp or not string.find(pingResp, "EffaceMagique") then
                    -- Companion window has been closed! Terminate Live Sync cleanly.
                    break
                else
                    disconnectedCount = 0
                end
            end
        end
    end

    -- If user clicked cancel in Lightroom activity center, request companion window to close
    if progressScope:isCanceled() then
        local closeUrl = string.format("http://127.0.0.1:%d/api/close", port)
        LrHttp.post(closeUrl, "{}", headers)
    end

    progressScope:done()
end

LrTasks.startAsyncTask(runLiveBridge)
