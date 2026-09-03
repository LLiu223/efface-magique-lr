--[[----------------------------------------------------------------------------
GenerativeEraser.lua
Efface Magique LR - Adobe Lightroom Classic Plugin
Main workflow: Export selected photo -> Invoke AI Companion -> Reimport & Stack
------------------------------------------------------------------------------]]

local LrApplication = import 'LrApplication'
local LrDialogs = import 'LrDialogs'
local LrExportSession = import 'LrExportSession'
local LrFileUtils = import 'LrFileUtils'
local LrPathUtils = import 'LrPathUtils'
local LrProgressScope = import 'LrProgressScope'
local LrTasks = import 'LrTasks'
local LrShell = import 'LrShell'

local PluginUtils = require 'PluginUtils'

local function runGenerativeEraser()
    local catalog = LrApplication.activeCatalog()
    local targetPhoto = catalog:getTargetPhoto()

    if not targetPhoto then
        LrDialogs.message(
            "AI Generative Eraser",
            "Please select a photo in Lightroom before running AI Generative Eraser.",
            "info"
        )
        return
    end

    local progressScope = LrProgressScope({
        title = "AI Generative Eraser: Preparing photo export...",
        caption = "Exporting 16-bit high-resolution TIFF...",
    })
    progressScope:setCancelable(true)

    -- Determine export temporary directory
    local root = PluginUtils.getProjectRoot()
    local tempDir = LrPathUtils.child(root, ".tmp")
    if not LrFileUtils.exists(tempDir) then
        LrFileUtils.createAllDirectories(tempDir)
    end

    local originalPath = targetPhoto:getRawMetadata('path')
    local baseName = LrPathUtils.removeExtension(LrPathUtils.leafName(originalPath))
    local timestamp = os.date("%Y%m%d_%H%M%S")
    local exportedFileName = string.format("%s_ai_edit_%s.tif", baseName, timestamp)
    local exportedFilePath = LrPathUtils.child(tempDir, exportedFileName)

    -- Configure 16-bit TIFF export session preserving color fidelity
    local exportSettings = {
        LR_export_destinationType = 'specificFolder',
        LR_export_destinationPathPrefix = tempDir,
        LR_export_useSubfolder = false,
        LR_collisionHandling = 'overwrite',
        LR_format = 'TIFF',
        LR_export_bitDepth = 16,
        LR_export_colorSpace = 'ProPhotoRGB',
        LR_export_compressionMethod = 'ZIP',
        LR_size_doConstrain = false,
        LR_outputSharpeningOn = false,
        LR_metadata_exportExif = true,
        LR_metadata_exportIptc = true,
        LR_reimportExportedPhoto = false,
        LR_renamingTokensOn = true,
        LR_tokens = string.format("%s_ai_edit_%s", baseName, timestamp),
    }

    local exportSession = LrExportSession({
        photosToExport = { targetPhoto },
        exportSettings = exportSettings,
    })

    exportSession:doExportOnCurrentTask()

    if progressScope:isCanceled() then
        progressScope:done()
        return
    end

    -- Verify that the exported file exists
    if LrFileUtils.exists(exportedFilePath) ~= "file" then
        -- Check if it was exported with a slightly different extension/name
        local files = LrFileUtils.directoryEntries(tempDir)
        local found = false
        if type(files) == "function" then
            for file in files do
                if string.find(file, baseName .. "_ai_edit_" .. timestamp) then
                    exportedFilePath = file
                    found = true
                    break
                end
            end
        elseif type(files) == "table" then
            for _, file in ipairs(files) do
                if string.find(file, baseName .. "_ai_edit_" .. timestamp) then
                    exportedFilePath = file
                    found = true
                    break
                end
            end
        end

        if not found then
            progressScope:done()
            LrDialogs.message(
                "AI Generative Eraser Error",
                "Failed to export high-resolution TIFF to workspace:\n" .. exportedFilePath,
                "critical"
            )
            return
        end
    end

    progressScope:setPortionComplete(0.3, 1.0)
    progressScope:setCaption("Launching local AI Companion app...")

    -- Resolve Python and Companion script paths
    local pythonExe = PluginUtils.getPythonExecutable()
    local companionScript = PluginUtils.getCompanionAppPath()

    if LrFileUtils.exists(companionScript) ~= "file" then
        progressScope:done()
        LrDialogs.message(
            "AI Generative Eraser Configuration Error",
            "Could not locate the Python companion app at:\n" .. companionScript ..
            "\n\nPlease ensure you have installed the project dependencies.",
            "critical"
        )
        return
    end

    -- Command to execute companion app
    local command = string.format(
        '%s %s --input %s --output %s',
        PluginUtils.quoteArg(pythonExe),
        PluginUtils.quoteArg(companionScript),
        PluginUtils.quoteArg(exportedFilePath),
        PluginUtils.quoteArg(exportedFilePath)
    )

    if WIN_ENV then
        -- Wrap whole command in outer quotes so cmd.exe /c preserves argument quotes
        command = '"' .. command .. '"'
    end

    progressScope:setCaption("AI Companion is running. Complete your edits in the app...")
    progressScope:setPortionComplete(0.5, 1.0)

    -- Execute companion app synchronously on the current task
    local exitCode = LrTasks.execute(command)

    if progressScope:isCanceled() then
        progressScope:done()
        return
    end

    if exitCode == 0 then
        progressScope:setPortionComplete(0.8, 1.0)
        progressScope:setCaption("Re-importing edited photo into Lightroom Catalog...")

        -- Move edited photo or import directly into catalog next to original
        local originalFolder = LrPathUtils.parent(originalPath)
        local finalDestPath = LrPathUtils.child(originalFolder, exportedFileName)

        -- Copy or move edited TIFF to original photo directory
        if LrFileUtils.exists(exportedFilePath) == "file" then
            if LrFileUtils.exists(finalDestPath) then
                LrFileUtils.delete(finalDestPath)
            end
            local success = LrFileUtils.copy(exportedFilePath, finalDestPath)
            if success then
                -- Clean up temporary export file
                LrFileUtils.delete(exportedFilePath)
            else
                -- Fallback to exportedFilePath if copying fails (e.g. read-only folder)
                finalDestPath = exportedFilePath
            end
        end

        catalog:withWriteAccessDo("Import AI Generative Eraser Edit", function()
            local newPhoto = catalog:addPhoto(finalDestPath, targetPhoto, 'above')
            if newPhoto then
                -- Stack adjacent to original and set the newly edited photo as active in Lightroom
                catalog:setSelectedPhotos(newPhoto, { newPhoto })
            end
        end)

        progressScope:setPortionComplete(1.0, 1.0)
        progressScope:done()

        if LrDialogs.showBeep then
            LrDialogs.showBeep()
        end
    else
        progressScope:done()
        -- Clean up temporary export file
        if LrFileUtils.exists(exportedFilePath) == "file" then
            LrFileUtils.delete(exportedFilePath)
        end
        if exitCode ~= -1 and exitCode ~= 130 then -- user didn't simply cancel
            LrDialogs.message(
                "AI Generative Eraser",
                string.format("AI Companion closed with exit code %s. No edits were imported.", tostring(exitCode)),
                "info"
            )
        end
    end
end

-- Run within an asynchronous Lightroom task
LrTasks.startAsyncTask(runGenerativeEraser)
