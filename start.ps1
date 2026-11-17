$ErrorActionPreference = 'Stop'

$projectRoot = $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$pdfDir = Join-Path $projectRoot 'download_corpus\rag_corpus\pdfs'
$chunkDir = Join-Path $projectRoot 'preprocessing\data\chunks'

if (-not (Test-Path $python)) {
    throw "Project Python environment not found at $python. Create it with uv sync before running the pipeline."
}

function Invoke-PipelineScript {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RelativePath,
        [string[]]$ScriptArguments = @()
    )

    $scriptPath = Join-Path $projectRoot $RelativePath
    Push-Location $projectRoot
    try {
        & $python $scriptPath @ScriptArguments
        if ($LASTEXITCODE -ne 0) {
            throw "$RelativePath failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
}

Write-Host 'Step 1/4: Downloading PDFs and extracting text...'
Invoke-PipelineScript 'download_corpus\download_pdfs.py' $args

Write-Host 'Step 2/4: Chunking the corpus (GROBID is needed only when TEI files are missing)...'
Invoke-PipelineScript 'preprocessing\chunking.py'

$pdfIds = @(
    Get-ChildItem -LiteralPath $pdfDir -Filter '*.pdf' -File -ErrorAction SilentlyContinue |
        ForEach-Object { $_.BaseName }
)
$chunkFiles = @(Get-ChildItem -LiteralPath $chunkDir -Filter '*_chunks.json' -File -ErrorAction SilentlyContinue)
$expectedIds = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
foreach ($id in $pdfIds) { [void]$expectedIds.Add($id) }
$actualIds = @($chunkFiles | ForEach-Object { $_.BaseName -replace '_chunks$', '' })
$unexpectedIds = @($actualIds | Where-Object { -not $expectedIds.Contains($_) })
$missingIds = @($pdfIds | Where-Object { $_ -notin $actualIds })
if ($pdfIds.Count -eq 0 -or $unexpectedIds.Count -gt 0 -or $missingIds.Count -gt 0) {
    throw "Chunk files must match the downloaded PDF corpus. Missing: $($missingIds -join ', '); unexpected: $($unexpectedIds -join ', ')."
}

Write-Host "Step 3/4: Generating or resuming Qwen embeddings for $($pdfIds.Count) papers..."
Invoke-PipelineScript 'preprocessing\embeddings.py'

Write-Host 'Step 4/4: Validating the embedding cache and ingesting it into Qdrant...'
Invoke-PipelineScript 'retrieval\qdrant_setup.py'

Write-Host 'Corpus ingestion finished. Query-time retrieval is not yet implemented.'
