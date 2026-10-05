$ErrorActionPreference = 'Stop'

$projectRoot = $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$evalFile = Join-Path $projectRoot 'evals\embeddings\evals_v2.jsonl'
$chunkDir = Join-Path $projectRoot 'preprocessing\data\chunks'

if (-not (Test-Path $python)) {
    throw "Project Python environment not found at $python. Create it with uv sync before running the pipeline."
}

if (-not (Test-Path $evalFile)) {
    throw "V2 evaluation dataset not found at $evalFile."
}

$dataset = Get-Content -LiteralPath $evalFile -TotalCount 1 | ConvertFrom-Json
if ($dataset.record_type -ne 'dataset_metadata' -or -not $dataset.candidate_paper_ids) {
    throw "$evalFile must start with dataset metadata containing candidate_paper_ids."
}
$candidateIds = @($dataset.candidate_paper_ids | ForEach-Object { [string]$_ })

Write-Host 'Step 1/3: Downloading PDFs and extracting text...'
Push-Location $projectRoot
try {
    & $python (Join-Path $projectRoot 'download_corpus\download_pdfs.py') @args
    if ($LASTEXITCODE -ne 0) {
        throw "download_pdfs.py failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

Write-Host "Step 2/3: Chunking $($candidateIds.Count) V2 candidate papers (GROBID is needed only when TEI files are missing)..."
Push-Location $projectRoot
try {
    & $python (Join-Path $projectRoot 'preprocessing\chunking.py') --ids $candidateIds
    if ($LASTEXITCODE -ne 0) {
        throw "chunking.py failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

$expectedIds = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
foreach ($id in $candidateIds) { [void]$expectedIds.Add($id) }
$actualIds = @(
    Get-ChildItem -LiteralPath $chunkDir -Filter '*_chunks.json' -File -ErrorAction SilentlyContinue |
        ForEach-Object { $_.BaseName -replace '_chunks$', '' }
)
$unexpectedIds = @($actualIds | Where-Object { -not $expectedIds.Contains($_) })
$missingIds = @($candidateIds | Where-Object { $_ -notin $actualIds })
if ($unexpectedIds.Count -gt 0 -or $missingIds.Count -gt 0) {
    throw "Chunk files must match the V2 candidate list. Missing: $($missingIds -join ', '); unexpected: $($unexpectedIds -join ', ')."
}

Write-Host 'Step 3/3: Running V2 embedding model comparison (requires a CUDA GPU)...'
Push-Location $projectRoot
try {
    & $python (Join-Path $projectRoot 'evals\embeddings\compare_embeddings_v2.py')
    if ($LASTEXITCODE -ne 0) {
        throw "compare_embeddings_v2.py failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

Write-Host 'Pipeline finished.'
