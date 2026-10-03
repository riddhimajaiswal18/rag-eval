$ErrorActionPreference = 'Stop'

$projectRoot = $PSScriptRoot
$python = (Get-Command python -ErrorAction Stop).Source

Write-Host 'Step 1/2: Downloading PDFs and extracting text...'
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

Write-Host 'Step 2/2: Chunking papers (GROBID must be running at http://localhost:8070)...'
Push-Location $projectRoot
try {
    & $python (Join-Path $projectRoot 'preprocessing\chunking.py')
    if ($LASTEXITCODE -ne 0) {
        throw "chunking.py failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

if (Test-Path (Join-Path $projectRoot 'evals\embeddings\eval.json')) {
    Write-Host 'Step 3/3: Running embedding model comparison (requires a CUDA GPU)...'
    Push-Location $projectRoot
    try {
        & $python (Join-Path $projectRoot 'evals\embeddings\compare_embeddings.py')
        if ($LASTEXITCODE -ne 0) {
            throw "compare_embeddings.py failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
}
else {
    Write-Host 'Embedding comparison skipped: evals\embeddings\eval.json is missing.'
    Write-Host 'Create evals\embeddings\eval.json with question/chunk_id pairs, then run .\start.ps1 again.'
}

Write-Host 'Pipeline finished.'
