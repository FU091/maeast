conda run -n mae_v2 python -m py_compile train.py
if ($LASTEXITCODE -eq 0) { Write-Host "train.py OK" } else { Write-Host "train.py FAIL" }

conda run -n mae_v2 python -m py_compile dataset.py
if ($LASTEXITCODE -eq 0) { Write-Host "dataset.py OK" } else { Write-Host "dataset.py FAIL" }

conda run -n mae_v2 python -m py_compile config.py
if ($LASTEXITCODE -eq 0) { Write-Host "config.py OK" } else { Write-Host "config.py FAIL" }
