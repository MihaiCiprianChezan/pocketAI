pip list --outdated | ForEach-Object {
   $package = ($_ -split '\s+')[0]
   pip install --upgrade $package
}

pip check