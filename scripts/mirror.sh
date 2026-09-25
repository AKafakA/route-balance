# The second location, shared by run_cell.sh and run_scope.sh (sourced). A destination is rsync syntax: a local
# path or host:path. Nothing here ever deletes or overwrites a different run on the mirror.

fresh() {  # <source dir> <dest dir>: a copy would only create files (the destination is absent, partial or identical)
  local plan
  plan=$(rsync -a --checksum --dry-run --itemize-changes --delete "$1/" "$2/") || return 1
  ! grep -v -e '^$' -e '^created directory ' -e '+++++++++' <<< "$plan" | grep -q .
}

mirror() {  # <source dir> <dest dir>: copy, then a dry run must succeed and list no difference or extra file
  local diff
  rsync -a --checksum "$1/" "$2/" || return 1
  diff=$(rsync -a --checksum --dry-run --itemize-changes --delete "$1/" "$2/") || return 1
  [ -z "$diff" ]
}

mirror_cell() {  # <cells root> <mirror root> <cell>: never over another run of the cell, then identical
  fresh "$1/$3" "$2/$3" || { echo "$2/$3 holds another run of $3: move it aside (retire it there too) first" >&2; return 1; }
  mirror "$1/$3" "$2/$3" || { echo "the mirror of $3 at $2/$3 is not identical" >&2; return 1; }
}
