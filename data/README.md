# data/

Some subject folders are physically moved locally by preprocessing/verify_setup.py:
- data/Subjects/excluded/     FD motion + missing-file exclusions (sub-004, sub-009, sub-012, sub-018, sub-019, sub-020, sub-032)
- data/Subjects/junk_removed/ syncthing duplicate folders ("sub-XXX N")

Both dirs are gitignored (blanket * in data/.gitignore). The original pre-exclusion data
is unchanged in git history at data/Subjects/<sid>/ (on origin/main). Restore with:
  git checkout origin/main -- data/Subjects/<sid>
