# Git

` git remote add upstream [git://github.com.....]` - add remote as upstream to pull from

`git pull upstream master`

`git submodule update --init --recursive` - pull all submodules after first git clone of a repository with submodules

`git submodule update --init --force --remote` - force update if init and update commands ignored for submodules

`git reset --hard HEAD~1` reset local merge that hasn't been pushed to any remote / or reset to hash of commit before merge

`git merge --no-ff` no fastforward when merging (even when possible), force merge commit creation