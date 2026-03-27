#!/bin/bash
FILTER_BRANCH_SQUELCH_WARNING=1
export FILTER_BRANCH_SQUELCH_WARNING
git filter-branch -f --env-filter '
GIT_AUTHOR_EMAIL=felipe.agro.terrazas@gmail.com
GIT_COMMITTER_EMAIL=felipe.agro.terrazas@gmail.com
GIT_AUTHOR_NAME="Felipe Ramires Terrazas"
GIT_COMMITTER_NAME="Felipe Ramires Terrazas"
export GIT_AUTHOR_EMAIL GIT_COMMITTER_EMAIL GIT_AUTHOR_NAME GIT_COMMITTER_NAME
' -- --all
