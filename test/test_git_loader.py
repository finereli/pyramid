"""Tests for the git log loader."""
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import pytest


# Test parse_git_log independently (no git repo needed)
def test_parse_git_log_basic():
    from loaders import parse_git_log, COMMIT_SEPARATOR

    raw = f"""
{COMMIT_SEPARATOR}
abc1234def5678
2026-02-19T10:00:00+00:00
Test Author
Add feature X

This adds feature X to the system.

diff --git a/foo.py b/foo.py
index 123..456 100644
--- a/foo.py
+++ b/foo.py
@@ -1,3 +1,5 @@
+import bar
 def hello():
-    pass
+    bar.do_thing()
+    return True

{COMMIT_SEPARATOR}
def9876abc5432
2026-02-19T11:00:00+00:00
Another Author
Fix bug in feature X

diff --git a/foo.py b/foo.py
--- a/foo.py
+++ b/foo.py
@@ -1,5 +1,5 @@
 import bar
 def hello():
-    bar.do_thing()
+    bar.do_thing(safely=True)
     return True
"""

    commits = parse_git_log(raw)
    assert len(commits) == 2

    assert commits[0]['hash'] == 'abc1234def5678'
    assert commits[0]['author'] == 'Test Author'
    assert 'Add feature X' in commits[0]['message']
    assert 'This adds feature X' in commits[0]['message']
    assert 'import bar' in commits[0]['diff']
    assert commits[0]['timestamp'] == '2026-02-19T10:00:00+00:00'

    assert commits[1]['hash'] == 'def9876abc5432'
    assert commits[1]['author'] == 'Another Author'
    assert 'Fix bug' in commits[1]['message']
    assert 'safely=True' in commits[1]['diff']


def test_parse_git_log_truncates_large_diffs():
    from loaders import parse_git_log, COMMIT_SEPARATOR, MAX_DIFF_CHARS

    huge_diff = "diff --git a/big.py b/big.py\n" + ("+" * (MAX_DIFF_CHARS + 1000))

    raw = f"""
{COMMIT_SEPARATOR}
abc123
2026-01-01T00:00:00+00:00
Author
Big change

{huge_diff}
"""

    commits = parse_git_log(raw)
    assert len(commits) == 1
    assert len(commits[0]['diff']) <= MAX_DIFF_CHARS + 50  # truncated message adds a bit
    assert '... [diff truncated]' in commits[0]['diff']


def test_parse_git_log_empty():
    from loaders import parse_git_log
    commits = parse_git_log('')
    assert commits == []


def test_load_git_commits_on_real_repo():
    """Test on the pyramid repo itself (we know it exists)."""
    from loaders import load_git_commits

    repo_path = str(Path(__file__).parent.parent)
    messages, info = load_git_commits(repo_path, limit=5)

    assert len(messages) > 0
    assert len(messages) <= 5
    assert 'Loaded' in info

    for msg in messages:
        assert msg['role'] == 'assistant'
        assert 'Commit:' in msg['content']
        assert 'Author:' in msg['content']
        assert 'Message:' in msg['content']
        assert msg['timestamp']


def test_load_git_commits_not_a_repo():
    from loaders import load_git_commits

    messages, info = load_git_commits('/tmp')
    assert messages == []
    assert 'Not a git repository' in info


def test_load_git_commits_to_message_format():
    """Verify the message format works with group_messages_by_week."""
    from loaders import load_git_commits, group_messages_by_week

    repo_path = str(Path(__file__).parent.parent)
    messages, _ = load_git_commits(repo_path, limit=10)

    by_week = group_messages_by_week(messages)
    assert len(by_week) > 0

    for week, week_msgs in by_week.items():
        assert week != 'unknown'
        for msg in week_msgs:
            assert 'timestamp' in msg
            assert 'content' in msg
            assert 'role' in msg


def test_load_git_incremental():
    """Test incremental loading returns only new commits."""
    from loaders import load_git_commits, load_git_incremental

    repo_path = str(Path(__file__).parent.parent)

    # Get all commits
    all_messages, _ = load_git_commits(repo_path)
    if len(all_messages) < 3:
        pytest.skip('Need at least 3 commits to test incremental')

    # Get the hash of the 3rd-to-last commit
    third_content = all_messages[-3]['content']
    hash_line = [l for l in third_content.split('\n') if l.startswith('Commit:')][0]
    third_hash = hash_line.split('Commit:')[1].strip()

    # Load incrementally since that commit
    messages, head, count = load_git_incremental(repo_path, third_hash)

    # Should get the last 2 commits (not including the one we specified)
    assert count == 2
    assert len(messages) == 2
