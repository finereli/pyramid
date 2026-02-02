import pytest
from generate import render_memory, CORE_MODELS, TIER_LABELS


def test_render_memory_with_content():
    assistant_content = "I am Glenn, an AI assistant."
    user_content = "Eli is a software engineer."
    other_models = []
    
    content = render_memory(assistant_content, user_content, other_models)
    
    assert '# Memory' in content
    assert '## Self' in content
    assert '## User' in content
    assert 'Glenn' in content
    assert 'Eli' in content
    assert '## Other Models' not in content


def test_render_memory_with_other_models():
    assistant_content = "Assistant content"
    user_content = "User content"
    other_models = [
        ({'description': 'Python project'}, 'models/python.md'),
        ({'description': 'Japan trip'}, 'models/japan-2025.md'),
    ]
    
    content = render_memory(assistant_content, user_content, other_models)
    
    assert '## Self' in content
    assert '## User' in content
    assert '## Other Models' in content
    assert 'models/python.md' in content
    assert 'models/japan-2025.md' in content


def test_render_memory_empty_content():
    content = render_memory('', '', [])
    
    assert '*No assistant observations yet.*' in content
    assert '*No user observations yet.*' in content


def test_core_models():
    assert 'assistant' in CORE_MODELS
    assert 'user' in CORE_MODELS


def test_tier_labels():
    assert TIER_LABELS[0] == 'Recent'
    assert TIER_LABELS[1] == 'This Month'
    assert TIER_LABELS[2] == 'Historical'
