"""
Automated unit tests for Assignment 8 core modules and fallback logic.
"""
import pytest
from app import (
    local_fallback_sentiment_analysis,
    local_fallback_keyword_extraction,
    mock_content_generation,
    mock_podcast_plan,
)

def test_local_sentiment_analysis_positive():
    sample = "Artificial Intelligence is a useful and helpful technology. It makes learning easier and provides students with quick feedback."
    result = local_fallback_sentiment_analysis(sample)
    assert result["sentiment"] == "Positive"
    assert result["score"] > 0
    assert "useful" in sample.lower()

def test_local_sentiment_analysis_negative():
    sample = "The tool has severe bugs, dangerous flaws, inaccurate results, and caused massive failure."
    result = local_fallback_sentiment_analysis(sample)
    assert result["sentiment"] == "Negative"
    assert result["score"] < 0

def test_keyword_extraction():
    sample = "Artificial Intelligence is a useful and helpful technology. It makes learning easier and provides students with quick feedback."
    keywords = local_fallback_keyword_extraction(sample, top_n=5)
    assert len(keywords) > 0
    words = [kw[0] for kw in keywords]
    assert any("intelligence" in w or "artificial" in w or "learning" in w for w in words)

def test_mock_content_generation():
    post = mock_content_generation("Social Media Post", "AI in Healthcare")
    assert "AI in Healthcare" in post
    assert "Title:" in post
    assert "Key Takeaway:" in post

    story = mock_content_generation("Short Story", "Cybersecurity")
    assert "Cybersecurity" in story
    assert "Breakthrough" in story

    poem = mock_content_generation("Poem", "Cloud Computing")
    assert "Cloud Computing" in poem

def test_mock_podcast_plan():
    plan = mock_podcast_plan("Blockchain")
    assert "Podcast Title:" in plan
    assert "Guest Type:" in plan
    assert "Interview Questions:" in plan
    assert "8." in plan

if __name__ == "__main__":
    pytest.main(["-v", __file__])
