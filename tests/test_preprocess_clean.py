from rag.ingest.preprocess import clean_text


def test_clean_text_strips_html_tags():
    raw = "<p>Текст <b>важный</b> <img src='x'> и таблица <table><tr><td>1</td></tr></table></p>"
    cleaned = clean_text(raw)
    assert "<" not in cleaned
    assert "Текст важный" in cleaned


def test_clean_text_removes_pict_noise():
    raw = "Документ ?pict.jpg&oid=12345 содержит мусор"
    cleaned = clean_text(raw)
    assert "?pict" not in cleaned
    assert "Документ" in cleaned


def test_clean_text_normalizes_whitespace():
    raw = "Статья   1\n\n   О   налоге\n   \n  Продолжение"
    cleaned = clean_text(raw)
    assert "  " not in cleaned
    assert "Статья 1" in cleaned
