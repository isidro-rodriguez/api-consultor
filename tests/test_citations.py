from api_consultor.citations import Citations, check_citations, doc_id


def test_doc_id_is_zero_padded() -> None:
    assert doc_id(1) == "DOC-01"
    assert doc_id(12) == "DOC-12"


def test_valid_citations_are_sorted_and_deduplicated() -> None:
    result = check_citations("Usa X [DOC-03] y Y [DOC-01][DOC-03].", 3)
    assert result == Citations(valid=(1, 3), invalid=())
    assert result.any


def test_out_of_range_citations_are_invalid() -> None:
    result = check_citations("[DOC-00] [DOC-07] [DOC-07] [DOC-02]", 2)
    assert result.valid == (2,)
    assert result.invalid == ("DOC-00", "DOC-07")


def test_grouped_citations_are_detected() -> None:
    assert check_citations("[DOC-01, DOC-02]", 2).valid == (1, 2)


def test_no_citations() -> None:
    result = check_citations("Sin citas. Ver DOCS-1 o doc-01.", 5)
    assert result == Citations((), ())
    assert not result.any
