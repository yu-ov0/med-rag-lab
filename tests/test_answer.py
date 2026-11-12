from generation.answer import answer_question, validate_citations

HIT = {
    "chunk_id": "hypertension_2024:p11:c0",
    "doc_id": "hypertension_2024",
    "page": 11,
    "text": "非同日3次测量诊室血压≥140/90 mmHg。",
    "score": 0.9,
}


def test_extractive_answer_has_citation():
    result = answer_question("诊室血压诊断界值？", [HIT])
    assert not result.refused
    assert result.citations[0]["page"] == 11
    assert "[1]" in result.answer


def test_refuses_when_below_threshold():
    result = answer_question("诊室血压诊断界值？", [HIT], threshold=0.95)
    assert result.refused
    assert result.reason == "below_threshold"


def test_refuses_without_candidates():
    result = answer_question("任意问题", [])
    assert result.refused
    assert result.reason == "no_candidates"


def test_invalid_citations_detected():
    assert validate_citations("结论[1]，另一结论[3]。", 2) == [3]
    assert validate_citations("结论[1][2]。", 2) == []
