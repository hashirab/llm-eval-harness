from evalharness.metrics import extract_entities, fact_found, hallucination, key_fact_recall

SOURCE = (
    "Host web-prod-03 logged 1,412 failed SSH logins from 203.0.113.45 at 02:11. "
    "A script was downloaded from update-check.example.net. Related to CVE-2021-44228."
)


def test_fact_found_verbatim():
    assert fact_found("203.0.113.45", "Attacker IP was 203.0.113.45.").found


def test_fact_found_rephrased_by_token_coverage():
    result = fact_found("successful login as deploy", "There was a successful deploy login.")
    assert result.found
    assert result.coverage >= 0.8


def test_fact_not_found():
    result = fact_found("IP blocked at the firewall", "The account was disabled.")
    assert not result.found


def test_key_fact_recall_score():
    facts = ["203.0.113.45", "web-prod-03", "account disabled", "firewall block"]
    out = "Brute force from 203.0.113.45 against web-prod-03; the account was disabled."
    recall = key_fact_recall(out, facts)
    assert recall.score == 0.75
    assert [f.fact for f in recall.facts if not f.found] == ["firewall block"]


def test_empty_key_facts_scores_full():
    assert key_fact_recall("anything", []).score == 1.0


def test_extract_entities_kinds():
    kinds = {e.kind for e in extract_entities(SOURCE)}
    assert {"ipv4", "domain", "time", "number", "cve"} <= kinds


def test_ip_is_not_double_counted_as_numbers():
    entities = extract_entities("Traffic from 198.51.100.23 only.")
    assert [(e.kind, e.value) for e in entities] == [("ipv4", "198.51.100.23")]


def test_hash_detected():
    sha256 = "a" * 64
    entities = extract_entities(f"hash {sha256}")
    assert entities[0].kind == "hash"


def test_no_hallucination_when_entities_are_in_source():
    out = "1,412 failed logins from 203.0.113.45 on web-prod-03 at 02:11 (CVE-2021-44228)."
    result = hallucination(out, SOURCE)
    assert result.rate == 0.0
    assert result.total_entities > 0


def test_invented_entities_are_flagged():
    out = "Logins came from 203.0.113.99 and 1,412 attempts were made via evil.example.org."
    result = hallucination(out, SOURCE)
    flagged = {e.value for e in result.unsupported}
    assert flagged == {"203.0.113.99", "evil.example.org"}
    assert result.rate == 2 / 3


def test_number_formatting_is_tolerated():
    assert hallucination("There were 1412 failures.", SOURCE).rate == 0.0


def test_no_entities_means_zero_rate():
    result = hallucination("An attacker tried to log in.", SOURCE)
    assert result.rate == 0.0 and result.total_entities == 0
