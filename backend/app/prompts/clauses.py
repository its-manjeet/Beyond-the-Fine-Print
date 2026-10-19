"""Clause extractor prompt: split the numbered lines of a rental agreement into clauses."""

VERSION = "clauses-v1"

PROMPT = """Above are the numbered lines of a rental agreement, written as "p<page> L<line>: text".
Split it into clauses, in document order. For each clause return:
- clause_code: C-01, C-02, ... in order
- title: a short title; use the document's own heading if it has one
- category: parties_property, rent_payment, security_deposit, rent_escalation, maintenance, lock_in, termination_notice, penalty_late_fee, renewal, use_restrictions, dispute_resolution or other
- span: page and line_start of the clause's first line (including its heading line), and line_end of its last line. Set page_end only if the clause continues onto a later page; then line_end is a line on page_end.
Treat the parties and property recitals at the top as one parties_property clause. Skip page numbers, signature and witness lines.
Use only line numbers that appear above. Do not copy the clause text."""

RETRY = """

Your previous answer had these problems:
{errors}
Return the full corrected list of clauses."""
