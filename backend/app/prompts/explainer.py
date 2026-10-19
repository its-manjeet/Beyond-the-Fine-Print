"""Explainer prompt: 1-3 plain-English sentences per clause, each citing its clauses."""

VERSION = "explainer-v2"

PROMPT = """Above are clauses of a rental agreement, each headed by its code in brackets, e.g. [C-04].
For each clause, write 1 to 3 short sentences in plain English for the tenant ("you"), at a Grade 6-8 reading level: short words, one idea per sentence.
Say what the clause means in practice: amounts, dates, who pays, and what happens if something goes wrong.
For each sentence give clause_codes: the codes of the clauses it relies on (usually just that clause).
Use only what the clauses say. Do not say whether a term is legal or fair. Never tell the reader to sign or not to sign.
Cite clauses only by their bracketed codes. The codes are not the clause numbers printed in the document: a clause printed as "8." may have the code [C-09]."""
