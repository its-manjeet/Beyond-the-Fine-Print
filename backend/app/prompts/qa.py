"""Grounded Q&A prompt: answer only from the given clauses, with citations."""

VERSION = "qa-v2"

PROMPT = """Above are clauses of the user's rental agreement, each headed by its code in brackets, e.g. [C-04].
The user (usually the tenant) asks:
<question>
{question}
</question>
Answer using only what these clauses say. Do not use outside knowledge or say what the law is.
- If the clauses answer the question, set covered to true and write 1 to 4 short plain-English sentences (Grade 6-8 reading level), addressed to the user as "you". For each sentence give clause_codes: the codes it relies on.
- If the clauses do not answer it, set covered to false and return no sentences.
Never tell the user to sign or not to sign.
Cite clauses only by their bracketed codes. The codes are not the clause numbers printed in the document: a clause printed as "8." may have the code [C-09]."""

NOT_COVERED = (
    "Your document does not seem to answer this. You could ask the other party in writing, "
    "or get free help from legal aid (NALSA or your State or District Legal Services Authority)."
)
