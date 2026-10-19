"""Key facts prompt: the facts a tenant needs at a glance, each tied to its clause."""

VERSION = "key_facts-v2"

PROMPT = """Above are clauses of a rental agreement, each headed by its code in brackets, e.g. [C-04].
Extract these key facts where the clauses state them:
- owner: the landlord or licensor's name; tenant: the tenant or licensee's name
- property: the address of the rented property
- monthly_rent, security_deposit: amounts as written, e.g. "Rs. 18,000 per month"
- start_date, duration, lock_in
- notice_period_owner, notice_period_tenant: notice each side must give to end the agreement
- late_payment_penalty, early_exit_penalty, other_penalty (you may list several penalties)
- rent_increase
For each fact give a short value in plain words, and clause_code: the code of the clause that states it.
Leave out facts the clauses do not state. Do not guess.
Cite clauses only by their bracketed codes. The codes are not the clause numbers printed in the document: a clause printed as "8." may have the code [C-09]."""
