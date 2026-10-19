"""Risk radar prompt: the model only detects patterns and amounts; the score is computed in code."""

VERSION = "risk-v2"

PROMPT = """Above are clauses of a rental agreement, each headed by its code in brackets, e.g. [C-04].
For every clause, return its clause_code and:
- category: parties_property, rent_payment, security_deposit, rent_escalation, maintenance, lock_in, termination_notice, penalty_late_fee, renewal, use_restrictions, dispute_resolution or other
- patterns: only those clearly present in the clause's own words (often none):
  lock_in_with_penalty: the tenant must pay a penalty or lose money for leaving during a fixed period
  auto_renewal: the agreement renews automatically unless someone acts
  one_sided_termination: one side can end the agreement much more easily than the other (shorter notice, no reason needed, or the other side cannot end it at all)
  broad_exclusion: the owner is excused from responsibility in wide terms
  unlimited_liability: the tenant is liable for losses without any limit
  distant_arbitration: disputes must be settled in a city far from the property
  short_reply_deadline: the tenant must respond or act within a very short time
  non_refundable_deposit: all or part of the deposit will not be returned
  unilateral_rent_increase: the owner can raise the rent alone, or the rent rises automatically by a fixed rate
- amount_at_stake_rupees: the largest rupee amount the tenant could lose or have to pay because of this clause (penalty, compensation, deposit at risk, yearly increase). Not the normal monthly rent. Null if none.
Also return monthly_rent_rupees: the monthly rent, or null if no clause states it.
Cite clauses only by their bracketed codes. The codes are not the clause numbers printed in the document: a clause printed as "8." may have the code [C-09]."""
