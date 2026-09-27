# PII labeling policy

Label `1` (PII) when the text contains information that identifies, contacts, or is sensitive about a **specific private individual**:

- A private person's name in a personal, medical, legal, financial or HR context (a name on its own, in a sensitive setting, counts)
- Contact details for a person: personal email, phone number, home address
- Government or account identifiers: SSN/TIN, passport, driver license, bank/card/IBAN numbers, medical record numbers, employee IDs
- Date of birth, IP address, username or device ID tied to a person
- Credentials: passwords, API keys, tokens
- Combinations of quasi-identifiers that single out one person, even without a name ("the only female cardiologist at the Boise clinic, born in 1971")

Label `0` (clean) for everything else, including the lookalikes that make up `false_positives`:

- Public figures, historical people, or fictional characters in their public or literary role
- Organization contact points (`support@`, main office numbers, headquarters addresses)
- Redacted or placeholder values (`[REDACTED]`, `XXX-XX-XXXX`, `<email>`)
- Numbers that look like identifiers but aren't personal: ISBNs, SKUs, order numbers, version strings, error codes, timestamps, coordinates of landmarks
- Discussion of PII as a topic, or schemas and code that name PII fields without containing any values
- Roles without identities ("the patient", "a customer in Ohio")

All names, numbers and addresses in the positive examples are invented. Ambiguous cases are left out rather than guessed.

## Files

| File | Label | What it tests |
|---|---|---|
| `pii_easy.json` | 1 | Explicit, labeled identifiers in standard formats |
| `pii_medium.json` | 1 | Identifiers in natural prose, varied formats, sensitive context |
| `pii_hard.json` | 1 | Obfuscated, spelled out, embedded in code/logs, non-English, quasi-identifiers, buried in long text |
| `clean_easy.json` | 0 | Ordinary text with no PII |
| `false_positives.json` | 0 | Text that looks like PII but isn't (hard negatives) |

Each sample carries a `category` field. The loaders ignore it (they read only `text` and `label`), but it is there for per-category evaluation.
