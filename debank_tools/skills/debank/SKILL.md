---
name: debank
description: Use when the user asks about DeBank wallet/address data, EVM address portfolio analysis, token holdings, protocol or DeFi positions, NFT holdings, token approvals/allowances, transaction history, DeBank OpenAPI usage, or building tools around DeBank data. Prefer official DeBank Cloud OpenAPI with AccessKey; do not scrape DeBank pages unless the user explicitly requests it and the API cannot cover the task.
---

# DeBank

Use official DeBank Cloud OpenAPI for wallet and DeFi portfolio analysis. Use the bundled CLI whenever possible instead of hand-writing curl calls.

## Quick Workflow

1. Validate the input is an EVM address: `0x` plus 40 hex characters.
2. Check authentication:
   - Prefer `DEBANK_ACCESS_KEY` from the environment.
   - Never print or log the access key.
3. Pick the command:
   - Summary: `python scripts/debank_client.py summary <address>`
   - API units: `python scripts/debank_client.py units`
   - Tokens: `python scripts/debank_client.py tokens <address>`
   - Protocols: `python scripts/debank_client.py protocols <address>`
   - Complex protocol positions: `python scripts/debank_client.py protocols <address> --complex`
   - Approvals: `python scripts/debank_client.py approvals <address> --chain eth`
   - History: `python scripts/debank_client.py history <address> --chain eth`
4. Summarize results in Chinese by default for this user unless they ask otherwise.
5. State that DeBank data is third-party indexed data and may lag chain state.

## Data Source Rules

- Use `https://pro-openapi.debank.com` by default.
- Use header auth: `AccessKey: <key>`.
- Treat `401` as missing/invalid key, `403` as no units or forbidden, and `429` as rate limited.
- Avoid crawler/scraping by default. Only discuss scraping as a last-resort manual fallback after explaining instability and policy risk.
- For high-value or operational decisions, tell the user to verify on-chain or with another source.

## Reference

Read `references/api-reference.md` when endpoint choice, fields, units, or error handling matter.
