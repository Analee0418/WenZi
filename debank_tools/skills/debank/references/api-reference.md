# DeBank API Reference Notes

Official docs:

- OpenAPI welcome: https://docs.cloud.debank.com/en/readme/open-api
- API Pro reference: https://docs.cloud.debank.com/en/readme/api-pro-reference
- User endpoints: https://docs.cloud.debank.com/en/readme/api-pro-reference/user
- Units usage: https://docs.cloud.debank.com/en/readme/auxiliary-feature/units
- Error codes: https://docs.cloud.debank.com/en/readme/error-code

Base URL:

```text
https://pro-openapi.debank.com
```

Authentication:

```text
AccessKey: <DEBANK_ACCESS_KEY>
```

Common User endpoints:

```text
GET /v1/user/total_balance?id=<address>
GET /v1/user/used_chain_list?id=<address>
GET /v1/user/all_token_list?id=<address>&is_all=false
GET /v1/user/all_simple_protocol_list?id=<address>
GET /v1/user/all_complex_protocol_list?id=<address>
GET /v1/user/all_nft_list?id=<address>
GET /v1/user/token_authorized_list?id=<address>&chain_id=eth
GET /v1/user/history_list?id=<address>&chain_id=eth&page_count=20
GET /v1/account/units
```

Expected analysis patterns:

- Portfolio summary: total balance + used chains + sorted top tokens + sorted top protocols.
- DeFi position detail: complex protocol list, grouped by protocol and chain.
- Approval risk: token authorized list, highlight unlimited or very high allowances when available.
- History: always require a chain id; DeBank history endpoints are chain-scoped.

Crawler policy:

- Do not scrape DeBank pages by default.
- Scraping can break due to frontend changes, login/session requirements, rate limits, and anti-bot controls.
- Use crawling only when the user explicitly requests it and accepts instability.
