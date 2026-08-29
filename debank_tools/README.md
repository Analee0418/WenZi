# debank-tools

Local DeBank OpenAPI helper tools for Codex skills and later WenZi plugin work.

Use:

```bash
export DEBANK_ACCESS_KEY=...
python cli/debank_client.py summary 0x...
python cli/debank_client.py units
```

The CLI only uses official DeBank OpenAPI endpoints. It does not scrape DeBank web pages.
