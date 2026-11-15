# Test fixtures

`village_2026-11-14.json` is the USC Village menu for Saturday 14 Nov 2026 from
`hospitality.usc.edu/wp-json/hsp-api/v1/get-res-dining-menus/university-village?y=2026&m=11&d=14`.

It was transcribed from the live response through a web-fetch tool, not saved
byte for byte, so key order and whitespace differ from the original. To replace
it with an exact copy, run this on a machine that can reach the API and copy
the cached file over the fixture:

    python -m menu --date 2026-11-14
    cp data/menus/university-village/2026-11-14.json tests/fixtures/village_2026-11-14.json

If the tests then fail, the transcription missed something: fix the rules, not the file.
