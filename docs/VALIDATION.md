# Portfolio validation â€” 22 October 2026

The local portfolio build was checked before the recorded synthetic walkthrough.

- Python: 8 passed, 1 skipped using `python -m pytest SENTINEL-2.5D/tests -q`.
- The skipped test is the opt-in CUDA runtime parity check. It was not enabled.
- Frontend: `npm run build` passed compilation, TypeScript and static page generation.
- Browser: verified live synthetic streaming, grid / sampled point-cloud / comparison views, and representation-size analysis. The captured demo was configured for a 5 FPS target and recorded at 4 screenshots per second; the MP4 repeats captures to encode at 24 FPS. It contains no voiceover.
- Fixed the Python 3.10 timeout exception boundary so a slow client is removed without stopping healthy clients. A regression test verifies that behavior.
- Replaced fixed comparison-panel performance, memory and tactical claims with frame-derived sizes and explicit ideal-link assumptions.
- No weights, datasets or toolchains were downloaded, and no real-scan ML inference or fresh GPU benchmark was performed.

The 60-second video combines title/end cards with actual captured UI sequences. It uses synthetic fixture labels in CPU mode. The short GIF is a reduced-size excerpt; neither is model-accuracy evidence.
