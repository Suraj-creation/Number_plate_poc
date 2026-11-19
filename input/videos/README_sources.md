# Local video acceptance samples

These clips were downloaded for local pipeline QA on 2026-11-19. Pexels allows free use and modification under its [Pexels License](https://www.pexels.com/license/). They contain no ground-truth plate boxes or verified plate text, so they are suitable for integration and throughput checks, not accuracy scoring. The separate original downloads were removed after processing; only the normalized, concatenated test inputs and annotated outputs remain locally.

## Day and general traffic compilation

`india_traffic_acceptance_2m14s_3fps.mp4` combines five distinct clips. Each was resized to fit 1920×1080, converted to 3 fps, had audio removed, then concatenated. The resulting input is 134.002 seconds and 402 frames. The pipeline was run at stride 1, so all 402 frames of this prepared input were inferred and annotated.

| Pexels source | Original duration | Original video |
| --- | ---: | --- |
| [Traffic and pedestrians by CST, Mumbai](https://www.pexels.com/video/traffic-in-the-city-1193769/) | 21.48 s | 1920×1080, 25 fps |
| [Busy Mumbai street with heavy traffic](https://www.pexels.com/video/busy-mumbai-street-with-heavy-traffic-30230719/) | 61.97 s | 2560×1440, 30 fps |
| [Vibrant busy street in Mumbai](https://www.pexels.com/video/vibrant-busy-street-scene-in-mumbai-india-30279179/) | 12.20 s | 2560×1440, 30 fps |
| [Busy urban traffic in Mumbai](https://www.pexels.com/video/busy-urban-traffic-scene-in-mumbai-city-30400506/) | 30.93 s | 2560×1440, 30 fps |
| [Busy Indian city street traffic](https://www.pexels.com/video/busy-urban-traffic-on-indian-city-street-30608961/) | 7.47 s | 2560×1440, 30 fps |

## Night traffic compilation

`india_night_traffic_validation_53s_3fps.mp4` combines the two clips below, normalized to 1920×1080 at 3 fps with audio removed. It is 53.001 seconds and 159 frames; all frames were inferred at stride 1.

| Pexels source | Original duration | Original video |
| --- | ---: | --- |
| [Night street with cars and motorcycles](https://www.pexels.com/video/a-night-view-of-a-busy-street-with-cars-and-motorcycles-17812439/) | 34.77 s | 2560×1440, 29.97 fps |
| [Night market traffic in India](https://www.pexels.com/video/busy-urban-night-market-with-traffic-34793603/) | 18.30 s | 1920×1080, 60 fps |

The 3 fps versions deliberately reduce the source temporal sampling for a practical multi-minute GPU integration run. Results do not establish performance at the original 25–60 fps, nor Indian plate accuracy.
