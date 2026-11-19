# Operator workspace audit

The audit used local Chromium at 1400×960 and 390×844, with the original application captured before UI changes. Plain screenshots cover the empty states. Browser interactions exercise actual model inference and downloads.

1. **Find the input.** The [previous desktop layout](01-baseline-desktop.png) displayed seven configuration controls before the upload surface. The [previous phone layout](02-baseline-mobile.png) squeezed the settings and checkboxes into narrow columns, pushing upload out of view. The [current desktop](03-workspace-desktop.png) puts media and preview next to each other. The [current phone view](04-workspace-mobile.png) stacks them and keeps processing settings collapsed.
2. **Submit and recover.** An empty submission shows a clear next action. A batch with two valid images and one corrupt image completes both valid inputs and names the failed file. GPU inference progress identifies frame counts. Stop requests preserve completed frames and end with a cancellation state.
3. **Review evidence.** Observations and vehicle tracks have separate tables. Selecting a row exposes the frame or track status with OCR evidence and reasons. Search and status filters apply to the on-screen preview. The UI states when the dataset exceeds the preview.
4. **Take results away.** The browser flow exposes complete ZIP plus JSON and CSV. Each job has a separate work directory, preventing same-name uploads from replacing a prior job's downloads.

The audit checked viewport sizing, recovery, result sections, and browser interactions. It did not include a screen reader or an accessibility compliance assessment. Live camera reporting was exercised only as an unavailable device case.
