# Agent usage

Shimin Zhang and Junyang He used OpenAI Codex as a development assistant. During the recorded project work it helped implement and troubleshoot the pipeline, prepare cluster jobs, inspect logs and evaluation artifacts, build comparison viewers, and package the code. For this release Codex restored omitted Python modules, documented run commands, tested the CPU path in a fresh clone, and assembled the audit packet using the existing team report and saved experiment outputs.

The team specified the paired 3D-versus-video research question, corrected earlier baseline and evaluation choices, selected the comparison stages, and requested evaluation against the source motion and the MotionFix instruction. The final analysis uses the saved matched cohorts. Earlier pilot experiments used different protocols.

**Model use in evaluation.** GPT-6-Luna acted as a frozen semantic judge. It received the source, instruction and sampled candidate frames. Its success, failure and uncertain categories describe model judgments. The 100-case table uses its verdicts. The repository also includes a separate Qwen3-VL semantic evaluator. Neither judge trained the frozen tracking policy in the reported comparison.

**Checks and limitations.** The release smoke test and repository tests ran in a clean Python environment. The audit script recomputes table values from per-case outputs exported from the saved ICE reports. Neither check independently repeats checkpoint training, GPU execution, or VLM inference. The team retains responsibility for the submission.
