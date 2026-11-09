# Mac and Windows: one product and workflow

- macOS and Windows must share 100% of equivalent application and inference workflow logic. Reuse the existing implementation, entry points, decisions, state transitions, validation and error handling.
- Keep unavoidable OS operations in thin platform adapters behind the same interface. Compiler commands, GPU APIs, process/file APIs, package formats and signing mechanisms may differ; application behavior and workflow must remain shared.
- Preserve the existing Mac behavior. Do not change it merely to make a Windows port easier, or create a separate Windows implementation of shared logic.

## Every Mac step has a Windows counterpart in the same process

- This rule covers the whole lifecycle: dependency setup, development, testing, building, packaging, installation, updates, versioning, releases, publication and documentation. It applies to CI/CD as well as runtime code.
- Whenever a Mac workflow has a step, put the corresponding Windows step in that same workflow/process with the same purpose, inputs, outputs and success criteria. Reuse common orchestration; call platform-specific jobs or adapters only for OS mechanics.
- Before adding a Windows step, inspect and reuse the existing Mac step. When changing a shared stage, update both counterparts together. Do not create an independent Windows release process, numbering scheme or additional manual publication step.
- Each numbered `vX.Y.Z` engine release must build and check both platforms from the same tagged commit and publish their packages, manifests and installers in one release. Validate their common version and build identity, wait for both checks, and update the normal platform feeds through that shared release process.
- Do not create Windows-only GitHub releases or preview tags unless the user explicitly requests a separate channel. CI test artifacts do not need their own product release.
- Explain any unavoidable platform gap explicitly. Keep the difference at the OS adapter boundary; never silently invent a different workflow to bypass it.
- Verify shared behavior and the changed platform steps with appropriate checks. Report which platforms actually ran; shared source alone does not prove native execution.
