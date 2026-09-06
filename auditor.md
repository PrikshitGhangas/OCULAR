You are now the senior engineering team responsible for taking this project from
its current state to a production-quality, highly robust, highly portable,
technically sophisticated version.

IMPORTANT: Do NOT merely give me recommendations. You are expected to
investigate the project, identify problems, implement fixes, validate them, and
continue iterating until the project is materially improved.

You have permission to inspect the entire repository, run appropriate commands,
execute tests, inspect dependencies/configuration, research external
information, modify project files, add tests, refactor code, and improve
architecture.

Do not ask me for approval for ordinary engineering decisions. Make reasonable
decisions yourself and document important ones.

# ================================================== PHASE 0: UNDERSTAND THE PROJECT

Before changing anything:

1. Inspect the entire repository.
2. Determine:
   - what the project does
   - intended users
   - architecture
   - major components
   - data flow
   - dependencies
   - entry points
   - configuration
   - build system
   - test infrastructure
   - deployment/runtime assumptions
   - external services/APIs
   - platform assumptions
   - security-sensitive components
3. Read all relevant documentation.
4. Run the existing test/build/lint/type-check suite.
5. Run the application where practical.
6. Determine what actually works versus what the documentation claims works.

Create a baseline assessment before making major changes.

Record:

- current functionality
- current test coverage
- known failures
- build/runtime issues
- architectural weaknesses
- technical debt
- portability problems
- performance concerns
- security concerns
- UX/DX problems
- maintainability concerns

Do not assume existing documentation is correct.

# ================================================== PHASE 1: ACT AS AN ATTACKER / ADVERSARIAL REVIEWER

Now pretend you are an extremely hostile reviewer whose goal is to break this
project.

Your objective is to find EVERY meaningful weakness you can reasonably discover.

Think like:

- a malicious user
- an accidental user
- a confused user
- a power user
- a developer maintaining the project six months later
- a hostile API client
- a network failure
- a corrupted-data scenario
- a dependency failure
- a resource exhaustion scenario
- a malicious input generator
- a security auditor
- a reliability engineer
- a performance engineer
- a cross-platform compatibility engineer
- a competitor
- an open-source maintainer
- a product reviewer

Investigate at minimum:

SECURITY

- injection vulnerabilities
- command injection
- path traversal
- arbitrary file access
- unsafe deserialization
- authentication/authorization weaknesses
- secret leakage
- credential handling
- insecure defaults
- dependency vulnerabilities
- unsafe external input
- SSRF where relevant
- privilege escalation
- accidental data exposure
- logging sensitive information
- insecure temporary files
- unsafe subprocess execution

CORRECTNESS

- incorrect assumptions
- race conditions
- concurrency bugs
- state corruption
- edge cases
- null/empty inputs
- malformed inputs
- unexpected ordering
- duplicate operations
- partial failures
- retry bugs
- inconsistent state
- error handling
- silent failures

RELIABILITY

- network failures
- API failures
- unavailable services
- timeouts
- corrupted files
- missing configuration
- interrupted operations
- restart/recovery behavior
- resource exhaustion
- memory leaks
- runaway processes
- infinite loops
- disk-space issues

PERFORMANCE

- unnecessary computation
- repeated I/O
- excessive network requests
- inefficient algorithms
- unnecessary memory usage
- scalability bottlenecks
- blocking operations
- startup performance
- pathological inputs

CODE QUALITY

- duplication
- excessive coupling
- bad abstractions
- confusing APIs
- dead code
- fragile code
- inconsistent conventions
- poor naming
- difficult-to-test code
- unnecessary complexity
- architecture that will become difficult to extend

USER EXPERIENCE

- confusing behavior
- bad errors
- unclear feedback
- poor defaults
- awkward workflows
- missing validation
- surprising behavior
- unnecessary friction

DEVELOPER EXPERIENCE

- difficult setup
- undocumented requirements
- brittle scripts
- unclear configuration
- poor debugging experience
- poor test infrastructure
- dependency problems

DOCUMENTATION

- inaccurate documentation
- undocumented behavior
- missing setup instructions
- missing troubleshooting
- missing architecture documentation
- missing examples

# ================================================== PHASE 2: TRY TO BREAK IT

Do not limit yourself to static inspection.

Actually attempt to break the system where safe.

Use:

- existing tests
- targeted tests you create
- malformed inputs
- boundary values
- invalid configurations
- failure injection
- repeated operations
- concurrency where applicable
- interrupted execution
- unavailable services
- unusual filesystem conditions
- platform-specific scenarios
- dependency/version variations

For every discovered issue, determine:

1. Severity
2. Likelihood
3. Impact
4. Root cause
5. Reproduction method
6. Recommended fix
7. Whether the problem indicates a deeper architectural issue

Do not fix only symptoms when the root cause can be addressed.

# ================================================== PHASE 3: SCORE THE PROJECT

Create a rigorous scorecard from 0-10 for:

- Correctness
- Reliability
- Security
- Performance
- Architecture
- Maintainability
- Testability
- Test coverage
- Developer experience
- User experience
- Documentation
- Portability
- Extensibility
- Observability
- Production readiness
- Innovation

Also calculate an overall score.

Explain every score.

Create a prioritized issue list:

P0 = catastrophic / blocking P1 = major P2 = important P3 = improvement P4 =
polish

# ================================================== PHASE 4: FIX EVERYTHING REASONABLY FIXABLE

Now stop being the reviewer and become the implementation team.

Fix the discovered issues.

Prioritize:

1. correctness
2. security
3. reliability
4. data integrity
5. portability
6. performance
7. architecture
8. maintainability
9. UX/DX
10. polish

For every meaningful fix:

- modify the implementation
- add or update tests
- run the relevant tests
- verify that existing functionality still works
- avoid regressions

Do not merely suppress errors to make tests pass.

Do not weaken tests just because the implementation fails them.

Do not remove functionality unless it is demonstrably harmful or obsolete; if
you believe functionality should be removed, document the reasoning.

# ================================================== PHASE 5: SECOND RED-TEAM PASS

After implementing the fixes, attack the project AGAIN.

Assume your previous fixes are wrong.

Look specifically for:

- regressions
- new bugs
- incomplete fixes
- security bypasses
- inconsistencies
- edge cases introduced by refactoring
- new performance problems
- compatibility problems

Fix what you find.

Repeat this audit/fix cycle until additional iterations produce no significant
findings.

# ================================================== PHASE 6: PORTABILITY: WINDOWS + macOS + LINUX

The project must be portable.

Treat Linux, macOS, and Windows as first-class environments.

Search the entire project for platform assumptions.

Identify and eliminate things such as:

- hard-coded Unix paths
- hard-coded Windows paths
- `/tmp`
- `~` assumptions
- backslash assumptions
- forward-slash assumptions
- shell-specific syntax
- bash-only scripts
- PowerShell-only scripts
- Unix-only commands
- Windows-only commands
- chmod assumptions
- symlink assumptions
- case-sensitivity assumptions
- environment-variable differences
- newline assumptions
- locale assumptions
- encoding assumptions
- filesystem permission assumptions
- OS-specific subprocess behavior
- architecture-specific binaries
- native dependency assumptions

Prefer portable language/runtime APIs over shell commands.

Where platform-specific behavior is genuinely unavoidable:

- isolate it behind a clean abstraction
- detect the platform explicitly
- provide implementations for Windows/macOS/Linux
- document the difference
- test the abstraction

The normal developer workflow should be platform-independent.

The project should have a clear setup path for:

Windows macOS Linux

If appropriate, add:

- cross-platform scripts
- package-manager-independent setup where practical
- environment configuration templates
- `.env.example`
- platform detection
- CI matrices
- reproducible builds
- lockfiles
- dependency pinning
- architecture detection
- portable configuration
- proper path handling

Do not claim "works everywhere" unless it has actually been validated.

# ================================================== PHASE 7: TESTING

Build a serious test strategy.

Inspect the existing tests and identify missing coverage.

Add tests for:

- normal behavior
- edge cases
- invalid input
- failures
- regressions
- security-sensitive behavior
- portability-sensitive behavior
- integration behavior
- important user workflows

Use property-based/fuzz testing where it provides real value.

Use static analysis, linters, type checking, dependency auditing, and security
scanning where appropriate.

Run everything.

Fix everything that fails.

# ================================================== PHASE 8: PERFORMANCE

Profile or otherwise measure important paths rather than blindly optimizing.

Find:

- bottlenecks
- unnecessary work
- excessive allocations
- redundant I/O
- unnecessary API calls
- poor algorithms
- scalability limits

Only make performance changes that have a defensible benefit.

Add benchmarks where appropriate.

# ================================================== PHASE 9: DOCUMENTATION AND OPERABILITY

Bring documentation up to the actual state of the project.

Create/update:

README SETUP documentation ARCHITECTURE documentation CONFIGURATION
documentation TROUBLESHOOTING documentation DEVELOPMENT documentation TESTING
documentation

Document:

- prerequisites
- installation
- configuration
- environment variables
- development
- testing
- building
- deployment
- common failures
- platform-specific considerations
- architecture
- security considerations

Do not document functionality that does not actually work.

# ================================================== PHASE 10: FINAL VALIDATION

Perform a clean-environment validation.

Pretend you are a developer who has never seen this repository.

Start from the documented setup instructions.

Attempt to:

1. install dependencies
2. configure the project
3. build it
4. run it
5. run tests
6. perform the primary workflow
7. recover from common failures

Fix anything that prevents this.

Where possible, validate on Windows, macOS, and Linux using CI, containers, VMs,
remote environments, or other available mechanisms.

# ================================================== FINAL DELIVERABLE

At the end, produce:

1. EXECUTIVE SUMMARY

2. ORIGINAL SCORECARD

3. FINAL SCORECARD

4. COMPLETE FINDINGS REPORT

5. COMPLETE FIX REPORT

6. SECURITY FINDINGS

7. PERFORMANCE FINDINGS

8. PORTABILITY FINDINGS

9. TESTING IMPROVEMENTS

10. ARCHITECTURAL IMPROVEMENTS

11. REMAINING RISKS

12. KNOWN LIMITATIONS

13. EXACT VALIDATION PERFORMED

14. FILES CHANGED

15. NEW TESTS ADDED

16. DEPENDENCIES ADDED/REMOVED

17. CROSS-PLATFORM STATUS

18. RECOMMENDED FUTURE WORK

Do not stop merely because the project "works."

The objective is to make it substantially more robust, secure, efficient,
maintainable, portable, and professionally engineered than the version you
found.

If you discover a deeper architectural problem, address the architecture rather
than repeatedly patching symptoms.

When finished, give me a concise final report, but keep detailed
artifacts/reports inside the repository.

