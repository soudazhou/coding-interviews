# Go Pair-Programming Interview Prep

This folder contains heavily-commented Go snippets that you can review before a
pair-programming or code-review interview. The examples in
[`syntax_and_review_examples.go`](syntax_and_review_examples.go) cover:

- Core syntax (structs, methods, interfaces, slices, maps, generics).
- Idiomatic patterns for error handling, resource cleanup, and concurrency.
- Common anti-patterns to avoid during a code review discussion.

You can format and compile the examples with the standard Go toolchain:

```bash
go fmt ./...
go test ./...
```

Both commands should succeed without additional setup because the module only
uses the Go standard library.
