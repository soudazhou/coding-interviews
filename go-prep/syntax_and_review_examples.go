// Package prep contains commented snippets that highlight core Go syntax,
// idiomatic patterns, and common anti-patterns that are frequently discussed
// during interviews and pair-programming sessions.
package prep

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"
)

// -----------------------------------------------------------------------------
// SECTION 1: Basic declarations and zero values.
// -----------------------------------------------------------------------------

// User models a simple domain object. Struct fields start with an uppercase
// letter when they need to be exported from the package.
type User struct {
	ID        int
	Name      string
	Tags      []string
	CreatedAt time.Time
}

// NewUser showcases short variable declarations, composite literals, and
// how to return multiple values (a pointer and an error).
func NewUser(name string, tags []string) (*User, error) {
	// Anti-pattern: returning nil slices when the zero value works fine. The
	// zero value (nil) for slices is safe to range over, so we can accept the
	// caller passing nil without defensive allocation.
	name = strings.TrimSpace(name)
	if name == "" {
		return nil, fmt.Errorf("name must be provided")
	}

	// Defensive copy so the caller cannot mutate our internal slice. Copying
	// is cheap for small slices and avoids subtle bugs in interviews.
	copiedTags := append([]string(nil), tags...)

	// Struct literals allow field names for clarity.
	return &User{
		ID:        time.Now().Nanosecond(), // Demo-only: do not use as a real ID.
		Name:      name,
		Tags:      copiedTags,
		CreatedAt: time.Now().UTC(),
	}, nil
}

// DisplayName demonstrates value receivers and formatting helpers.
func (u User) DisplayName() string {
	// Prefer fmt.Sprintf for readability over manual concatenation when
	// building human-readable strings.
	return fmt.Sprintf("%s (#%d)", u.Name, u.ID)
}

// AddTag uses a pointer receiver so that we can mutate the struct in place.
func (u *User) AddTag(tag string) {
	// Anti-pattern: forgetting to check for nil receiver when pointer
	// receivers are exported. Here we guard against it defensively.
	if u == nil {
		return
	}

	// Anti-pattern: appending empty strings which bloat slices.
	tag = strings.TrimSpace(tag)
	if tag == "" {
		return
	}

	u.Tags = append(u.Tags, tag)
}

// -----------------------------------------------------------------------------
// SECTION 2: Interfaces, methods, and polymorphism.
// -----------------------------------------------------------------------------

// Greeter is a small interface illustrating implicit implementation.
type Greeter interface {
	Greet() string
}

// StaticGreeter implements Greeter without requiring explicit declarations.
type StaticGreeter struct {
	Prefix string
}

// Greet satisfies the Greeter interface.
func (g StaticGreeter) Greet() string {
	if g.Prefix == "" {
		return "Hello!"
	}
	return g.Prefix + "!"
}

// ProvideGreeting shows how interfaces enable polymorphism.
func ProvideGreeting(g Greeter) string {
	if g == nil {
		return "<nil greeter>"
	}
	return g.Greet()
}

// Anti-pattern: returning a pointer to an interface value (e.g. *Greeter) is
// almost never necessary and complicates nil checks. Return the interface
// directly as done above.

// -----------------------------------------------------------------------------
// SECTION 3: Errors and custom error types.
// -----------------------------------------------------------------------------

// ErrEmptyInput is a sentinel error value that can be checked with errors.Is.
var ErrEmptyInput = errors.New("input slice is empty")

// FilterAndDouble is an idiomatic example that works with slices, defers
// resource cleanup, and returns detailed errors.
func FilterAndDouble(nums []int, predicate func(int) bool) ([]int, error) {
	if len(nums) == 0 {
		return nil, ErrEmptyInput
	}
	if predicate == nil {
		return nil, fmt.Errorf("predicate is required")
	}

	// Anti-pattern: pre-allocating with len(nums) when you are filtering can
	// lead to partially-filled slices and requires tracking an index. Use
	// zero-length slices with capacity when possible.
	result := make([]int, 0, len(nums))
	for _, n := range nums {
		if predicate(n) {
			doubled := n * 2
			result = append(result, doubled)
		}
	}

	if len(result) == 0 {
		// Wrapping errors preserves context while allowing callers to
		// interrogate the cause with errors.Is / errors.As.
		return nil, fmt.Errorf("no values passed the predicate: %w", ErrEmptyInput)
	}

	return result, nil
}

// -----------------------------------------------------------------------------
// SECTION 4: Generics and higher-order functions.
// -----------------------------------------------------------------------------

// MapSlice is a generic helper that transforms a slice.
func MapSlice[T any, R any](values []T, transform func(T) R) []R {
	if len(values) == 0 {
		return nil
	}
	result := make([]R, 0, len(values))
	for _, v := range values {
		result = append(result, transform(v))
	}
	return result
}

// Reduce aggregates a slice into a single value using an accumulator.
func Reduce[T any, R any](values []T, seed R, accumulate func(R, T) R) R {
	acc := seed
	for _, v := range values {
		acc = accumulate(acc, v)
	}
	return acc
}

// Anti-pattern: using interface{} instead of generics in Go 1.18+ code. It
// loses type safety and requires manual casting.

// -----------------------------------------------------------------------------
// SECTION 5: Concurrency primitives (goroutines, channels, WaitGroup).
// -----------------------------------------------------------------------------

// SafeCounter illustrates protecting shared state with mutexes.
type SafeCounter struct {
	mu    sync.RWMutex
	value int
}

// Inc increments the counter safely.
func (c *SafeCounter) Inc() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.value++
}

// Value reads the counter using a read lock. Returning by value avoids
// exposing internal state.
func (c *SafeCounter) Value() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.value
}

// ConcurrentSum demonstrates spawning goroutines with context cancellation and
// proper channel closing.
func ConcurrentSum(ctx context.Context, inputs []int, workers int) (int, error) {
	if workers <= 0 {
		return 0, fmt.Errorf("workers must be > 0")
	}

	jobs := make(chan int)
	results := make(chan int)
	var wg sync.WaitGroup

	worker := func() {
		defer wg.Done()
		for {
			select {
			case <-ctx.Done():
				return
			case n, ok := <-jobs:
				if !ok {
					return
				}
				results <- n
			}
		}
	}

	wg.Add(workers)
	for i := 0; i < workers; i++ {
		go worker()
	}

	go func() {
		defer close(jobs)
		for _, n := range inputs {
			select {
			case <-ctx.Done():
				return
			case jobs <- n:
			}
		}
	}()

	go func() {
		wg.Wait()
		close(results)
	}()

	sum := 0
	for {
		select {
		case <-ctx.Done():
			return 0, ctx.Err()
		case n, ok := <-results:
			if !ok {
				return sum, nil
			}
			sum += n
		}
	}
}

// Anti-patterns for concurrency (do NOT copy these behaviors):
//   * Launching goroutines in tight loops without a WaitGroup or cancellation.
//   * Writing to a shared map without synchronization (maps are not safe for
//     concurrent writes).
//   * Forgetting to close channels when senders exit, causing receivers to
//     block forever.

// -----------------------------------------------------------------------------
// SECTION 6: Context propagation and resource cleanup.
// -----------------------------------------------------------------------------

// ExternalServiceCall simulates work with cancellation support.
func ExternalServiceCall(ctx context.Context, timeout time.Duration) error {
	// Always derive child contexts so callers can cancel.
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	select {
	case <-ctx.Done():
		// Returning context errors instead of panicking keeps APIs predictable.
		return ctx.Err()
	case <-time.After(10 * time.Millisecond):
		return nil
	}
}

// WithResource demonstrates deferring cleanup even when returning early.
type Resource struct {
	closed bool
}

// Close toggles the closed flag; imagine closing a file or network connection.
func (r *Resource) Close() error {
	r.closed = true
	return nil
}

// UseResource wraps acquiring and releasing a resource.
func UseResource() error {
	r := &Resource{}
	// Anti-pattern: forgetting to check the error from Close when it matters.
	defer func() {
		if err := r.Close(); err != nil {
			// In real code you might log the error. Avoid panicking.
		}
	}()

	// Do work with r here.
	if r == nil {
		return errors.New("resource unavailable")
	}

	return nil
}

// Anti-pattern: deferring Close on a nil resource causes a panic. Always ensure
// the resource was acquired successfully before deferring cleanup.

// -----------------------------------------------------------------------------
// SECTION 7: Panic and recover usage.
// -----------------------------------------------------------------------------

// SafeExecute shows how to convert a panic into an error for API boundaries.
func SafeExecute(fn func()) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("panic: %v", r)
		}
	}()

	fn()
	return nil
}

// Anti-pattern: using panic for expected error conditions. Prefer returning
// errors so callers can handle them gracefully.

// -----------------------------------------------------------------------------
// SECTION 8: Testing hooks (pure functions are easiest to test).
// -----------------------------------------------------------------------------

// SumSlice is deterministic and side-effect free, making it trivial to unit
// test. Interviewers often expect functions like this for whiteboard problems.
func SumSlice(nums []int) int {
	total := 0
	for _, n := range nums {
		total += n
	}
	return total
}

// Anti-pattern: writing functions with hidden dependencies (e.g., global
// variables or network calls) makes testing hard and is discouraged.

// -----------------------------------------------------------------------------
// SECTION 9: Slices and maps deep dive.
// -----------------------------------------------------------------------------

// CloneStrings showcases copying slices to avoid aliasing shared backing arrays.
func CloneStrings(values []string) []string {
	if len(values) == 0 {
		return nil
	}

	// Using append with a nil destination creates a copy while preserving
	// capacity. copy(dst, src) also works, but append reads cleaner.
	cloned := append([]string(nil), values...)
	return cloned
}

// CountOccurrences demonstrates the zero value of maps (nil) and proper
// initialization before assignment.
func CountOccurrences(values []string) map[string]int {
	counts := make(map[string]int, len(values))
	for _, v := range values {
		v = strings.TrimSpace(v)
		if v == "" {
			continue
		}
		counts[v]++
	}
	return counts
}

// MergeMaps shows how to combine maps while avoiding needless allocations.
func MergeMaps(dst, src map[string]int) map[string]int {
	if dst == nil {
		dst = make(map[string]int, len(src))
	}
	for k, v := range src {
		dst[k] += v
	}
	return dst
}

// Anti-patterns to remember:
//   * Taking the address of range variables (the variable is reused each loop).
//   * Re-slicing without copying when you need independent storage.
//   * Forgetting that deleting from a map while ranging is allowed, but adding
//     new keys during iteration can lead to unpredictable ordering.

// -----------------------------------------------------------------------------
// SECTION 10: Embedding and composition.
// -----------------------------------------------------------------------------

// Auditable embeds common metadata for domain objects.
type Auditable struct {
	CreatedAt time.Time
	UpdatedAt time.Time
}

// Touch updates timestamps in a reusable, testable way.
func (a *Auditable) Touch(now time.Time) {
	if a == nil {
		return
	}
	if a.CreatedAt.IsZero() {
		a.CreatedAt = now
	}
	a.UpdatedAt = now
}

// AdminUser composes User and Auditable via embedding.
type AdminUser struct {
	User
	Auditable
	Permissions []string
}

// Promote showcases pointer semantics for embedded structs.
func (a *AdminUser) Promote(now time.Time, newPermissions ...string) {
	if a == nil {
		return
	}
	a.Touch(now)
	a.Permissions = append(a.Permissions, newPermissions...)
}

// Anti-pattern: exporting embedded fields you do not want part of the public
// API. Prefer explicit accessor methods when you need stricter control.

// -----------------------------------------------------------------------------
// SECTION 11: Lazy initialization with sync.Once.
// -----------------------------------------------------------------------------

// LazyClient simulates expensive setup guarded by sync.Once.
type LazyClient struct {
	once    sync.Once
	client  *Resource
	initErr error
}

// getResource lazily initializes the client exactly once.
func (l *LazyClient) getResource() (*Resource, error) {
	if l == nil {
		return nil, errors.New("lazy client is nil")
	}

	l.once.Do(func() {
		// Imagine dialing a remote service here. For interviews, simply
		// reuse Resource to show the pattern without network code.
		l.client = &Resource{}
		// You could set initErr if something failed.
	})

	return l.client, l.initErr
}

// Close ensures the lazily created resource is cleaned up.
func (l *LazyClient) Close() error {
	if l == nil || l.client == nil {
		return nil
	}
	return l.client.Close()
}

// Anti-pattern: ignoring initErr leads to hidden initialization failures.

// -----------------------------------------------------------------------------
// SECTION 12: Timers, tickers, and select patterns.
// -----------------------------------------------------------------------------

// PollUntilSuccess illustrates using a ticker with context cancellation.
func PollUntilSuccess(ctx context.Context, interval time.Duration, check func() bool) error {
	if interval <= 0 {
		return fmt.Errorf("interval must be positive")
	}
	if check == nil {
		return fmt.Errorf("check function is required")
	}

	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			if check() {
				return nil
			}
		}
	}
}

// NonBlockingSend demonstrates a select with default to avoid blocking.
func NonBlockingSend(ch chan<- int, value int) bool {
	select {
	case ch <- value:
		return true
	default:
		return false
	}
}

// Anti-pattern: leaking goroutines waiting on a ticker when the context is
// done. Always stop tickers and exit select loops on cancellation signals.
