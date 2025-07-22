---
type: "always_apply"
---

# Unit Testing Guidelines for Coding LLMs

## Core Principles

### 1. FIRST-U Rules
- **Fast**: Tests must run quickly (under 200ms typically)
- **Isolated/Independent**: Tests cannot depend on other tests or external state
- **Repeatable**: Same result every time, regardless of environment
- **Self-validating**: Clear pass/fail without manual interpretation
- **Timely**: Write tests during development, not after
- **Understandable**: Descriptive names and clear purpose

### 2. Critical Rule: Tests Must Actually Fail
- **Every test MUST have a realistic failure scenario**
- Test both success cases AND failure cases
- Example: For password validation requiring 8+ chars with symbols:
  - ✅ Test "secret01" (should fail)
  - ✅ Test "Secret123!" (should pass)

### 3. Test Structure: Arrange-Act-Assert
```
// Arrange: Set up test data and expected state
// Act: Execute the method under test
// Assert: Verify actual vs expected result
```

### 4. What to Test (Priority Order)
1. **Complex code with few dependencies** - High value, low cost
2. **Trivial code with few dependencies** - Easy to test, just do it
3. **Complex code with many dependencies** - Break down first, then test
4. **Trivial coordinators with many dependencies** - Skip or minimal testing

### 5. Essential Test Requirements
- **Meaningful test names**: `MethodName_StateUnderTest_ExpectedBehavior`
- **One assertion per logical concept**
- **Test edge cases and boundaries**
- **Mock external dependencies** (databases, APIs, files)
- **Verify exceptions are thrown when expected**

### 6. Critical Validation Checklist
Before accepting any test:
- [ ] Can this test actually fail with incorrect implementation?
- [ ] Does it test real behavior, not just code execution?
- [ ] Are test inputs realistic and meaningful?
- [ ] Is the expected outcome specific and verifiable?
- [ ] Does it follow Arrange-Act-Assert pattern?

### 7. Common Anti-Patterns to Avoid
- Tests that can never fail (e.g., `Assert.IsTrue(true)`)
- Tests dependent on other tests
- Tests requiring manual setup
- Vague assertions that don't verify specific behavior
- Testing implementation details instead of behavior

### 8. Focus Areas
- **Business logic and algorithms** (high priority)
- **Input validation and edge cases**
- **Error handling and exceptions**
- **Boundary conditions**
- **Critical user workflows**

Remember: Well-written tests are assets; badly written tests are burdens. Every test should have clear value and be able to catch real bugs.