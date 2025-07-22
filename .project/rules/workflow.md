---
type: "always_apply"
---

## Development Workflow

### Branch Management
1. **Create a new branch** before starting development work defined (normally defined by a PRP)
2. **Define tasks** in detail before implementation
3. **Follow TDD workflow** for all new code

### TDD Cycle Steps
1. **Write a Test**:
   - Start by writing a test for a new feature or functionality
   - Test should define the expected behavior

2. **Run the Test**:
   - Execute the test suite
   - Verify the new test fails (since feature isn't implemented yet)

3. **Write Code**:
   - Implement the minimum code necessary to make the test pass
   - Focus on simplicity and meeting test requirements

4. **Run the Test Again**:
   - Execute the test suite again
   - Verify the new test now passes

5. **Refactor**:
   - Clean up the code while ensuring all tests still pass
   - Improve structure, remove duplication, enhance readability

6. **Repeat**:
   - Return to step 1 for the next feature or requirement

### Commit Procedure
1. **Commit After Each Cycle**:
   - Commit after each complete TDD cycle. **One commit per test-implementation cycle** (not per feature)
   - Makes it easier to track changes and understand evolution

2. **Commit Messages**:
   - Write clear, descriptive commit messages using conventional commits format. Don't mention tests in the commit message, only in the body.
   - Explain the implementation (not the tests) What was added/changed, why you chose to implement it this way, and how it works (what's the logic)

3. **Example Commit Message**:
   ```
   feat(services): add database service implementation

   - Created test for document storage functionality
   - Implemented storage logic to pass the test
   - Refactored for better error handling
   - Used Pydantic validation for type safety
   - I implemented it this way because ...
   - The way it works is ...
   ```

4. **Mark Task as Done** after completing the full cycle

---

## Anti-Patterns to Avoid
- ❌ Don't move code without understanding dependencies (especially Neo4j/knowledge graph)
- ❌ Don't create circular imports between modules
- ❌ Don't put business logic in tool files (keep tools as thin wrappers)
- ❌ Don't skip writing tests "to save time" (especially for knowledge graph features)
- ❌ Don't use synchronous Supabase calls in async functions without proper handling
- ❌ Don't forget to validate all inputs with Pydantic models (all 16 tools)
- ❌ Don't ignore Neo4j connection failures (graceful degradation required)
- ❌ Don't leave temporary files uncleaned (repository analysis cleanup)
- ❌ Don't hardcode file paths (use Path objects and proper configuration)
- ❌ Don't mix knowledge graph logic with core RAG functionality
- ❌ Don't break the existing MCP tool interface (16 tools must work identically)